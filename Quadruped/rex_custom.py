import math
import time
import gym
import pybullet
import numpy as np
import pybullet_data
from gym import spaces
from gym.utils import seeding
from pkg_resources import parse_version

from rex_gym.model import rex, motor
from rex_gym.util import bullet_client

NUM_MOTORS = 12
MOTOR_ANGLE_OBSERVATION_INDEX = 0
MOTOR_VELOCITY_OBSERVATION_INDEX = MOTOR_ANGLE_OBSERVATION_INDEX + NUM_MOTORS
MOTOR_TORQUE_OBSERVATION_INDEX = MOTOR_VELOCITY_OBSERVATION_INDEX + NUM_MOTORS
BASE_ORIENTATION_OBSERVATION_INDEX = MOTOR_TORQUE_OBSERVATION_INDEX + NUM_MOTORS
ACTION_EPS = 0.01
OBSERVATION_EPS = 0.01
RENDER_HEIGHT = 360
RENDER_WIDTH = 480
SENSOR_NOISE_STDDEV = rex.SENSOR_NOISE_STDDEV
DEFAULT_URDF_VERSION = "default"
NUM_SIMULATION_ITERATION_STEPS = 300

REX_URDF_VERSION_MAP = {
    DEFAULT_URDF_VERSION: rex.Rex
}


def convert_to_list(obj):
    try:
        iter(obj)
        return obj
    except TypeError:
        return [obj]


from Quadruped.QuadrupedController import *



class RexGymEnv(gym.Env):
    """The gym environment for Rex.

  It simulates the locomotion of Rex, a quadruped robot. The state space
  include the angles, velocities and torques for all the motors and the action
  space is the desired motor angle for each motor. The reward function is based
  on how far Rex walks in 1000 steps and penalizes the energy
  expenditure.

  """
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 100}

    def __init__(self,
                 urdf_root=pybullet_data.getDataPath(),
                 urdf_version=None,
                 distance_weight=1.0,
                 energy_weight=0.005,
                 shake_weight=0.0,
                 drift_weight=2.0,
                 distance_limit=float("inf"),
                 observation_noise_stdev=SENSOR_NOISE_STDDEV,
                 self_collision_enabled=True,
                 motor_velocity_limit=np.inf,
                 pd_control_enabled=False,
                 leg_model_enabled=True,
                 accurate_motor_model_enabled=False,
                 remove_default_joint_damping=False,
                 motor_kp=1.0,
                 motor_kd=0.02,
                 control_latency=0.0,
                 pd_latency=0.0,
                 torque_control_enabled=False,
                 motor_overheat_protection=False,
                 hard_reset=True,
                 on_rack=False,
                 render=True,
                 num_steps_to_log=1000,
                 action_repeat=1,
                 control_time_step=None,
                 env_randomizer=None,
                 forward_reward_cap=float("inf"),
                 reflection=True,
                 log_path=None):
        """Initialize the rex gym environment.

    Args:
      urdf_root: The path to the urdf data folder.
      urdf_version: [DEFAULT_URDF_VERSION] are allowable
        versions. If None, DEFAULT_URDF_VERSION is used.
      distance_weight: The weight of the distance term in the reward.
      energy_weight: The weight of the energy term in the reward.
      shake_weight: The weight of the vertical shakiness term in the reward.
      drift_weight: The weight of the sideways drift term in the reward.
      distance_limit: The maximum distance to terminate the episode.
      observation_noise_stdev: The standard deviation of observation noise.
      self_collision_enabled: Whether to enable self collision in the sim.
      motor_velocity_limit: The velocity limit of each motor.
      pd_control_enabled: Whether to use PD controller for each motor.
      leg_model_enabled: Whether to use a leg motor to reparameterize the action
        space.
      accurate_motor_model_enabled: Whether to use the accurate DC motor model.
      remove_default_joint_damping: Whether to remove the default joint damping.
      motor_kp: proportional gain for the accurate motor model.
      motor_kd: derivative gain for the accurate motor model.
      control_latency: It is the delay in the controller between when an
        observation is made at some point, and when that reading is reported
        back to the Neural Network.
      pd_latency: latency of the PD controller loop. PD calculates PWM based on
        the motor angle and velocity. The latency measures the time between when
        the motor angle and velocity are observed on the microcontroller and
        when the true state happens on the motor. It is typically (0.001-
        0.002s).
      torque_control_enabled: Whether to use the torque control, if set to
        False, pose control will be used.
      motor_overheat_protection: Whether to shutdown the motor that has exerted
        large torque (OVERHEAT_SHUTDOWN_TORQUE) for an extended amount of time
        (OVERHEAT_SHUTDOWN_TIME). See ApplyAction() in rex.py for more
        details.
      hard_reset: Whether to wipe the simulation and load everything when reset
        is called. If set to false, reset just place Rex back to start
        position and set its pose to initial configuration.
      on_rack: Whether to place Rex on rack. This is only used to debug
        the walking gait. In this mode, Rex's base is hanged midair so
        that its walking gait is clearer to visualize.
      render: Whether to render the simulation.
      num_steps_to_log: The max number of control steps in one episode that will
        be logged. If the number of steps is more than num_steps_to_log, the
        environment will still be running, but only first num_steps_to_log will
        be recorded in logging.
      action_repeat: The number of simulation steps before actions are applied.
      control_time_step: The time step between two successive control signals.
      env_randomizer: An instance (or a list) of EnvRandomizer(s). An
        EnvRandomizer may randomize the physical property of rex, change
          the terrrain during reset(), or add perturbation forces during step().
      forward_reward_cap: The maximum value that forward reward is capped at.
        Disabled (Inf) by default.
      log_path: The path to write out logs. For the details of logging, refer to
        rex_logging.proto.
    Raises:
      ValueError: If the urdf_version is not supported.
    """
        # Set up logging.
        self._log_path = log_path
        # @TODO fix logging
        self.logging = None
        # PD control needs smaller time step for stability.
        if control_time_step is not None:
            self.control_time_step = control_time_step
            self._action_repeat = action_repeat
            self._time_step = control_time_step / action_repeat
        else:
            # Default values for time step and action repeat
            if accurate_motor_model_enabled or pd_control_enabled:
                self._time_step = 0.002
                self._action_repeat = 5
            else:
                self._time_step = 0.01
                self._action_repeat = 1
            self.control_time_step = self._time_step * self._action_repeat
        # TODO: Fix the value of self._num_bullet_solver_iterations.
        self._num_bullet_solver_iterations = int(NUM_SIMULATION_ITERATION_STEPS / self._action_repeat)
        self._urdf_root = urdf_root
        self._self_collision_enabled = self_collision_enabled
        self._motor_velocity_limit = motor_velocity_limit
        self._observation = []
        self._true_observation = []
        self._objectives = []
        self._objective_weights = [distance_weight, energy_weight, drift_weight, shake_weight]
        self._env_step_counter = 0
        self._num_steps_to_log = num_steps_to_log
        self._is_render = render
        self._last_base_position = [0, 0, 0]
        self._last_base_orientation = [0, 0, 0, 1]
        self._distance_weight = distance_weight
        self._energy_weight = energy_weight
        self._drift_weight = drift_weight
        self._shake_weight = shake_weight
        self._distance_limit = distance_limit
        self._observation_noise_stdev = observation_noise_stdev
        self._action_bound = 1
        self._pd_control_enabled = pd_control_enabled
        self._leg_model_enabled = leg_model_enabled
        self._accurate_motor_model_enabled = accurate_motor_model_enabled
        self._remove_default_joint_damping = remove_default_joint_damping
        self._motor_kp = motor_kp
        self._motor_kd = motor_kd
        self._torque_control_enabled = torque_control_enabled
        self._motor_overheat_protection = motor_overheat_protection
        self._on_rack = on_rack
        self._cam_dist = 1.0
        self._cam_yaw = 0
        self._cam_pitch = -30
        self._forward_reward_cap = forward_reward_cap
        self._hard_reset = True
        self._last_frame_time = 0.0
        self._control_latency = control_latency
        self._pd_latency = pd_latency
        self._urdf_version = urdf_version
        self._ground_id = None
        self._reflection = reflection
        self._env_randomizers = convert_to_list(env_randomizer) if env_randomizer else []
        # @TODO fix logging
        self._episode_proto = None
        if self._is_render:
            self._pybullet_client = bullet_client.BulletClient(connection_mode=pybullet.GUI)
        else:
            self._pybullet_client = bullet_client.BulletClient()
        if self._urdf_version is None:
            self._urdf_version = DEFAULT_URDF_VERSION
        self._pybullet_client.setPhysicsEngineParameter(enableConeFriction=0)
        self.seed()
        self.reset()
        observation_high = (self._get_observation_upper_bound() + OBSERVATION_EPS)
        observation_low = (self._get_observation_lower_bound() - OBSERVATION_EPS)
        action_dim = NUM_MOTORS
        action_high = np.array([self._action_bound] * action_dim)
        self.action_space = spaces.Box(-action_high, action_high)
        self.observation_space = spaces.Box(observation_low, observation_high)
        self.viewer = None
        self._hard_reset = hard_reset  # This assignment need to be after reset()
        self.goal_reached = False

    def close(self):
        if self._env_step_counter > 0:
            self.logging.save_episode(self._episode_proto)
        self.rex.Terminate()

    def add_env_randomizer(self, env_randomizer):
        self._env_randomizers.append(env_randomizer)

    def reset(self, initial_motor_angles=None, reset_duration=1.0):
        self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_RENDERING, 0)
        # @TODO fix logging
        # if self._env_step_counter > 0:
        #     self.logging.save_episode(self._episode_proto)
        # self._episode_proto = rex_logging_pb2.RexEpisode()
        # rex_logging.preallocate_episode_proto(self._episode_proto, self._num_steps_to_log)
        if self._hard_reset:
            self._pybullet_client.resetSimulation()
            self._pybullet_client.setPhysicsEngineParameter(
                numSolverIterations=int(self._num_bullet_solver_iterations))
            self._pybullet_client.setTimeStep(self._time_step)
            self._ground_id = self._pybullet_client.loadURDF("%s/plane.urdf" % self._urdf_root)
            if self._reflection:
                self._pybullet_client.changeVisualShape(self._ground_id, -1, rgbaColor=[1, 1, 1, 0.8])
                self._pybullet_client.configureDebugVisualizer(
                    self._pybullet_client.COV_ENABLE_PLANAR_REFLECTION, self._ground_id)
            self._pybullet_client.setGravity(0, 0, -10)
            acc_motor = self._accurate_motor_model_enabled
            motor_protect = self._motor_overheat_protection
            if self._urdf_version not in REX_URDF_VERSION_MAP:
                raise ValueError("%s is not a supported urdf_version." % self._urdf_version)
            else:
                self.rex = (REX_URDF_VERSION_MAP[self._urdf_version](
                    pybullet_client=self._pybullet_client,
                    action_repeat=self._action_repeat,
                    urdf_root=self._urdf_root,
                    time_step=self._time_step,
                    self_collision_enabled=self._self_collision_enabled,
                    motor_velocity_limit=self._motor_velocity_limit,
                    pd_control_enabled=self._pd_control_enabled,
                    accurate_motor_model_enabled=acc_motor,
                    remove_default_joint_damping=self._remove_default_joint_damping,
                    motor_kp=self._motor_kp,
                    motor_kd=self._motor_kd,
                    control_latency=self._control_latency,
                    pd_latency=self._pd_latency,
                    observation_noise_stdev=self._observation_noise_stdev,
                    torque_control_enabled=self._torque_control_enabled,
                    motor_overheat_protection=motor_protect,
                    on_rack=self._on_rack))
        self.rex.Reset(reload_urdf=False,
                       default_motor_angles=initial_motor_angles,
                       reset_time=reset_duration)

        # Loop over all env randomizers.
        for env_randomizer in self._env_randomizers:
            env_randomizer.randomize_env(self)

        self._pybullet_client.setPhysicsEngineParameter(enableConeFriction=0)
        self._env_step_counter = 0
        self._last_base_position = [0, 0, 0]
        self._last_base_orientation = [0, 0, 0, 1]
        self._objectives = []
        self._pybullet_client.resetDebugVisualizerCamera(self._cam_dist, self._cam_yaw,
                                                         self._cam_pitch, [0, 0, 0])
        self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_RENDERING, 1)
        return self._get_observation()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _transform_action_to_motor_command(self, action):
        if self._leg_model_enabled:
            for i, action_component in enumerate(action):
                if not (-self._action_bound - ACTION_EPS <= action_component <=
                        self._action_bound + ACTION_EPS):
                    raise ValueError("{}th action {} out of bounds.".format(i, action_component))
            action = self.rex.ConvertFromLegModel(action)
        return action

    def step(self, action):
        """Step forward the simulation, given the action.

    Args:
      action: A list of desired motor angles for eight motors.

    Returns:
      observations: The angles, velocities and torques of all motors.
      reward: The reward for the current state-action pair.
      done: Whether the episode has ended.
      info: A dictionary that stores diagnostic information.

    Raises:
      ValueError: The action dimension is not the same as the number of motors.
      ValueError: The magnitude of actions is out of bounds.
    """
        self._last_base_position = self.rex.GetBasePosition()
        self._last_base_orientation = self.rex.GetBaseOrientation()
        # print("ACTION:")
        # print(action)
        if self._is_render:
            # Sleep, otherwise the computation takes less time than real time,
            # which will make the visualization like a fast-forward video.
            time_spent = time.time() - self._last_frame_time
            self._last_frame_time = time.time()
            time_to_sleep = self.control_time_step - time_spent
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)
            base_pos = self.rex.GetBasePosition()
            # Keep the previous orientation of the camera set by the user.
            [yaw, pitch, dist] = self._pybullet_client.getDebugVisualizerCamera()[8:11]
            self._pybullet_client.resetDebugVisualizerCamera(dist, yaw, pitch, base_pos)

        for env_randomizer in self._env_randomizers:
            env_randomizer.randomize_step(self)

        action = self._transform_action_to_motor_command(action)
        self.rex.Step(action)
        reward = self._reward()
        done = self._termination()
        self._env_step_counter += 1
        if done:
            self.rex.Terminate()
        return np.array(self._get_observation()), reward, done, {}

    def render(self, mode="rgb_array", close=False):
        if mode != "rgb_array":
            return np.array([])
        base_pos = self.rex.GetBasePosition()
        view_matrix = self._pybullet_client.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=base_pos,
            distance=self._cam_dist,
            yaw=self._cam_yaw,
            pitch=self._cam_pitch,
            roll=0,
            upAxisIndex=2)
        proj_matrix = self._pybullet_client.computeProjectionMatrixFOV(fov=60,
                                                                       aspect=float(RENDER_WIDTH) /
                                                                              RENDER_HEIGHT,
                                                                       nearVal=0.1,
                                                                       farVal=100.0)
        (_, _, px, _, _) = self._pybullet_client.getCameraImage(
            width=RENDER_WIDTH,
            height=RENDER_HEIGHT,
            renderer=self._pybullet_client.ER_BULLET_HARDWARE_OPENGL,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix)
        rgb_array = np.array(px)
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def get_rex_motor_angles(self):
        """Get the rex's motor angles.

    Returns:
      A numpy array of motor angles.
    """
        return np.array(self._observation[MOTOR_ANGLE_OBSERVATION_INDEX:MOTOR_ANGLE_OBSERVATION_INDEX +
                                                                        NUM_MOTORS])

    def get_rex_motor_velocities(self):
        """Get the rex's motor velocities.

    Returns:
      A numpy array of motor velocities.
    """
        return np.array(
            self._observation[MOTOR_VELOCITY_OBSERVATION_INDEX:MOTOR_VELOCITY_OBSERVATION_INDEX +
                                                               NUM_MOTORS])

    def get_rex_motor_torques(self):
        """Get the rex's motor torques.

    Returns:
      A numpy array of motor torques.
    """
        return np.array(
            self._observation[MOTOR_TORQUE_OBSERVATION_INDEX:MOTOR_TORQUE_OBSERVATION_INDEX +
                                                             NUM_MOTORS])

    def get_rex_base_orientation(self):
        """Get the rex's base orientation, represented by a quaternion.

    Returns:
      A numpy array of rex's orientation.
    """
        return np.array(self._observation[BASE_ORIENTATION_OBSERVATION_INDEX:])

    def is_fallen(self):
        """Decide whether Rex has fallen.

    If the up directions between the base and the world is larger (the dot
    product is smaller than 0.85) or the base is very low on the ground
    (the height is smaller than 0.13 meter), rex is considered fallen.

    Returns:
      Boolean value that indicates whether rex has fallen.
    """
        orientation = self.rex.GetBaseOrientation()
        rot_mat = self._pybullet_client.getMatrixFromQuaternion(orientation)
        local_up = rot_mat[6:]
        pos = self.rex.GetBasePosition()
        #  or pos[2] < 0.13
        return (np.dot(np.asarray([0, 0, 1]), np.asarray(local_up)) < 0.85)

    def _termination(self):
        position = self.rex.GetBasePosition()
        # distance = math.sqrt(position[0] ** 2 + position[1] ** 2)
        # print("POSITION")
        # print(position)
        #if self.is_fallen():
        #    print("IS FALLEN!")
        o = self.rex.GetBaseOrientation()
        #if o[1] < -0.13:
        #    print("IS ROTATING!")
        # if position[2] <= 0.12:
        #     print("LOW POSITION")
        # or position[2] <= 0.12
        return self.is_fallen() or o[1] < -0.13

    def _reward(self):
        current_base_position = self.rex.GetBasePosition()
        # side_penality = -abs(current_base_position[1])
        # forward direction
        forward_reward = -current_base_position[0] + self._last_base_position[0]
        # target_reward = 0.0
        # if forward_reward >0:
        #     target_reward = (-current_base_position[0] / 3)
        # Cap the forward reward if a cap is set.
        forward_reward = min(forward_reward, self._forward_reward_cap)
        # Penalty for sideways translation.
        drift_reward = -abs(current_base_position[1] - self._last_base_position[1])
        # Penalty for sideways rotation of the body.
        orientation = self.rex.GetBaseOrientation()
        rot_matrix = pybullet.getMatrixFromQuaternion(orientation)
        local_up_vec = rot_matrix[6:]
        shake_reward = -abs(np.dot(np.asarray([1, 1, 0]), np.asarray(local_up_vec)))
        energy_reward = -np.abs(
            np.dot(self.rex.GetMotorTorques(),
                   self.rex.GetMotorVelocities())) * self._time_step
        objectives = [forward_reward, energy_reward, drift_reward, shake_reward]
        weighted_objectives = [o * w for o, w in zip(objectives, self._objective_weights)]
        reward = sum(weighted_objectives)
                 # - side_penality
        self._objectives.append(objectives)
        # print("REWARD:")
        # print(reward)
        return reward

    def get_objectives(self):
        return self._objectives

    @property
    def objective_weights(self):
        """Accessor for the weights for all the objectives.

    Returns:
      List of floating points that corresponds to weights for the objectives in
      the order that objectives are stored.
    """
        return self._objective_weights

    def _get_observation(self):
        """Get observation of this environment, including noise and latency.

    rex class maintains a history of true observations. Based on the
    latency, this function will find the observation at the right time,
    interpolate if necessary. Then Gaussian noise is added to this observation
    based on self.observation_noise_stdev.

    Returns:
      The noisy observation with latency.
    """

        observation = []
        observation.extend(self.rex.GetMotorAngles().tolist())
        observation.extend(self.rex.GetMotorVelocities().tolist())
        observation.extend(self.rex.GetMotorTorques().tolist())
        observation.extend(list(self.rex.GetBaseOrientation()))
        self._observation = observation
        return self._observation

    def _get_true_observation(self):
        """Get the observations of this environment.

    It includes the angles, velocities, torques and the orientation of the base.

    Returns:
      The observation list. observation[0:8] are motor angles. observation[8:16]
      are motor velocities, observation[16:24] are motor torques.
      observation[24:28] is the orientation of the base, in quaternion form.
    """
        observation = []
        observation.extend(self.rex.GetTrueMotorAngles().tolist())
        observation.extend(self.rex.GetTrueMotorVelocities().tolist())
        observation.extend(self.rex.GetTrueMotorTorques().tolist())
        observation.extend(list(self.rex.GetTrueBaseOrientation()))

        self._true_observation = observation
        return self._true_observation

    def _get_observation_upper_bound(self):
        """Get the upper bound of the observation.

    Returns:
      The upper bound of an observation. See GetObservation() for the details
        of each element of an observation.
    """
        upper_bound = np.zeros(self._get_observation_dimension())
        num_motors = self.rex.num_motors
        upper_bound[0:num_motors] = math.pi  # Joint angle.
        upper_bound[num_motors:2 * num_motors] = (motor.MOTOR_SPEED_LIMIT)  # Joint velocity.
        upper_bound[2 * num_motors:3 * num_motors] = (motor.OBSERVED_TORQUE_LIMIT)  # Joint torque.
        upper_bound[3 * num_motors:] = 1.0  # Quaternion of base orientation.
        return upper_bound

    def _get_observation_lower_bound(self):
        """Get the lower bound of the observation."""
        return -self._get_observation_upper_bound()

    def _get_observation_dimension(self):
        """Get the length of the observation list.

    Returns:
      The length of the observation list.
    """
        return len(self._get_observation())

    if parse_version(gym.__version__) < parse_version('0.9.6'):
        _render = render
        _reset = reset
        _seed = seed
        _seed = seed
        _step = step

    def set_time_step(self, control_step, simulation_step=0.001):
        """Sets the time step of the environment.

    Args:
      control_step: The time period (in seconds) between two adjacent control
        actions are applied.
      simulation_step: The simulation time step in PyBullet. By default, the
        simulation step is 0.001s, which is a good trade-off between simulation
        speed and accuracy.
    Raises:
      ValueError: If the control step is smaller than the simulation step.
    """
        if control_step < simulation_step:
            raise ValueError("Control step should be larger than or equal to simulation step.")
        self.control_time_step = control_step
        self._time_step = simulation_step
        self._action_repeat = int(round(control_step / simulation_step))
        self._num_bullet_solver_iterations = (NUM_SIMULATION_ITERATION_STEPS / self._action_repeat)
        self._pybullet_client.setPhysicsEngineParameter(
            numSolverIterations=self._num_bullet_solver_iterations)
        self._pybullet_client.setTimeStep(self._time_step)
        self.rex.SetTimeSteps(action_repeat=self._action_repeat, simulation_step=self._time_step)

    @property
    def pybullet_client(self):
        return self._pybullet_client

    @property
    def ground_id(self):
        return self._ground_id

    @ground_id.setter
    def ground_id(self, new_ground_id):
        self._ground_id = new_ground_id

    @property
    def env_step_counter(self):
        return self._env_step_counter



class CustomRexEnv(RexGymEnv):
    """The gym environment for the rex.

  It simulates the locomotion of a rex, a quadruped robot. The state space
  include the angles, velocities and torques for all the motors and the action
  space is the desired motor angle for each motor. The reward function is based
  on how far the rex walks in 1000 steps and penalizes the energy
  expenditure.

  """
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 66}

    def __init__(self,
                 urdf_version=None,
                 control_time_step=0.006,
                 action_repeat=6,
                 control_latency=0,
                 pd_latency=0,
                 on_rack=False,
                 motor_kp=1.0,
                 motor_kd=0.02,
                 remove_default_joint_damping=False,
                 render=True,
                 num_steps_to_log=1000,
                 env_randomizer=None,
                 log_path=None):
        """Initialize the rex alternating legs gym environment.

    Args:
      urdf_version: [DEFAULT_URDF_VERSION, DERPY_V0_URDF_VERSION] are allowable
        versions. If None, DEFAULT_URDF_VERSION is used. Refer to
        rex_gym_env for more details.
      control_time_step: The time step between two successive control signals.
      action_repeat: The number of simulation steps that an action is repeated.
      control_latency: The latency between get_observation() and the actual
        observation. See minituar.py for more details.
      pd_latency: The latency used to get motor angles/velocities used to
        compute PD controllers. See rex.py for more details.
      on_rack: Whether to place the rex on rack. This is only used to debug
        the walking gait. In this mode, the rex's base is hung midair so
        that its walking gait is clearer to visualize.
      motor_kp: The P gain of the motor.
      motor_kd: The D gain of the motor.
      remove_default_joint_damping: Whether to remove the default joint damping.
      render: Whether to render the simulation.
      num_steps_to_log: The max number of control steps in one episode. If the
        number of steps is over num_steps_to_log, the environment will still
        be running, but only first num_steps_to_log will be recorded in logging.
      env_randomizer: An instance (or a list) of EnvRanzomier(s) that can
        randomize the environment during when env.reset() is called and add
        perturbations when env.step() is called.
      log_path: The path to write out logs. For the details of logging, refer to
        rex_logging.proto.
    """
        self.env_type = "turn"
        self.walk = WalkController()
        self.turn = TurnController()
        self.env_types = {0: "walk", 1: "turn"}

        super(CustomRexEnv,
              self).__init__(urdf_version=urdf_version,
                             accurate_motor_model_enabled=True,
                             motor_overheat_protection=True,
                             hard_reset=False,
                             motor_kp=motor_kp,
                             motor_kd=motor_kd,
                             remove_default_joint_damping=remove_default_joint_damping,
                             control_latency=control_latency,
                             pd_latency=pd_latency,
                             on_rack=on_rack,
                             render=render,
                             num_steps_to_log=num_steps_to_log,
                             env_randomizer=env_randomizer,
                             log_path=log_path,
                             control_time_step=control_time_step,
                             action_repeat=action_repeat)

        if self.env_type == "walk":
            action_dim = 12
            action_high = np.array([0.1] * action_dim)
            self.action_space = spaces.Box(-action_high, action_high)
            self._cam_dist = 1.0
            self._cam_yaw = 30
            self._cam_pitch = -30

        elif self.env_type == "turn":
            action_dim = 12
            action_high = np.array([0.1] * action_dim)
            self.action_space = spaces.Box(-action_high, action_high)
            self._cam_dist = 1.0
            self._cam_yaw = 30
            self._cam_pitch = -30
            self.last_step = 0
            self.stand = False
            self.brake = False
            self.t_orient = [0.0, 0.0, 0.0]
            if self._on_rack:
                self._cam_pitch = 0

    def reset(self):
        if self.env_type == "walk":
            self.desired_pitch = self.walk.DESIRED_PITCH
            super(CustomRexEnv, self).reset()
            return self._get_observation()
        elif self.env_type == "turn":
            self.desired_pitch = 0
            self.goal_reached = False
            self.stand = False
            super(CustomRexEnv, self).reset()
            position = self.rex.init_position
            if not self._on_rack:
                self.random_start_angle = np.random.uniform(0.2, 5.8)
            else:
                self.random_start_angle = 2.1
                position = self.rex.init_on_rack_position
            q = self.pybullet_client.getQuaternionFromEuler([0, 0, self.random_start_angle])
            self.pybullet_client.resetBasePositionAndOrientation(self.rex.quadruped, position, q)
            return self._get_observation()

    def _convert_from_leg_model(self, leg_pose, t=None):
        if self.env_type == "walk":
            motor_pose = np.zeros(NUM_MOTORS)
            for i in range(self.walk.NUM_LEGS):
                if i % 2 == 0:
                    motor_pose[3 * i] = 0.1
                else:
                    motor_pose[3 * i] = -0.1
                motor_pose[3 * i + 1] = leg_pose[3 * i + 1]
                motor_pose[3 * i + 2] = leg_pose[3 * i + 2]
            return motor_pose
        elif self.env_type == "turn":
            if t < .4:
                # set init position
                return self.rex.INIT_POSES['stand_high']
            motor_pose = np.zeros(NUM_MOTORS)
            for i in range(self.turn.NUM_LEGS):
                if self.goal_reached:
                    return self.rex.initial_pose
                else:
                    motor_pose[3 * i] = leg_pose[3 * i]
                    motor_pose[3 * i + 1] = leg_pose[3 * i + 1]
                    motor_pose[3 * i + 2] = leg_pose[3 * i + 2]
            return motor_pose

    def _signal(self, t):
        if self.env_type == "walk":
            initial_pose = self.rex.initial_pose
            period = self.walk.STEP_PERIOD
            l_extension = 0.2 * math.cos(2 * math.pi / period * t)
            l_swing = -l_extension
            extension = 0.3 * math.cos(2 * math.pi / period * t)
            swing = -extension
            pose = np.array([0, l_extension, extension,
                             0, l_swing, swing,
                             0, l_swing, swing,
                             0, l_extension, extension])

            signal = initial_pose + pose
            return signal
        elif self.env_type == "turn":
            initial_pose = self.rex.INIT_POSES['stand_high']
            period = self.turn.STEP_PERIOD
            extension = 0.1
            swing = 0.2
            swipe = 0.1
            ith_leg = int(t / period) % 2

            pose = {
                'left_0': np.array([swipe, extension, -swing,
                                    -swipe, extension, swing,
                                    swipe, -extension, swing,
                                    -swipe, -extension, -swing]),
                'left_1': np.array([-swipe, 0, swing,
                                    swipe, 0, -swing,
                                    -swipe, 0, -swing,
                                    swipe, 0, swing]),
                'right_0': np.array([-swipe, extension, -swing,
                                     swipe, -extension, swing,
                                     -swipe, -extension, swing,
                                     swipe, -extension, -swing]),
                'right_1': np.array([swipe, 0, swing,
                                     -swipe, 0, -swing,
                                     swipe, 0, -swing,
                                     -swipe, 0, swing])
            }

            if self.random_start_angle > 1:
                # turn left
                first_leg = pose['left_0']
                second_leg = pose['left_1']
            else:
                # turn right
                first_leg = pose['right_0']
                second_leg = pose['right_1']

            if ith_leg:
                signal = initial_pose + second_leg
            else:
                signal = initial_pose + first_leg
            return signal

    def _transform_action_to_motor_command(self, action):
        if self.env_type == "walk":
            action += self._signal(self.rex.GetTimeSinceReset())
            action = self._convert_from_leg_model(action)
            return action
        elif self.env_type == "turn":
            t = self.rex.GetTimeSinceReset()
            action += self._signal(t)
            action = self._convert_from_leg_model(action, t)
            return action

    def is_fallen(self):
        """Decide whether the rex has fallen.

        If the up directions between the base and the world is large (the dot
        product is smaller than 0.85), the rex is considered fallen.

        Returns:
          Boolean value that indicates whether the rex has fallen.
        """
        if self.env_type == "walk":
            orientation = self.rex.GetBaseOrientation()
            rot_mat = self._pybullet_client.getMatrixFromQuaternion(orientation)
            local_up = rot_mat[6:]
            return np.dot(np.asarray([0, 0, 1]), np.asarray(local_up)) < 0.85
        elif self.env_type == "turn":
            orientation = self.rex.GetBaseOrientation()
            rot_mat = self._pybullet_client.getMatrixFromQuaternion(orientation)
            local_up = rot_mat[6:]
            return np.dot(np.asarray([0, 0, 1]), np.asarray(local_up)) < 0.85

    def _get_true_observation(self):
        """Get the true observations of this environment.

    It includes the roll, the error between current pitch and desired pitch,
    roll dot and pitch dot of the base.

    Returns:
      The observation list.
    """
        if self.env_type == "walk":
            observation = []
            roll, pitch, _ = self.rex.GetTrueBaseRollPitchYaw()
            roll_rate, pitch_rate, _ = self.rex.GetTrueBaseRollPitchYawRate()
            observation.extend([roll, pitch, roll_rate, pitch_rate])
            self._observation = observation
            return self._true_observation
        elif self.env_type == "turn":
            observation = []
            roll, pitch, _ = self.rex.GetTrueBaseRollPitchYaw()
            roll_rate, pitch_rate, _ = self.rex.GetTrueBaseRollPitchYawRate()
            observation.extend([roll, pitch, roll_rate, pitch_rate])
            observation[1] -= self.desired_pitch  # observation[1] is the pitch
            self._observation = observation
            return self._true_observation

    def _get_observation(self):
        if self.env_type == "walk":
            observation = []
            roll, pitch, _ = self.rex.GetBaseRollPitchYaw()
            roll_rate, pitch_rate, _ = self.rex.GetBaseRollPitchYawRate()
            observation.extend([roll, pitch, roll_rate, pitch_rate])
            self._observation = observation
            return self._observation
        elif self.env_type == "turn":
            observation = []
            roll, pitch, _ = self.rex.GetBaseRollPitchYaw()
            roll_rate, pitch_rate, _ = self.rex.GetBaseRollPitchYawRate()
            observation.extend([roll, pitch, roll_rate, pitch_rate])
            observation[1] -= self.desired_pitch  # observation[1] is the pitch
            self._observation = observation
            return self._observation

    def _get_observation_upper_bound(self):
        """Get the upper bound of the observation.

    Returns:
      The upper bound of an observation. See GetObservation() for the details
        of each element of an observation.
    """
        if self.env_type == "walk":
            upper_bound = np.zeros(self._get_observation_dimension())
            upper_bound[0:2] = 2 * math.pi  # Roll, pitch, yaw of the base.
            upper_bound[2:4] = 2 * math.pi / self._time_step  # Roll, pitch, yaw rate.
            return upper_bound
        elif self.env_type == "turn":
            upper_bound = np.zeros(self._get_observation_dimension())
            upper_bound[0:2] = 2 * math.pi  # Roll, pitch, yaw of the base.
            upper_bound[2:4] = 2 * math.pi / self._time_step  # Roll, pitch, yaw rate.
            return upper_bound

    def _get_observation_lower_bound(self):
        if self.env_type == "walk":
            lower_bound = -self._get_observation_upper_bound()
            return lower_bound
        elif self.env_type == "turn":
            lower_bound = -self._get_observation_upper_bound()
            return lower_bound

    def set_swing_offset(self, value):
        """Set the swing offset of each leg.

    It is to mimic the bent leg.

    Args:
      value: A list of four values.
    """
        self._swing_offset = value

    def set_extension_offset(self, value):
        """Set the extension offset of each leg.

    It is to mimic the bent leg.

    Args:
      value: A list of four values.
    """
        self._extension_offset = value

    def set_desired_pitch(self, value):
        """Set the desired pitch of the base, which is a user input.

    Args:
      value: A scalar.
    """
        self.desired_pitch = value

    def _reward(self):
        current_base_position = self.rex.GetBasePosition()
        current_base_orientation = self.pybullet_client.getEulerFromQuaternion(
            self.rex.GetBaseOrientation())
        proximity_reward = abs(self.t_orient[0] - current_base_orientation[0]) + \
                           abs(self.t_orient[1] - current_base_orientation[1]) + \
                           abs(self.t_orient[2] - current_base_orientation[2])

        position_reward = abs(self.turn.TARGET_POSITION[0] - current_base_position[0]) + \
                          abs(self.turn.TARGET_POSITION[1] - current_base_position[1]) + \
                          abs(self.turn.TARGET_POSITION[2] - current_base_position[2])

        is_oriented = False
        is_pos = False
        if abs(proximity_reward) < 0.1:
            proximity_reward = 100 - proximity_reward
            is_oriented = True
        else:
            proximity_reward = -proximity_reward

        if abs(position_reward) < 0.1:
            position_reward = 100 - position_reward
            is_pos = True
        else:
            position_reward = -position_reward

        if is_pos and is_oriented:
            self.goal_reached = True
            self.goal_t = self.rex.GetTimeSinceReset()

        reward = position_reward + proximity_reward
        # print(reward)
        return reward







