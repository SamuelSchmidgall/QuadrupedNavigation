import math
import numpy as np


STAND_HIGH = np.array([ 0., -0.658319, 1.0472, 0., -0.658319,
    1.0472, 0., -0.658319, 1.0472, 0., -0.658319, 1.0472])

INIT_POSE = np.array([ 0.1, -0.82, 1.35, -0.1, -0.82,
    1.35, 0.1, -0.87, 1.35, -0.1, -0.87, 1.35])


class TurnController:
    def __init__(self):
        self.NUM_LEGS = 4
        self.desired_pitch = 0
        self.goal_reached = False
        self.NUM_MOTORS = 3 * self.NUM_LEGS
        self.STEP_PERIOD = 1.0 / 10.0  # 10 steps per second.
        self.TARGET_POSITION = [0.0, 0.0, 0.21]

    def reset(self):
        self.desired_pitch = 0
        self.goal_reached = False

    def step(self, action, time):
        action = self._transform_action_to_motor_command(action, time)
        return action

    def _transform_action_to_motor_command(self, action, time_since_reset):
        action += self._signal(time_since_reset)
        action = self._convert_from_leg_model(action, time_since_reset)
        return action

    def _convert_from_leg_model(self, leg_pose, t):
        if t < .4:
            # set init position
            return STAND_HIGH
        motor_pose = np.zeros(self.NUM_MOTORS)
        for i in range(self.NUM_LEGS):
            if self.goal_reached:
                return INIT_POSE
            else:
                motor_pose[3 * i] = leg_pose[3 * i]
                motor_pose[3 * i + 1] = leg_pose[3 * i + 1]
                motor_pose[3 * i + 2] = leg_pose[3 * i + 2]
        return motor_pose

    def _signal(self, t):
        # iterates at 0.006
        initial_pose = STAND_HIGH
        period = self.STEP_PERIOD
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
        # todo: compute this
        angle_distance_to_goal = 0.0
        if angle_distance_to_goal > 1:
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


class WalkController:
    def __init__(self):
        self.NUM_LEGS = 4
        self.DESIRED_PITCH = 0
        self.NUM_MOTORS = 3 * self.NUM_LEGS
        self.STEP_PERIOD = 1.0 / 4.5

    def step(self, action, t):
        # iterates at 0.006

        # action should be zeros?
        # time_since_reset = self.rex.GetTimeSinceReset()
        action += self._signal(t, INIT_POSE)
        action = self._convert_from_leg_model(action)
        return action

    def reset(self):
        pass

    def _convert_from_leg_model(self, leg_pose):
        motor_pose = np.zeros(self.NUM_MOTORS)
        for i in range(self.NUM_LEGS):
            if i % 2 == 0:
                motor_pose[3 * i] = 0.1
            else:
                motor_pose[3 * i] = -0.1
            motor_pose[3 * i + 1] = leg_pose[3 * i + 1]
            motor_pose[3 * i + 2] = leg_pose[3 * i + 2]
        return motor_pose

    def _signal(self, t, initial_pose):
        """
        :param t:
        :param initial_pose: self.rex.initial_pose
        :return:
        """
        period = self.STEP_PERIOD
        l_extension = 0.2 * math.cos(2 * math.pi / period * t)
        l_swing = -l_extension
        extension = 0.3 * math.cos(2 * math.pi / period * t)
        swing = -extension
        pose = np.array([
            0, l_extension, extension,
            0, l_swing, swing,
            0, l_swing, swing,
            0, l_extension, extension
        ])
        signal = initial_pose + pose
        return signal



