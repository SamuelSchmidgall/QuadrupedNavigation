import math
import torch
import pickle
import numpy as np
from torch import nn
from scipy import signal
from torch.distributions import Normal


class ProximalPolicyOptimization(nn.Module):
    def __init__(self, actor_critic, epochs=10, minibatch_size=1000, timestep_size=4000, entropy_coefficient=0.01):
        super(ProximalPolicyOptimization, self).__init__()

        self.lamb = 0.95
        self.gamma = 0.99
        self.ppo_clip = 0.2
        self.learning_rate=0.0003

        self.epochs = epochs
        self.timestep_size = timestep_size
        self.minibatch_size = minibatch_size
        self.entropy_coefficient = entropy_coefficient

        self.actor_critic = actor_critic

    def forward(self, x, memory, evaluate=False, visual_obs=None, append=True):
        if visual_obs is None:
            action_mean, action_log_std = self.actor_critic(x)
        else:
            action_mean, action_log_std = self.actor_critic(x, visual_obs)
        action_std = torch.exp(action_log_std)

        distribution = Normal(loc=action_mean, scale=action_std)
        action = distribution.sample()
        log_probabilities = distribution.log_prob(action)
        log_probabilities = torch.sum(log_probabilities, dim=1)
        if append:
            memory.log_probs.append(log_probabilities.detach())
        if evaluate:
            return action_mean.detach(), None
        return action, memory

    def evaluate(self, x, old_action):
        action_mean, action_log_std = self.actor_critic(x)
        action_std = torch.exp(action_log_std)

        distribution = Normal(loc=action_mean, scale=action_std)
        log_probabilities = distribution.log_prob(old_action.squeeze(dim=1))
        log_probabilities = torch.sum(log_probabilities, dim=1)

        entropy = distribution.entropy()

        return log_probabilities, entropy

    def generalized_advantage_estimation(self, r, v, mask):
        batchsz = v.size(0)

        # v_target is worked out by Bellman equation.
        delta = torch.Tensor(batchsz)
        v_target = torch.Tensor(batchsz)
        adv_state = torch.Tensor(batchsz)

        prev_v = 0
        prev_v_target = 0
        prev_adv_state = 0
        for t in reversed(range(batchsz)):
            # V(s_t) = r_t + gamma * V(s_t+1)
            v_target[t] = r[t] + self.gamma * prev_v_target * mask[t]

            # delta(s_t) = r_t + gamma * V(s_t+1) - V(s_t)
            delta[t] = r[t] + self.gamma * prev_v * mask[t] - v[t]

            # A(s, a) = delta(s_t) + gamma * lamda * A(s_t+1, a_t+1)
            adv_state[t] = delta[t] + self.gamma * self.lamb * prev_adv_state * mask[t]

            # update previous
            prev_v_target = v_target[t]
            prev_v = v[t]
            prev_adv_state = adv_state[t]

        # normalize adv_state
        adv_state = (adv_state - adv_state.mean()) / (adv_state.std() + 1e-6)

        return adv_state, v_target


    def learn(self, memory, ratio, ep=-1):
        self.actor_critic.value_optim.param_groups[0]["lr"] = self.learning_rate*ratio
        self.actor_critic.policy_optim.param_groups[0]["lr"] = self.learning_rate*ratio

        replay_len = len(memory.rewards)
        minibatch_count = self.timestep_size / self.minibatch_size

        values = self.actor_critic.value(torch.FloatTensor(memory.sensor_states)).detach()

        advantages, value_target = self.generalized_advantage_estimation(
            torch.FloatTensor(memory.rewards).unsqueeze(1), values, torch.FloatTensor(memory.reset_flags).unsqueeze(1))

        advantages = advantages.detach().numpy()
        value_target = value_target.detach().numpy()

        #advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
        weight_update_reward = list()

        for _ in range(self.epochs):
            minibatch_indices = list(range(replay_len))
            np.random.shuffle(minibatch_indices)
            minibatches = [minibatch_indices[int(_ * (replay_len/minibatch_count)):
                int((_ + 1) * (replay_len/minibatch_count))] for _ in range(int(minibatch_count))]

            for batch in minibatches:

                mb_states = torch.FloatTensor(np.array(memory.sensor_states)[batch])
                mb_actions = torch.stack(memory.actions).index_select(0, torch.LongTensor(batch))
                mb_old_log_probabilities = torch.stack(memory.log_probs).index_select(0, torch.LongTensor(batch))

                predicted_values = self.actor_critic.value(mb_states)

                log_probabilities, entropy = self.evaluate(mb_states, mb_actions)

                mb_advantages = torch.FloatTensor(advantages[batch])

                ratio = (log_probabilities - mb_old_log_probabilities.squeeze()).exp()
                min_adv = torch.where(mb_advantages > 0,
                    (1 + self.ppo_clip) * mb_advantages, (1 - self.ppo_clip) * mb_advantages)
                policy_loss = -(torch.min(ratio * mb_advantages, min_adv)).mean() - self.entropy_coefficient*entropy.mean()

                value_loss = (torch.FloatTensor(value_target[batch]) - predicted_values.squeeze()).pow(2).mean()
                self.actor_critic.optimize(policy_loss, value_loss)
        weight_update_reward.append(sum(memory.rewards)/ep)
        return weight_update_reward



sensors = ['walker/actuator_activation',
'walker/appendages_pos',
'walker/joints_pos',
'walker/joints_vel',
'walker/sensors_accelerometer',
'walker/sensors_gyro',
'walker/sensors_touch',
'walker/sensors_velocimeter',
'walker/tendons_pos',
'walker/tendons_vel']



import numpy as np
def observation_def(obs_spec):
    obs_size = 0
    for _s in sensors:
        obs_size += obs_spec[_s].shape[0]
    return obs_size


def generate_obs(sensor_dict):
    sens = np.reshape(sensor_dict[sensors[0]], (1, sensor_dict[sensors[0]].shape[0]))
    for _s in sensors[1:]:
        sens1 = np.reshape(sensor_dict[_s], (1, sensor_dict[_s].shape[0]))
        sens = np.append(sens, sens1, axis=1)
    return sens.flatten()



from copy import deepcopy

def run(train_id=0):
    torch.set_num_threads(1)

    from dm_env import StepType
    from Networks.rodent_test import rodent_run
    env = rodent_run()
    action_spec = env.action_spec()
    obs_spec = observation_def(env.observation_spec())

    from networks import ActorCritic, NeuralNetwork, ReplayMemory, MultiSensoryNeuralNetwork

    agent_replay = ReplayMemory()
    #net = QuadrupedActorCritic(env)
    net = ActorCritic(NeuralNetwork([256, 256], observation_dim=obs_spec, action_dim=action_spec.shape[0]))

    agent = ProximalPolicyOptimization(
        net, epochs=15, minibatch_size=10000, timestep_size=30000, entropy_coefficient=0.001)

    timesteps = 0
    total_timesteps = 0
    max_timesteps = 100000000
    avg_action_magnitude = 0

    episode_itr = 0
    avg_sum_rewards = 0.0

    upd_list = list()
    saved_reward = list()
    saved_finish_mask = list()

    while total_timesteps < max_timesteps:
        game_over = False

        sensor_obs = env.reset()
        sensor_obs = generate_obs(sensor_obs.observation)

        while not game_over:
            #env.render()
            local_action, agent_replay = agent(x=torch.FloatTensor(sensor_obs).unsqueeze(0), memory=agent_replay)

            agent_replay.sensor_states.append(deepcopy(sensor_obs))

            agent_replay.actions.append(local_action)

            local_action = local_action.squeeze(dim=1).numpy()

            game_over, reward, _, _s_d = env.step(np.clip(local_action, a_min=-1, a_max=1))  # Step
            game_over = game_over == StepType.LAST
            sensor_obs = generate_obs(sensor_dict=_s_d)

            agent_replay.reset_flags.append(0 if game_over else 1)

            agent_replay.rewards.append(reward)

            avg_sum_rewards += reward

            timesteps += 1
            total_timesteps += 1

        episode_itr += 1

        if timesteps > agent.timestep_size:
            per = 1-(timesteps/max_timesteps)
            updates = agent.learn(memory=agent_replay, ratio=per, ep=episode_itr)
            upd_list += updates
            avg_action_magnitude /= timesteps

            print("Time: {} Reward: {}, Timestep: {}".format(
                round(timesteps/episode_itr, 8), round(avg_sum_rewards/episode_itr, 8),
                 total_timesteps))

            timesteps = 0
            episode_itr = 0
            avg_sum_rewards = 0.0
            avg_action_magnitude  = 0


            with open("saved_model_rodent_{}_{}.pkl".format("linear", train_id), "wb") as f:
                pickle.dump(agent, f)

            with open("saved_weightupd_rodent_rew.pkl", "wb") as f:
                pickle.dump(upd_list, f)

            agent_replay.clear()
            saved_reward.clear()
            saved_finish_mask.clear()



run(1345)










