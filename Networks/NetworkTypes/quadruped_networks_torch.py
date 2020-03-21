from torch.distributions import Normal
from Networks.NetworkLibrary.network_modules_torch import *

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


class ValueNetwork(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(ValueNetwork, self).__init__()

        ff1_meta = {
            "activation": None, "input_size": input_dim,
            "output_size": 256, "initialization": "orthogonal"}
        self.ff1 = NetworkConnectivityModule("linear", ff1_meta)
        ff2_meta = {
            "activation": None, "input_size": 256,
            "output_size": 256, "initialization": "orthogonal"}
        self.ff2 = NetworkConnectivityModule("linear", ff2_meta)
        ff3_meta = {
            "activation": None, "input_size": 256,
            "output_size": 1, "initialization": "orthogonal"}
        self.ff3 = NetworkConnectivityModule("linear", ff3_meta)

    def forward(self, x):
        x = torch.tanh(self.ff1(x))
        x = torch.tanh(self.ff2(x))
        x = self.ff3(x)
        return x


class QuadrupedNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, action_space):
        super(QuadrupedNetwork, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.action_scale = torch.FloatTensor(
            (action_space.high - action_space.low) / 2.)
        self.action_bias = torch.FloatTensor(
            (action_space.high + action_space.low) / 2.)

        ff1_meta = {
            "activation": None, "input_size": self.input_dim,
            "output_size": 256, "initialization": "orthogonal"}
        self.ff1 = NetworkConnectivityModule("linear", ff1_meta)

        ff2_meta = {
            "activation": None, "input_size": 256,
            "output_size": 256, "initialization": "orthogonal"}
        self.ff2 = NetworkConnectivityModule("linear", ff2_meta)

        ff3_meta = {
            "activation": None, "input_size": 256,
            "output_size": self.output_dim, "initialization": "orthogonal"}
        self.ff3 = NetworkConnectivityModule("linear", ff3_meta)

        #log_std_linear_meta = {
        #    "activation": None, "input_size": 256,
        #    "output_size": self.output_dim, "initialization": "orthogonal"}
        self.log_std_linear = torch.ones((1, self.output_dim))*-0.5#NetworkConnectivityModule("linear", log_std_linear_meta)

    def reset(self):
        pass # no recurrence

    def forward(self, x):
        x = torch.tanh(self.ff1(x))
        x = torch.tanh(self.ff2(x))
        mean = self.ff3(x)
        log_std = self.log_std_linear
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


class QuadrupedActorCritic(nn.Module):
    def __init__(self, env, policy_learning_rate=0.0003,
        value_learning_rate=0.0003, optimizer=optim.Adam, anneal=False):
        super(QuadrupedActorCritic, self).__init__()
        self.output_dim = env.action_space.shape[0]
        self.input_dim = env.observation_space.shape[0]
        self.anneal = anneal
        self.value_function = ValueNetwork(self.input_dim, 1)
        self.policy = QuadrupedNetwork(
            self.input_dim, self.output_dim, env.action_space)

        policy_params, value_params = self.params()

        self.value_optim = optimizer(params=value_params, lr=value_learning_rate)
        self.policy_optim = optimizer(params=policy_params, lr=policy_learning_rate)

    def anneal(self):
        pass

    def forward(self, x):
        return self.policy(x)

    def value(self, x):
        return self.value_function(x)

    def optimize(self, policy_loss, value_loss):
        self.value_optim.zero_grad()
        value_loss.backward()
        self.value_optim.step()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

    def params(self):
        return self.policy.parameters(), self.value_function.parameters()

    def reset(self):
        self.policy.reset()


class RodentNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(RodentNetwork, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        ff1_meta = {
            "activation": None, "input_size": self.input_dim,
            "output_size": 256, "initialization": "orthogonal"}
        self.ff1 = NetworkConnectivityModule("linear", ff1_meta)

        ff2_meta = {
            "activation": None, "input_size": 256,
            "output_size": 256, "initialization": "orthogonal"}
        self.ff2 = NetworkConnectivityModule("linear", ff2_meta)

        ff3_meta = {
            "activation": None, "input_size": 256,
            "output_size": self.output_dim, "initialization": "orthogonal"}
        self.ff3 = NetworkConnectivityModule("linear", ff3_meta)

        log_std_linear_meta = {
            "activation": None, "input_size": 256,
            "output_size": self.output_dim, "initialization": "orthogonal"}
        self.log_std_linear = NetworkConnectivityModule("linear", log_std_linear_meta)

    def reset(self):
        pass # no recurrence

    def forward(self, x):
        x = torch.tanh(self.ff1(x))
        x = torch.tanh(self.ff2(x))
        mean = self.ff3(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std



class RodentActorCritic(nn.Module):
    def __init__(self, inp_dim, outp_dim, policy_learning_rate=0.0003,
        value_learning_rate=0.0003, optimizer=optim.Adam, anneal=False):
        super(RodentActorCritic, self).__init__()
        self.anneal = anneal
        self.input_dim = outp_dim
        self.output_dim = inp_dim
        self.value_function = ValueNetwork(self.input_dim, 1)
        self.policy = RodentNetwork(
            self.input_dim, self.output_dim)

        policy_params, value_params = self.params()

        self.value_optim = optimizer(params=value_params, lr=value_learning_rate)
        self.policy_optim = optimizer(params=policy_params, lr=policy_learning_rate)

    def anneal(self):
        pass

    def forward(self, x):
        return self.policy(x)

    def value(self, x):
        return self.value_function(x)

    def optimize(self, policy_loss, value_loss):
        self.value_optim.zero_grad()
        value_loss.backward()
        self.value_optim.step()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

    def params(self):
        return self.policy.parameters(), self.value_function.parameters()

    def reset(self):
        self.policy.reset()



















