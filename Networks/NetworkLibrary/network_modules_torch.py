import torch
import torch.nn as nn
from copy import deepcopy
import torch.optim as optim
from Networks.Utilities.weight_init import *


class LinearModule(nn.Module):
    def __init__(self, input_size, output_size, weight_init, activation=None):
        """
        Module structure for traditional linear layer
        :param input_size: (int) input dimensionality
        :param output_size: (int) output dimensionality
        :param weight_init: (str) weight initialization type
        """
        super(LinearModule, self).__init__()
        # activation function
        self.activation = activation
        # simple feedforward linear layer
        self.layer = nn.Linear(input_size, output_size)

        # weight initialize linear layers if applies
        self.initialization = weight_init
        if self.initialization is not None:
            initialize_weights(self.layer, self.initialization)

    def reset(self):
        # No internal components to reset
        pass

    def forward(self, x):
        """
        Forward propagate value x
        :param x: (torch.Tensor) value to feedforward
        :return: (torch.Tensor) Forward propagated value
        """
        post_synaptic = self.layer(x)
        if self.activation is not None:
            post_synaptic = self.activation(post_synaptic)
        return post_synaptic


class EligibilityModule(nn.Module):
    """
    Module structure for eligibility feedforward layer
    :param input_size: (int) input dimensionality
    :param output_size: (int) output dimensionality
    :param weight_init: (str) weight initialization type
    """
    def __init__(self, input_size, output_size, weight_init, activation=None):
        super(EligibilityModule, self).__init__()
        # activation function
        self.activation = activation
        # plasticity traces
        self.hebbian_trace = torch.zeros(
            (input_size, output_size), requires_grad=False)
        self.eligibility_trace = torch.zeros(
            (input_size, output_size), requires_grad=False)
        # feedforward, plastic, and modulatory layers
        self.modulation_fan_in = nn.Linear(output_size, 1)
        self.modulation_fan_out = nn.Linear(1, output_size)
        self.eligibility_eta = nn.Parameter(
            torch.rand(1)*0.05, requires_grad=True)
        self.alpha_plasticity = nn.Parameter(torch.rand(
            output_size, output_size)*0.05, requires_grad=True)
        self.layer = nn.Linear(input_size, output_size)

        # weight initialize linear layers if applies
        self.initialization = weight_init
        if self.initialization is not None:
            initialize_weights(self.layer, self.initialization)
            initialize_weights(self.modulation_fan_out, self.initialization)
            initialize_weights(self.modulation_fan_in, self.initialization)

    def reset(self):
        """
        Reset internal states
        :return: None
        """
        self.hebbian_trace = self.hebbian_trace.detach() * 0
        self.eligibility_trace = self.eligibility_trace.detach() * 0

    def update_trace(self, pre_synaptic, post_synaptic):
        """
        Update internal plasticity trace
        :param pre_synaptic: (torch.Tensor) presynaptic activity
        :param post_synaptic: (torch.Tensor) postsynaptic activity
        :return: None, update internal states
        """
        modulatory_signal = self.modulation_fan_out(
            torch.tanh(self.modulation_fan_in(post_synaptic)))

        self.hebbian_trace = torch.clamp(
            self.hebbian_trace + modulatory_signal * self.eligibility_trace,
            max=self.module_metadata["clip"], min=self.module_metadata["clip"] * -1)

        self.eligibility_trace = (torch.ones(1) - self.eligibility_eta) *\
            self.eligibility_trace + self.eligibility_eta * (torch.mm(pre_synaptic, post_synaptic))

    def forward(self, x):
        """
        Forward propagate value x, update plastic weights
        :param x: (torch.Tensor) value to feedforward
        :return: (torch.Tensor) Forward propagated value
        """
        pre_synaptic = x.clone()  # x.detach().clone()
        fixed_weights = self.layer(x)
        plastic_weights = x.mm(self.alpha_plasticity * self.hebbian_trace)
        post_synaptic = fixed_weights + plastic_weights
        if self.activation is not None:
            post_synaptic = self.activation(post_synaptic)
        self.update_trace(pre_synaptic=pre_synaptic, post_synaptic=post_synaptic)
        return post_synaptic


class LSTMModule(nn.Module):
    """
    Module structure for LSTM layer
    :param input_size: (int) input dimensionality
    :param output_size: (int) output dimensionality
    :param weight_init: (str) weight initialization type
    """
    def __init__(self, input_size, output_size, weight_init, activation=None):
        super(LSTMModule, self).__init__()
        # activation function
        self.activation = activation
        # internal cell and recurrent states
        self.cell_state = torch.zeros(1, output_size)
        self.recurrent_trace = torch.zeros(1, output_size)
        # feedforward and recurrent layers
        self.layer_i = nn.Linear(input_size, output_size)
        self.layer_j = nn.Linear(input_size, output_size)
        self.layer_f = nn.Linear(input_size, output_size)
        self.layer_o = nn.Linear(input_size, output_size)
        self.recurrent_layer_hi = nn.Linear(output_size, output_size)
        self.recurrent_layer_hj = nn.Linear(output_size, output_size)
        self.recurrent_layer_hf = nn.Linear(output_size, output_size)
        self.recurrent_layer_ho = nn.Linear(output_size, output_size)

        # weight initialize linear layers if applies
        self.initialization = weight_init
        if self.initialization is not None:
            initialize_weights(self.layer, self.initialization)
            initialize_weights(self.layer_i, self.initialization)
            initialize_weights(self.layer_j, self.initialization)
            initialize_weights(self.layer_f, self.initialization)
            initialize_weights(self.layer_o, self.initialization)
            initialize_weights(self.recurrent_layer_hi, self.initialization)
            initialize_weights(self.recurrent_layer_hj, self.initialization)
            initialize_weights(self.recurrent_layer_hf, self.initialization)
            initialize_weights(self.recurrent_layer_ho, self.initialization)

    def reset(self):
        """
        Reset internal states
        :return: None
        """
        self.cell_state = self.cell_state.detach() * 0
        self.recurrent_trace = self.recurrent_trace.detach() * 0

    def forward(self, x):
        """
        Forward propagate value x, update recurrent weights
        :param x: (torch.Tensor) value to feedforward
        :return: (torch.Tensor) Forward propagated value
        """
        post_synaptic = torch.tanh(
            self.layer_i(x) + self.recurrent_layer_hi(self.recurrent_trace))
        j_t = torch.tanh(self.layer_j(x) + self.recurrent_layer_hj(self.recurrent_trace))
        f_t = torch.tanh(self.layer_f(x) + self.recurrent_layer_hf(self.recurrent_trace))
        o_t = torch.tanh(self.layer_o(x) + self.recurrent_layer_ho(self.recurrent_trace))
        c_t = f_t * self.cell_state + post_synaptic * j_t
        h_t = torch.tanh(c_t) * o_t
        post_synaptic = h_t
        return post_synaptic


class EligibilityLSTMModule(nn.Module):
    """
    Module structure for eligibility modulated LSTM layer
    :param input_size: (int) input dimensionality
    :param output_size: (int) output dimensionality
    :param weight_init: (str) weight initialization type
    """
    def __init__(self, input_size, output_size, weight_init, activation=None):
        super(EligibilityLSTMModule, self).__init__()
        # activation function
        self.activation = activation
        # internal recurrent and plastic traces
        self.cell_state = torch.zeros(1, output_size)
        self.recurrent_trace = torch.zeros(1, output_size)
        self.hebbian_trace = torch.zeros((input_size, output_size), requires_grad=False)
        self.eligibility_trace = torch.zeros((input_size, output_size), requires_grad=False)
        # modulatory, recurrent, and feedforward traces
        self.modulation_fan_in = nn.Linear(output_size, 1)
        self.modulation_fan_out = nn.Linear(1, output_size)
        self.eligibility_eta = nn.Parameter(
            torch.rand(1)*0.05, requires_grad=True)
        self.alpha_plasticity = nn.Parameter(torch.rand(
            output_size, output_size)*0.05, requires_grad=True)
        self.layer_i = nn.Linear(input_size, output_size)
        self.layer_j = nn.Linear(input_size, output_size)
        self.layer_f = nn.Linear(input_size, output_size)
        self.layer_o = nn.Linear(input_size, output_size)
        self.recurrent_layer_hi = nn.Linear(output_size, output_size)
        self.recurrent_layer_hj = nn.Linear(output_size, output_size)
        self.recurrent_layer_hf = nn.Linear(output_size, output_size)
        self.recurrent_layer_ho = nn.Linear(output_size, output_size)

        # weight initialize linear layers if applies
        self.initialization = weight_init
        if self.initialization is not None:
            initialize_weights(self.layer, self.initialization)
            initialize_weights(self.layer_i, self.initialization)
            initialize_weights(self.layer_j, self.initialization)
            initialize_weights(self.layer_f, self.initialization)
            initialize_weights(self.layer_o, self.initialization)
            initialize_weights(self.modulation_fan_in, self.initialization)
            initialize_weights(self.recurrent_layer_hi, self.initialization)
            initialize_weights(self.recurrent_layer_hj, self.initialization)
            initialize_weights(self.recurrent_layer_hf, self.initialization)
            initialize_weights(self.recurrent_layer_ho, self.initialization)
            initialize_weights(self.modulation_fan_out, self.initialization)

    def reset(self):
        """
        Reset internal states
        :return: None
        """
        self.hebbian_trace = self.hebbian_trace.detach() * 0
        self.recurrent_trace = self.recurrent_trace.detach() * 0
        self.eligibility_trace = self.eligibility_trace.detach() * 0
        self.cell_state = self.cell_state.detach() * 0

    def update_trace(self, pre_synaptic, post_synaptic):
        """
        Update internal plasticity trace
        :param pre_synaptic: (torch.Tensor) presynaptic activity
        :param post_synaptic: (torch.Tensor) postsynaptic activity
        :return: None, update internal states
        """
        modulatory_signal = self.modulation_fan_out(
            torch.tanh(self.modulation_fan_in(post_synaptic)))
        self.hebbian_trace = torch.clamp(
            self.hebbian_trace + modulatory_signal * self.eligibility_trace,
            max=self.module_metadata["clip"], min=self.module_metadata["clip"] * -1)
        self.eligibility_trace = (torch.ones(1) - self.eligibility_eta) * \
            self.eligibility_trace + self.eligibility_eta * (torch.mm(pre_synaptic.t(), post_synaptic))

    def forward(self, x):
        """
        Forward propagate value x, update plastic+recurrent weights
        :param x: (torch.Tensor) value to feedforward
        :return: (torch.Tensor) Forward propagated value
        """
        pre_synaptic = x
        plastic_weights = x.mm(self.alpha_plasticity * self.hebbian_trace)
        post_synaptic = plastic_weights + \
            self.layer_i(x) + self.recurrent_layer_hi(self.recurrent_trace)
        post_synaptic = torch.tanh(post_synaptic)
        j_t = torch.tanh(self.layer_j(x) + self.recurrent_layer_hj(self.recurrent_trace))
        f_t = torch.tanh(self.layer_f(x) + self.recurrent_layer_hf(self.recurrent_trace))
        o_t = torch.tanh(self.layer_o(x) + self.recurrent_layer_ho(self.recurrent_trace))
        c_t = f_t * self.cell_state + post_synaptic * j_t
        h_t = torch.tanh(c_t) * o_t
        self.update_trace(pre_synaptic=pre_synaptic, post_synaptic=post_synaptic)
        post_synaptic = h_t
        return post_synaptic


class NetworkConnectivityModule(nn.Module):
    def __init__(self, module_type, module_metadata):
        """
        Network Sub-Module responsible for processing
         and connecting information between two layers
        :param module_type: (str) -> module connection type
        :param module_metadata: (dict) -> dictionary containing relavent metadata
        """
        # todo: small intialization, especially for neuromodulation
        # todo: grad clip? weight norm?
        super(NetworkConnectivityModule, self).__init__()
        self.module_type = module_type
        self.module_metadata = module_metadata
        self.activation = self.module_metadata["activation"]
        # if weight initialization specified, set variable
        if "weight_initialization" in self.module_metadata:
            self.weight_initialization = self.module_metadata["weight_initialization"]
        else:
            self.weight_initialization = None
        # if activation (metabolic) penalty specified, set variable
        if "activation_penalties" in self.module_metadata:
            self.activation_penalties = self.module_metadata["activation_penalties"]
        else:
            self.initialization = None
        # initialize respective module type
        # feedforward linear module
        if self.module_type == "linear":
            self.module = LinearModule(module_metadata["input_size"],
                module_metadata["output_size"], weight_init=self.weight_initialization, activation=self.activation)
        # feedforward eligibility trace module
        elif self.module_type == "eligibility":
            self.module = EligibilityModule(module_metadata["input_size"],
                module_metadata["output_size"], weight_init=self.weight_initialization, activation=self.activation)
        # LSTM eligibility modulated module
        elif self.module_type == "eligibility_lstm":
            self.module = EligibilityLSTMModule(module_metadata["input_size"],
                module_metadata["output_size"], weight_init=self.weight_initialization, activation=self.activation)
        else:
            raise Exception("({}) does not exist as a module type".format(module_type))

    def reset(self):
        """
        Reset module internal states
        :return: None
        """
        self.module.reset()

    def forward(self, x):
        """
        Forward propagate value x, update trace if applies
        :param x: (torch.Tensor) value to feedforward
        :return: (torch.Tensor) Forward propagated value
        """
        return self.module.forward(x)














