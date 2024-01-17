import jax.numpy as jnp
import jax
import numpy as np
import math
from controllers.controller import Controller
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


class NNController(Controller):
    def __init__(
        self,
        hidden_layers: list,
        activation_func,
        max_val: float,
        min_val: float,
        learning_rate: float,
    ):
        self.layers = hidden_layers.copy()
        self.activation_func = activation_func.copy()
        self.max_val = max_val
        self.min_val = min_val
        self.learning_rate = learning_rate

        if len(self.layers) != len(self.activation_func):
            raise ValueError(
                "Number of hidden layers and activation functions must be the same: {} is not {}".format(
                    len(self.layers), len(self.activation_func))
            )

    def initialize(self) -> dict:
        """
        Initialize the params of the neural network
        """
        # Initialize input and output layer because for this plant it is fixed
        self.layers.insert(0, 3)
        self.layers.append(1)
        self.layers = jnp.array(self.layers)
        self.activation_func.insert(0, self.activation_func[0])
        self.activation_func.append(self.activation_func[-1])
        self.activation_func = jnp.array(self.activation_func)

        # Initialize weights and biases
        sender = self.layers[0]
        params = []
        for receiver in self.layers[1:]:
            key = jax.random.PRNGKey(0)
            weights = jax.random.uniform(
                key, minval=self.min_val, maxval=self.max_val, shape=(sender, receiver))
            biases = jax.random.uniform(
                key, minval=self.min_val, maxval=self.max_val, shape=(receiver,))
            sender = receiver
            params.append((weights, biases))
        return params

    def calculate_control_signal(self, params, error_list: list, dx=1.0) -> float:
        """
        Calculate the control signal
        """
        error = error_list[-1]
        error_change = (error_list[-1] - error_list[-2]) / dx
        sum_error = sum(error_list)

        activations = jnp.array(
            [error, error_change, sum_error]
        ).ravel()  # Flatten array

        for i, (weights, biases) in enumerate(params):
            activations = self.activation(
                i, jnp.dot(activations, weights) + biases)

        return activations[0]

    def update_params(self, params: dict, gradients):
        """
        Update the network's weights and biases
        """
        return [
            (
                weight - self.learning_rate * weight_gradient,
                bias - self.learning_rate * bias_gradient,
            )
            for (weight, bias), (weight_gradient, bias_gradient) in zip(
                params, gradients
            )
        ]

    def activation(self, layer, val):
        '''
        Choose activation function based on layer's configuration
        '''
        if self.activation_func[layer] == 0:
            return self.sigmoid(val)
        elif self.activation_func[layer] == 1:
            return self.tanh(val)
        elif self.activation_func[layer] == 2:
            return self.relu(val)
        elif self.activation_func[layer] == 3:
            return self.linear(val)
        return None

    def sigmoid(self, val):
        """
        Calculate the sigmoid value
        """
        return 1 / (1 + jnp.exp(-val))
    def tanh(self, val):
        """
        Calculate the tanh value
        """
        return jnp.tanh(val)

    def relu(self, val):
        """
        Calculate the max of the value and 0
        """
        return jnp.maximum(0, val)

    def linear(self, val):
        """
        Calculate the linear value
        """
        return val
