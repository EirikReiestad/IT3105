import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.controllers.controller import Controller
import math
import numpy as np
import jax
import jax.numpy as jnp


class NNController(Controller):
    def __init__(
        self,
        hidden_layers: list,
        activation_func,
        max_val: float,
        min_val: float,
        learning_rate: float,
    ):
        self.layers = hidden_layers
        self.activation_func = activation_func
        self.max_val = max_val
        self.min_val = min_val
        self.learning_rate = learning_rate

    def initialize(self) -> dict:
        # Initialize input and output layer because for this plant it is fixed
        self.layers.insert(0, 3)
        self.layers.append(1)

        # Initialize weights and biases
        sender = self.layers[0]
        params = []
        for receiver in self.layers[1:]:
            weights = np.random.uniform(self.min_val, self.max_val, (sender, receiver))
            biases = np.random.uniform(self.min_val, self.max_val, (receiver))
            sender = receiver
            params.append([weights, biases])
        return params

    def calculate_control_signal(self, params, error_list: list, dx=1.0) -> float:
        error = error_list[-1]
        error_change = error_list[-1] - error_list[-2] / dx
        sum_error = sum(error_list)

        activations = jnp.array(
            [error, error_change, sum_error]
        ).ravel()  # Flatten array
        for weights, biases in params:
            activations = self.activation(jnp.dot(activations, weights) + biases)
        return activations.item()  # Return scalar

    def update_params(self, params: dict, gradients):
        return [
            (
                weight - self.learning_rate * weight_gradient,
                bias - self.learning_rate * bias_gradient,
            )
            for (weight, bias), (weight_gradient, bias_gradient) in zip(
                params, gradients
            )
        ]

    def activation(self, val):
        if self.activation_func == 0:
            return self.sigmoid(val)
        elif self.activation_func == 1:
            return self.tanh(val)
        elif self.activation_func == 2:
            return self.relu(val)
        return None

    def sigmoid(self, val):
        return 1 / (1 + math.e ** (-val))

    def tanh(self, val):
        return math.tanh(val)

    def relu(self, val):
        return max(0, val)
