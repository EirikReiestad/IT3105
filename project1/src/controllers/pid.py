from controllers.controller import Controller
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


class PIDController(Controller):
    def __init__(self, learning_rate, k_p=0.5, k_d=0.5, k_i=0.3):
        self.learning_rate = learning_rate
        self.k_p = k_p
        self.k_d = k_d
        self.k_i = k_i

    def initialize(self):
        """
        Initialize the controller.

        Parameters:

        """
        return {"k_p": self.k_p, "k_d": self.k_d, "k_i": self.k_i}

    def calculate_control_signal(self, params, error_list: list, dx=1.0) -> float:
        """
        Calculate the control signal for the PID controller.
        Formula: U = Kp * E + Kd(dE/dt) + Ki * Sum(E)

        Parameters:
            error (float): The error to the system.

        Returns:
            control_signal (float): The derivative to a control output.
        """
        u_p = params["k_p"] * error_list[-1]
        u_d = params["k_d"] * ((error_list[-1] - error_list[-2]) / dx)
        u_i = params["k_i"] * sum(error_list)

        return u_p + u_d + u_i

    def update_params(self, params, gradients):
        params = params.copy()
        params["k_p"] = params["k_p"] - self.learning_rate * gradients["k_p"]
        params["k_d"] = params["k_d"] - self.learning_rate * gradients["k_d"]
        params["k_i"] = params["k_i"] - self.learning_rate * gradients["k_i"]
        return params
