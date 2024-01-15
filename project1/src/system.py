from controllers import nn, pid
from plants import bathtub, cournot, plant3
import jax.numpy as jnp
import jax
import random
import matplotlib.pyplot as plt
from lib import jax_type_to_python_type


class System:
    def __init__(self, parameters, visualize: bool = False) -> None:
        self.params = parameters
        self.visualize = visualize

        self.controller = None
        if self.params["controller"] == 0:
            if len(self.params["pid"]) == 3:
                p, d, i = self.params["pid"]
                self.controller = pid.PIDController(
                    self.params["learning_rate"], p, d, i)
            else:
                self.controller = pid.PIDController(
                    self.params["learning_rate"])
        elif self.params["controller"] == 1:
            self.controller = nn.NNController(
                self.params["hidden_layers"],
                self.params["activation_func"],
                self.params["max_val"],
                self.params["min_val"],
                self.params["learning_rate"],
            )

        self.plant = None
        if self.params["plant"] == 0:
            self.plant = bathtub.Bathtub(
                self.params["area_bathtub"],
                self.params["area_bathtub_drain"],
                self.params["init_height_bathtub"],
            )
            self.target = self.params["init_height_bathtub"]
        elif self.params["plant"] == 1:
            self.plant = cournot.Cournot(
                self.params["max_price_cournot"], self.params["marg_cost_cournot"]
            )
            self.target = self.params["goal_profit"]
        elif self.params["plant"] == 2:
            self.plant = plant3.Plant3(self.params['init_population'])
            self.target = self.params['target_population']

        # PLANT

        self.mse_history = []
        self.params_history = []

    def run(self):
        # 1. Initialize the controller’s parameters (Ω): the three k values
        # for a standard PID controller and the
        # weights and biases for a neural-net-based controller.
        # print(f"Running {self.params['epochs']} epochs")
        # print("##############################")
        params = self.controller.initialize()
        gradfunc = jax.value_and_grad(self.run_one_epoch, argnums=0)

        # 2. For each epoch:
        for i in range(self.params["epochs"]):
            # print(f'Epoch: {i}')
            # (e) Compute the gradients: ∂(MSE)/∂Ω
            mse, gradients = gradfunc(params)
            self.mse_history.append(mse)
            # (f) Update Ω based on the gradients.
            params = self.controller.update_params(params, gradients)
            if self.params["controller"] == 0:
                self.params_history.append(params)
            if self.visualize:
                self.visualize_training()

        return self.mse_history

    def run_one_epoch(self, params):
        # (a) Initialize any other controller variables, such as the error history, and reset the plant to its initial state.
        error_history = []
        state = self.plant.reset()
        # (b) Generate a vector of random noise / disturbance (D), with one value per timestep.
        noise = [
            random.uniform(self.params["min_noise"], self.params["max_noise"])
            for _ in range(self.params["sim_timesteps"])
        ]  # TODO: do we need to use jnp.random instead? amount of timestep
        # (c) For each timestep:
        control_signal = 0
        for j in range(self.params["sim_timesteps"]):
            # • Update the plant
            state, output = self.plant.run_one_epoch(
                state, control_signal, noise[j])
            error = self.target - output
            # • Update the controller
            error_history.append(error)

            control_signal = 0
            if len(error_history) > 1:
                control_signal = self.controller.calculate_control_signal(
                    params, error_history
                )
                control_signal = jax_type_to_python_type(control_signal)

            # • Save the error (E) for this timestep in an error history.
        # (d) Compute MSE over the error history.
        mse = self.mse(error_history)
        return mse

    def mse(self, errors):
        return jnp.mean(jnp.square(jnp.array(errors)))

    def visualize_training(self):
        '''
        Visualize the error (MSEs) and PID parameters
        '''
        _, axis = plt.subplots(1, 2)

        axis[0].plot(self.mse_history)
        axis[0].set_title("Learning Progression")
        axis[0].set_xlabel("Epoch")
        axis[0].set_ylabel("MSE")

        if self.params["controller"] == 0:
            params = {
                key: [item[key] for item in self.params_history]
                for key in self.params_history[0]
            }
            axis[1].plot(params["k_p"], label="Kp")
            axis[1].plot(params["k_d"], label="Kd")
            axis[1].plot(params["k_i"], label="Ki")
            axis[1].set_title("Control Parameters")
            axis[1].set_xlabel("Epoch")
            axis[1].set_ylabel("Y")
            axis[1].legend()

        plt.show()
