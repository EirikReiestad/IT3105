import configparser
from controllers import nn, pid
from plants import bathtub, cournot, plant3
import jax.numpy as jnp
import random

class System():
    def __init__(self, config_path: str) -> None:
        config = configparser.RawConfigParser()
        config.read(config_path)
        parameters = dict(config.items("DEFAULT"))

        for k, v in parameters.items():
            try:
                parameters[k] = int(v)
            except ValueError:
                try:
                    parameters[k] = float(v)
                except ValueError:
                    pass

        self.params = parameters

        self.controller = None
        if self.params['controller'] == 0:
            self.controller = pid.PIDController(
                self.params['learning_rate']
            )
        elif self.params['controller'] == 1:
            self.controller = nn.NNController(
                self.params['num_layers'],
                self.params['num_neurons'],
                self.params['activation_func'],
                self.params['max_val'],
                self.params['min_val']
            )

        self.plant = None
        if self.params['plant'] == 0:
            self.plant = bathtub.Bathtub(
                self.params["area_bathtub"],
                self.params["area_bathtub_drain"],
                self.params["init_height_bathtub"],
            )
        elif self.params['plant'] == 1:
            self.plant = cournot.Cournot(
                self.params["max_price_cournot"],
                self.params["marg_cost_cournot"]
            )
        elif self.params['plant'] == 2:
            self.plant = plant3.Plant3()

        # PLANT
    
    def run(self):
        # 1. Initialize the controller’s parameters (Ω): the three k values 
        # for a standard PID controller and the
        # weights and biases for a neural-net-based controller.
        params = self.controller.initialize()

        # 2. For each epoch:
        for epoch in self.params['epochs']:
            # (a) Initialize any other controller variables, such as the error history, and reset the plant to its initial
            # state.
            error_history = []
            # (b) Generate a vector of random noise / disturbance (D), with one value per timestep.
            noise = [random.uniform(self.params['min_noise'], self.params['max_noise']) for _ in range(self.params['sim_timesteps'])] # TODO: do we need to use jnp.random instead? amount of timestep
            # (c) For each timestep:
            for _ in self.params['sim_timesteps']:
                # • Update the plant
                output = self.plant.run_one_epoch(control_signal, noise)
                error = self.target - output
                # • Update the controller
                control_signal = self.controller.calculate_control_signal(error)
                # • Save the error (E) for this timestep in an error history.
                error_history.append(error)
            # (d) Compute MSE over the error history.
            
            # (e) Compute the gradients: ∂(MSE)/∂Ω
            # (f) Update Ω based on the gradients.
                
    def mse(self, errors):
        return jnp.mean(jnp.square(jnp.array(errors)))