import os
import sys
import configparser
import numpy as np
from itertools import product

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.system import System

def tune(system_params, tuning_params):
    print(tuning_params)
    hidden_layers = [[i, j] for i in range(
        tuning_params["hidden_layers"]["min"], 
        tuning_params["hidden_layers"]["max"] + 1)
        for j in range(
            tuning_params["hidden_layers"]["min_num_neurons"], 
            tuning_params["hidden_layers"]["max_num_neurons"] + 1)]
    activation_func = tuning_params["activation_func"]
    epochs = list(np.linespace(tuning_params["epochs"]["min"], 
                        tuning_params["epochs"]["max"] + 1, 
                        tuning_params["epochs"]["iterations"], 
                        dtype=int))
    sim_timesteps= list(np.linespace(tuning_params["sim_timesteps"]["min"], 
                        tuning_params["sim_timesteps"]["max"] + 1, 
                        tuning_params["sim_timesteps"]["iterations"], 
                        dtype=int))
    learning_rate = list(np.linespace(tuning_params["learning_rate"]["min"], 
                        tuning_params["learning_rate"]["max"] + 1, 
                        tuning_params["learning_rate"]["iterations"]))

    param_space = list(product(hidden_layers, activation_func, epochs, sim_timesteps, learning_rate))

    for params in param_space:
        parameters = system_params.clone()
        hidden_layers, activation_func, epochs, sim_timesteps, learning_rate = params
        print("===== Running with parameters: =====")
        print("Hidden Layers: ", hidden_layers)
        print("Activation Function: ", activation_func)
        print("Epochs: ", epochs)
        print("Simulation Timesteps: ", sim_timesteps)
        print("Learning Rate: ", learning_rate)

        parameters["hidden_layers"] = hidden_layers
        parameters["activation_func"] = activation_func
        parameters["epochs"] = epochs
        parameters["sim_timesteps"] = sim_timesteps
        parameters["learning_rate"] = learning_rate

        system = System(parameters)
        system.run()

def read_configuration(config_path) -> dict:
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
    return parameters

if __name__ == "__main__":
    parameters_conf = read_configuration("parameters.conf")
    parametertuning_conf = read_configuration("parametertuning.conf")

    parameters = tune(parameters_conf, parametertuning_conf)
