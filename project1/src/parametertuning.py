import sys
import os
import configparser
import numpy as np
from itertools import product

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


def tune(system_params, tuning_params):
    print("===== Running Parameter Tuning =====")
    hidden_layers = [[i, j] for i in range(
        tuning_params["hidden_layers"][0],
        tuning_params["hidden_layers"][1] + 1)
        for j in range(
            tuning_params["hidden_layers"][2],
            tuning_params["hidden_layers"][3] + 1)]
    # activation_func = list(list(np.linspace(tuning_params["activation_func"][0],
    # tuning_params["activation_func"][1],
    # tuning_params["activation_func"][2],
    # dtype=int)) for i in range(len(hidden_layers)))
    activation_func = tuning_params["activation_func"]
    epochs = list(np.linspace(tuning_params["epochs"][0],
                              tuning_params["epochs"][1],
                              tuning_params["epochs"][2],
                              dtype=int))
    sim_timesteps = list(
        np.linspace(tuning_params["sim_timesteps"][0],
                    tuning_params["sim_timesteps"][1],
                    tuning_params["sim_timesteps"][2],
                    dtype=int))
    learning_rate = list(
        np.linspace(tuning_params["learning_rate"][0],
                    tuning_params["learning_rate"][1],
                    tuning_params["learning_rate"][2]))

    param_space = list(product(hidden_layers, activation_func,
                       epochs, sim_timesteps, learning_rate))

    best_params = None
    lowest_mse = np.inf

    total_iterations = len(param_space)

    one_percent = total_iterations // 100

    for i, params in enumerate(param_space):
        if i % one_percent == 0:
            print(f"{i}/{total_iterations} iterations done")
        parameters = system_params.copy()
        (hidden_layers, activation_func, epochs,
            sim_timesteps, learning_rate) = params

        system = System(parameters, visualize=False)
        mse = system.run()

        if mse[-1] > lowest_mse:
            continue

        lowest_mse = mse[-1]
        best_params = params

        print(
            f"\n===== Running with parameters ({i}/{total_iterations}): =====")
        print("Hidden Layers: ", hidden_layers)
        print("Activation Function: ", activation_func)
        print("Epochs: ", epochs)
        print("Simulation Timesteps: ", sim_timesteps)
        print("Learning Rate: ", learning_rate)
        print("MSE: ", lowest_mse)

        parameters["hidden_layers"] = hidden_layers
        parameters["activation_func"] = activation_func
        parameters["epochs"] = epochs
        parameters["sim_timesteps"] = sim_timesteps
        parameters["learning_rate"] = learning_rate

    print("\n===== Best Parameters: =====")
    print("Hidden Layers: ", best_params[0])
    print("Activation Function: ", best_params[1])
    print("Epochs: ", best_params[2])
    print("Simulation Timesteps: ", best_params[3])
    print("Learning Rate: ", best_params[4])
    print("MSE: ", lowest_mse)
    return best_params


def read_configuration(config_path, section="DEFAULT") -> dict:
    config = configparser.RawConfigParser()
    config.read(config_path)
    parameters = dict(config.items(section))

    for k, v in parameters.items():
        try:
            parameters[k] = eval(v)
        except ValueError:
            try:
                parameters[k] = int(v)
            except ValueError:
                try:
                    parameters[k] = float(v)
                except ValueError:
                    pass
    return parameters


if __name__ == "__main__":
    from src.system import System
    parameters_conf = read_configuration("parameters.conf", "PLANT3_NN")
    parametertuning_conf = read_configuration("parametertuning.conf")

    parameters = tune(parameters_conf, parametertuning_conf)
