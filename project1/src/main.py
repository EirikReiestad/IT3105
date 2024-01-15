from system import System
import os
import sys
import configparser

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

if __name__ == "__main__":
    config_path = "parameters.conf"

    config = configparser.RawConfigParser()
    config.read(config_path)

    parameters = dict(config.items("PLANT2_NN"))

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

    system = System(parameters, True)
    system.run()
