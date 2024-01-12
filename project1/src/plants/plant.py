"""
Plant: The plant is any system whose behavior the controller will try to regulate.
"""


class Plant:
    def __init__(self):
        pass

    def reset(self):
        """
        Reset the plant to its initial state.
        """
        pass

    def run_one_epoch(self, state: dict, control_signal: float, noise: float) -> dict:
        """
        Run one step of the PID controller.

        Parameters:
            state (dict): The state of the system.
            control_signal (float): The derivative to a control output.
            noise (float): The noise to the system.

        Returns:
            state (dict): The state of the system.
        """
        pass
