'''
Plant: The plant is any system whose behavior the controller will try to regulate.
'''


class Plant:
    def __init__(self):
        return self

    def run_one_step(self, external_disturbance: float, control_signal: float) -> float:
        '''
        Run one step of the PID controller.

        Parameters:
            external_disturbance (float): The external disturbance to the system.
                Reffered to as 'D' in the lecture notes.
            control_signal (float): The derivative to a control output.
                Reffered to as 'U' in the lecture notes.

        Returns:
            output (float): The output of the plant.
                Reffered to as 'Y' in the lecture notes.
        '''

        # TODO: Make this correct. The code below is just a placeholder.
        return external_disturbance + control_signal
