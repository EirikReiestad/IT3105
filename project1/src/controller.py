class Controller:
    def __init__(self):
        self.error_history = []

    def add_error(self, error: float):
        self.error_history.append(error)


class PID(Controller):
    def __init__(self, Kp: float, Ki: float, Kd: float):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        return self

    def run_one_step(self, error: float) -> float:
        '''
        Run one step of the PID controller.
        Parameters:
            error (float): The error to the system.
                Reffered to as 'E' in the lecture notes.

        Returns:
            control_signal (float): The derivative to a control output.
                Reffered to as 'U' in the lecture notes.
        '''
        return self._calculate_control_signal(error)

    def _calculate_control_signal(self, error: float) -> float:
        '''
        Calculate the control signal for the PID controller.
        Formula: U = Kp * E + Kd(dE/dt) + Ki * Sum(E)

        Parameters:
            error (float): The error to the system.

        Returns:
            control_signal (float): The derivative to a control output.
        '''
        pass
