from controller import Controller

class PIDController(Controller):
    def __init__(self):
        pass

    def initialize(self, k_p=0.1, k_d=0.1, k_i=0.3):
        '''
        Initialize the controller.

        Parameters:
            
        '''
        return {
            'k_p':k_p,
            'k_d':k_d,
            'k_i':k_i
        }

    def run_one_epoch(self, error: float) -> float:
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

    def calculate_control_signal(self, error: float) -> float:
        '''
        Calculate the control signal for the PID controller.
        Formula: U = Kp * E + Kd(dE/dt) + Ki * Sum(E)

        Parameters:
            error (float): The error to the system.

        Returns:
            control_signal (float): The derivative to a control output.
        '''
        pass
