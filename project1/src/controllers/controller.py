class Controller:
    def __init__(self):
        self.error_history = []

    def initialize(self):
        pass
    
    def add_error(self, error: float):
        self.error_history.append(error)
    
    def run_one_epoch(self, state):
        pass

    def calculate_control_signal(self, error: float):
        pass