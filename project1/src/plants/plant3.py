from plants.plant import Plant

class Plant3(Plant):
    def __init__(self, init_population):
        self.init_population = init_population

    def reset(self):
        """
        Return the initial state of the plant
        P: Population
        k: Growth rate
        """
        return {
            'P':self.init_population,
            'k':0
        }

    def run_one_epoch(self, state: dict, control_signal: float, noise: float) -> dict:
        """
        Update the plant's state based on the given control signal and noise 
        """
        state = state.copy()
        # Change the growth rate based on the control signal
        state['k'] += control_signal + noise
        # Calculate the delta population and change the current population based on that
        state['P'] += state['k']*state['P']
        # Limit so that population doesn't become 0
        state['P'] = max(0, state['P'])
        return state, state['P']