from plants.plant import Plant

class Plant3(Plant):
    def __init__(self, init_population):
        self.init_population = init_population

    def reset(self):
        return {
            'P':self.init_population,
            'k':0
        }

    def run_one_epoch(self, state: dict, control_signal: float, noise: float) -> dict:
        state = state.copy()
        state['k'] += control_signal + noise
        state['P'] += state['k']*state['P']
        state['P'] = max(0, state['P'])
        return state, state['P']