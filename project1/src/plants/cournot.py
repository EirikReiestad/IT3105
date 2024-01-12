from plants.plant import Plant

class Cournot(Plant):
    def __init__(self, pmax, cm):
        self.pmax = pmax
        self.cm = cm

    def reset(self):
        return {
            'q1': 0,
            'q2': 0
        }
    
    def run_one_epoch(self, state: dict, control_signal: float, noise: float) -> dict:
        state = state.copy()
        #1. q1 updates based on U.
        state['q1'] = control_signal + state['q1']
        if state['q1'] < 0:
            state['q1'] = 0
        elif state['q1'] > 1:
            state['q1'] = 1
        #2. q2 updates based on D.
        state['q2'] = noise + state['q2']
        if state['q2'] < 0:
            state['q2'] = 0
        elif state['q2'] > 1:
            state['q2'] = 1
        #3. q = q1 + q2
        q = state['q1'] + state['q2']
        #4. p(q) = pmax − q
        p = self.pmax-q
        p_1 = state['q1'] *(p-self.cm)
        return state, p_1