import numpy as np
from src.poker_oracle.oracle import Oracle
from src.game_state.states import PlayerState

def updateStrategy(node):
    s = node.state
    for c in node.successors:
        updateStrategy(c)
    if type(s) == PlayerState:
        P = s
        all_hole_pairs = Oracle.generate_all_hole_pairs()
        sigma_s = np.zeros((len(all_hole_pairs), len(s.actions)))
        R_s = np.zeros((len(all_hole_pairs), len(s.actions)))
        R_s_plus = np.zeros((len(all_hole_pairs), len(s.actions)))

        for h in range(len(all_hole_pairs)):
            # MAYBE NEED TO ADD A SET OF ACTIONS IN PLAYER STATE
            for a in s.actions:
                R_s[h][a] = R_s[h][a] + [v_p(S(a))[h] - v_p(S)[h]]
                R_s_plus[h][a] = max(0, R_s)
        for h in range(len(all_hole_pairs)):
            for a in s.actions:
                sigma_s[h][a] = R_s_plus[h][a]/sum([R_s_plus[h, a_p] for a_p in s.actions])
    
    return sigma_s