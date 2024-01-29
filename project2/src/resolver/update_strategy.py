from src.poker_oracle.oracle import Oracle

def updateStrategy(node):
    s = node.state
    for c in node.successors:
        updateStrategy(c)
    if playerState(S):
        P = Player(S)
        for h in Oracle.generate_all_hole_pairs():
            for a in Actions(S):
                R_s = R_s + [v_p(S(a))[h] - v_p(S)[h]]
                R_s_plus = max(0, R_s(h,a))
        for h in Oracle.generate_all_hole_pairs():
            for a in Actions(S):
                sigma_s(h, a) = R_s_plus(h,a)/sum([R_s_plus(h, a) for a in Actions(S)])
    
    return sigma_s