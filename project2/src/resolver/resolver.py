import numpy as np
from src.resolver.subtree_travesal_rollout import subtreeTraversalRollout, generateInitialSubtree
from src.resolver.update_strategy import updateStrategy
from src.game_state.states import GameState
from src.poker_oracle.oracle import Oracle
from src.game_state.states import PlayerState

class Node:
    def __init__(self):
        self.state: GameState = None

class Resolver:
    def __init__(self):
        pass
    
    def bayesianRangeUpdate(self, r1, a, sigma_flat):
        # TODO: må fikse arrays og sånt
        prob_pair = r1 / np.sum(r1)
        prob_act = np.sum(sigma_flat[a])/np.sum(sigma_flat)
        prob_act_given_pair = sigma_flat[a] * prob_pair
        
        updated_prob = (prob_act_given_pair * prob_pair) / prob_act
        
        return updated_prob
    
    def sampleActionAverageStrategy(self, sigma_flat):
        # Compute the sum of probabilities across columns
        action_probabilities = np.sum(sigma_flat, axis=0)
        
        # Normalize probabilities to ensure they sum up to 1
        action_probabilities /= np.sum(action_probabilities)
        
        # Sample an action based on the computed probabilities
        sampled_action = np.random.choice(np.arange(len(action_probabilities)), p=action_probabilities)
        
        return sampled_action
    
    # def updateStrategy(self, node):
    def updateStrategy(self, node, v_1, v_2, S, r1, r2, endStage, endDepth):
        s = node.state
        for c in node.successors:
            updateStrategy(c, v_2, v_1)
        if type(s) == PlayerState:
            # P = s
            all_hole_pairs = Oracle.generate_all_hole_pairs()
            sigma_s = np.zeros((len(all_hole_pairs), len(s.actions)))
            R_s = np.zeros((len(all_hole_pairs), len(s.actions)))
            R_s_plus = np.zeros((len(all_hole_pairs), len(s.actions)))

            for h in range(len(all_hole_pairs)):
                # MAYBE NEED TO ADD A SET OF ACTIONS IN PLAYER STATE
                for a in s.actions:
                    s_new = s.copy()
                    s_new.act(a)
                    # USIKKER HVA SKJER HER, siden for å få ny så må jo subtreeTraversalRollout bli gjort
                    v_1_s_new, v_2_s_new = subtreeTraversalRollout(s_new, r1, r2, endStage, endDepth)
                    # R_s[h][a] = R_s[h][a] + [v_1(s_new)[h] - v_1(s)[h]]
                    R_s[h][a] = R_s[h][a] + [v_1_s_new[h] - v_1[h]]
                    R_s_plus[h][a] = max(0, R_s)
            for h in range(len(all_hole_pairs)):
                for a in s.actions:
                    sigma_s[h][a] = R_s_plus[h][a]/sum([R_s_plus[h, a_p] for a_p in s.actions])
        
        return sigma_s

    def resolve(self, S, r1, r2, endStage, endDepth, T):
        # ▷ S = current state, r1 = Range of acting player, r2 = Range of other player, T = number of rollouts
        # Root ← GenerateInitialSubtree(S,EndStage,EndDepth)
        root = generateInitialSubtree(S, endStage, endDepth)
        sigmas = []
        # for t = 1 to T do ▷ T = number of rollouts
        for t in range(T):
            # ← SubtreeTraversalRollout(S,r1,r2,EndStage,EndDepth) ▷ Returns evals for P1, P2 at root
            v_1, v_2 = subtreeTraversalRollout(S, r1, r2, endStage, endDepth)
            # S ← UpdateStrategy(Root) ▷ Returns current strategy matrix for the root
            sigma = updateStrategy(root, v_1, v_2)
            sigmas.append(sigma)

        # ▷ Generate the Average Strategy over all rollouts
        sigma_flat = 1/t*sum(sigmas)
        # ▷ Sample an action based on the average strategy
        a = self.sampleActionAverageStrategy(sigma_flat)
        # ▷ r1(a∗) is presumed normalized.
        r1_a = self.bayesianRangeUpdate(r1, a, sigma_flat)

        ## MAYBE PLAYER SHOULD DO THE ACT?
        new_state = S.act(a)

        # NEW STATE S' made from strategy 'a'
        return a, new_state, r1_a, r2