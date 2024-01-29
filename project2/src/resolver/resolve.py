import numpy as np
from src.resolver.subtree_travesal_rollout import subtreeTraversalRollout
from src.resolver.update_strategy import updateStrategy
from src.game_state.states import GameState

# TEMPORARY ONLY USE EIRIK'S LATER
class Node:
    def __init__(self):
        self.state: GameState = None

def generateInitialSubtree(S, endStage, endDepth):
    return 0

def bayesianRangeUpdate(r1, a, sigma_flat):
    # TODO: må fikse arrays og sånt
    prob_pair = r1 / sum(r1)
    prob_act = sum(sigma_flat[a])/sum(sigma_flat)
    prob_act_given_pair = sigma_flat[a] * prob_pair
    
    updated_prob = (prob_act_given_pair * prob_pair) / prob_act
    
    return updated_prob

def sampleActionAverageStrategy(sigma_flat):
    # Compute the sum of probabilities across columns
    action_probabilities = np.sum(sigma_flat, axis=0)
    
    # Normalize probabilities to ensure they sum up to 1
    action_probabilities /= np.sum(action_probabilities)
    
    # Sample an action based on the computed probabilities
    sampled_action = np.random.choice(np.arange(len(action_probabilities)), p=action_probabilities)
    
    return sampled_action

# procedure RESOLVE(S,r1,r2,EndStage,EndDepth,T)
def resolve(S, r1, r2, endStage, endDepth, T):
    # ▷ S = current state, r1 = Range of acting player, r2 = Range of other player, T = number of rollouts
    # Root ← GenerateInitialSubtree(S,EndStage,EndDepth)
    root = generateInitialSubtree(S, endStage, endDepth)
    sigmas = []
    # for t = 1 to T do ▷ T = number of rollouts
    for t in range(T):
        # ← SubtreeTraversalRollout(S,r1,r2,EndStage,EndDepth) ▷ Returns evals for P1, P2 at root
        v_1, v_2 = subtreeTraversalRollout(S, r1, r2, endStage, endDepth)
        # S ← UpdateStrategy(Root) ▷ Returns current strategy matrix for the root
        sigma = updateStrategy(root, v_1)
        sigmas.append(sigma)

    # ▷ Generate the Average Strategy over all rollouts
    sigma_flat = 1/t*sum(sigmas)
    # ▷ Sample an action based on the average strategy
    a = sampleActionAverageStrategy(sigma_flat)
    # ▷ r1(a∗) is presumed normalized.
    r1_a = bayesianRangeUpdate(r1, a, sigma_flat)

    ## MAYBE PLAYER SHOULD DO THE ACT?
    new_state = S.act(a)

    # NEW STATE S' made from strategy 'a'
    return a, new_state, r1_a, r2