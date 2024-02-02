import numpy as np
from src.resolver.subtree_travesal_rollout import subtreeTraversalRollout, generateInitialSubtree
from src.resolver.update_strategy import updateStrategy
from src.game_state.states import GameState
from src.poker_oracle.oracle import Oracle
from src.game_state.states import PlayerState
from src.game_state.actions import Action
from src.game_state.game_stage import GameStage


class Node:
    def __init__(self):
        self.state: GameState = None


class Resolver:
    def __init__(self):
        pass

        # TODO: Not sure if action can be Action or if it need to be int
    def bayesian_range_update(self, p_dist: np.ndarray, action: Action, sigma_flat: np.ndarray):
        """
        Parameters
        ----------
        p_dist: np.ndarray
            Player's hand distribution
        o_dist: np.ndarray
            Opponent's hand distribution
        action: Action
            The action taken by the acting player
        sigma_flat: np.ndarray
            The average strategy over all rollouts
        """
        # TODO: må fikse arrays og sånt
        prob_pair = p_dist / np.sum(p_dist)
        prob_act = np.sum(sigma_flat[action])/np.sum(sigma_flat)
        prob_act_given_pair = sigma_flat[action] * prob_pair

        updated_prob = (prob_act_given_pair * prob_pair) / prob_act

        return updated_prob

    def sample_action_average_strategy(self, sigma_flat: np.ndarray) -> Action:
        """
        Parameters
        ----------
        sigma_flat: np.ndarray
            The average strategy over all rollouts
        Returns
        -------
        Action: The action sampled based on the average strategy
        """
        # Compute the sum of probabilities across columns
        action_probabilities = np.sum(sigma_flat, axis=0)

        # Normalize probabilities to ensure they sum up to 1
        action_probabilities /= np.sum(action_probabilities)

        # Sample an action based on the computed probabilities
        sampled_action = np.random.choice(
            np.arange(len(action_probabilities)), p=action_probabilities)

        return sampled_action

    # def updateStrategy(self, node):
    # TODO: snake_case, fix parameter names
    def update_strategy(self,
                        node,
                        p_value: np.ndarray,
                        o_value: np.ndarray,
                        state: GameState,  # NOTE: This is not used
                        p_dist: np.ndarray,
                        o_dist: np.ndarray,
                        end_stage: GameStage,
                        end_depth: int) -> np.ndarray:
        """
        Parameters
        ----------
        node: Node
            The current node
        p_value: np.ndarray
            Player's action values
        o_value: np.ndarray
            Opponent's action values
        state: GameState
            The current state
        p_dist: np.ndarray
            Player's hand distribution
        o_dist: np.ndarray
            Opponent's hand distribution
        end_stage: GameStage
            the stage to stop the rollout
        end_depth: int
            the depth to stop the rollout

        Returns
        -------
        np.ndarray: The current strategy matrix for the root
        """
        node_state = node.state
        for c in node.successors:
            updateStrategy(c, o_value, p_value)
        if type(node_state) == PlayerState:
            # P = s
            all_hole_pairs = Oracle.generate_all_hole_pairs()
            sigma_s = np.zeros((len(all_hole_pairs), len(node_state.actions)))
            R_s = np.zeros((len(all_hole_pairs), len(node_state.actions)))
            R_s_plus = np.zeros((len(all_hole_pairs), len(node_state.actions)))

            for pair in range(len(all_hole_pairs)):
                # MAYBE NEED TO ADD A SET OF ACTIONS IN PLAYER STATE
                for action in node_state.actions:
                    new_node_state = node_state.copy()
                    new_node_state.act(action)
                    # USIKKER HVA SKJER HER, siden for å få ny så må jo subtreeTraversalRollout bli gjort
                    new_p_value, new_o_value = subtreeTraversalRollout(
                        new_node_state, p_dist, o_dist, end_stage, end_depth)
                    # R_s[h][a] = R_s[h][a] + [v_1(s_new)[h] - v_1(s)[h]]
                    # TODO: Fix R_s parameter name
                    R_s[pair][action] = R_s[pair][action] + \
                        [new_p_value[pair] - p_value[pair]]
                    R_s_plus[pair][action] = max(0, R_s)
            for pair in range(len(all_hole_pairs)):
                for action in node_state.actions:
                    # TODO: Fix a_p parameter name
                    sigma_s[pair][action] = R_s_plus[pair][action] / \
                        sum([R_s_plus[pair, a_p]
                             for a_p in node_state.actions])

        return sigma_s

        # TODO: fix T parameter name
    def resolve(self,
                state: GameState,
                p_dist: np.ndarray,
                o_dist: np.ndarray,
                end_stage: GameStage,
                end_depth: int,
                num_rollouts: int):
        """
        Parameters
        ----------
        state: GameState
        p_dist: np.ndarray
            Player's hand distribution
        o_dist: np.ndarray
            Opponent's hand distribution
        end_stage: GameStage
        end_depth: int
        num_rollouts: int
        """
        # ▷ S = current state, r1 = Range of acting player, r2 = Range of other player, T = number of rollouts
        # Root ← GenerateInitialSubtree(S,EndStage,EndDepth)
        root = generateInitialSubtree(state, end_stage, end_depth)
        sigmas = []
        # for t = 1 to T do ▷ T = number of rollouts
        for t in range(num_rollouts):
            # ← SubtreeTraversalRollout(S,r1,r2,EndStage,EndDepth) ▷ Returns evals for P1, P2 at root
            p_value, o_value = subtreeTraversalRollout(
                state, p_dist, o_dist, end_stage, end_depth)
            # S ← UpdateStrategy(Root) ▷ Returns current strategy matrix for the root
            sigma = updateStrategy(root, p_value, o_value)
            sigmas.append(sigma)

        # ▷ Generate the Average Strategy over all rollouts
        sigma_flat = 1/t*sum(sigmas)
        # ▷ Sample an action based on the average strategy
        action = self.sampleActionAverageStrategy(sigma_flat)
        # ▷ r1(a∗) is presumed normalized.
        p_dist_action = self.bayesianRangeUpdate(p_dist, action, sigma_flat)

        # MAYBE PLAYER SHOULD DO THE ACT?
        new_state = state.act(action)

        # NEW STATE S' made from strategy 'a'
        return action, new_state, p_dist_action, o_dist
