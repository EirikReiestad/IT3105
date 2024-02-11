import numpy as np
from .subtree import SubTree
from src.game_state.game_state import PublicGameState
from src.game_state.player_state import PublicPlayerState, PublicPlayerState
from src.poker_oracle.oracle import Oracle
from src.game_manager.game_action import Action
from src.game_manager.game_stage import GameStage
from src.game_state.game_state import PublicGameState
from src.state_manager.manager import StateManager
from .subtree_travesal_rollout import SubtreeTraversalRollout
from .node import Node


class Resolver:
    def __init__(self):
        amount_of_pairs = len(Oracle.generate_all_hole_pairs())
        self.p_range: np.ndarray = np.full(
            (amount_of_pairs,), 1 / amount_of_pairs)
        self.o_range: np.ndarray = np.full(
            (amount_of_pairs,), 1 / amount_of_pairs)

    def bayesian_range_update(
        self,
        p_range: np.ndarray,
        action: Action,
        all_actions: list[Action],
        sigma_flat: np.ndarray,
    ):
        """
        Parameters
        ----------
        p_range: np.ndarray
            Player's hand distribution
        o_range: np.ndarray
            Opponent's hand distribution
        action: Action
            The action taken by the acting player
        sigma_flat: np.ndarray
            The average strategy over all rollouts
        """
        index = all_actions.index(action)
        prob_pair = p_range / np.sum(p_range)
        prob_act = np.sum(sigma_flat[index]) / np.sum(sigma_flat)
        prob_act_given_pair = sigma_flat[index] * prob_pair

        updated_prob = (prob_act_given_pair * prob_pair) / prob_act

        return updated_prob

    def sample_action_average_strategy(self, sigma_flat: np.ndarray, all_actions) -> Action:
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

        # Find the maximum value in action_probabilities
        max_action = np.max(action_probabilities)

        return all_actions[max_action]

    # def updateStrategy(self, node):
    def update_strategy(
        self,
        node: Node,
        p_value: np.ndarray,
        # o_value: np.ndarray,
        # state: PublicGameState,  # NOTE: This is not used
        p_range: np.ndarray,
        o_range: np.ndarray,
        end_stage: GameStage,
        end_depth: int,
    ) -> np.ndarray:
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
        p_range: np.ndarray
            Player's hand distribution
        o_range: np.ndarray
            Opponent's hand distribution
        end_stage: GameStage
            the stage to stop the rollout
        end_depth: int
            the depth to stop the rollout

        Returns
        -------
        np.ndarray: The current strategy matrix for the root
        """
        node_state = node.root
        for c in node.nodes:
            # self.update_strategy(c, o_value, p_value)
            self.update_strategy(c.strategy, p_value, p_range,
                                 o_range, end_stage, end_depth)

        # TODO: samme som subtreetraversal, remove true
        sigma_s = np.array([])
        if True:
            # P = s
            state_manager = StateManager(node_state)
            all_hole_pairs = Oracle.get_number_of_all_hole_pairs()

            all_actions = state_manager.get_legal_actions()
            sigma_s = np.zeros((len(all_hole_pairs), len(all_actions)))

            R_s = np.zeros((len(all_hole_pairs), len(all_actions)))
            R_s_plus = np.zeros((len(all_hole_pairs), len(all_actions)))

            for pair in all_hole_pairs:
                for action in all_actions:
                    index_pair = all_hole_pairs.index(pair)
                    index_action = state_manager.get_legal_actions().index(action)
                    new_node_state = state_manager.generate_state(action)
                    # TODO: USIKKER HVA SKJER HER, siden for å få ny så må jo subtreeTraversalRollout bli gjort

                    new_p_value, new_o_value = (
                        SubtreeTraversalRollout.subtree_traversal_rollout(
                            new_node_state, p_range, o_range, end_stage, end_depth
                        )
                    )
                    print("new_p_value", new_p_value)
                    # R_s[h][a] = R_s[h][a] + [v_1(s_new)[h] - v_1(s)[h]]
                    R_s[index_pair][index_action] += (
                        new_p_value[index_pair] - p_value[index_pair]
                    )
                    R_s_plus[index_pair][index_action] = max(
                        0, R_s[index_pair][index_action]
                    )
            for pair in all_hole_pairs:
                for action in all_actions:
                    index_pair = all_hole_pairs.index(pair)
                    index_action = all_actions.index(action)
                    # TODO: Fix a_p parameter name
                    node.strategy[index_pair][index_action] = R_s_plus[index_pair][
                        index_action
                    ] / sum(
                        [
                            R_s_plus[index_pair, all_actions.index(a_p)]
                            for a_p in all_actions
                        ]
                    )

        return sigma_s

    def resolve(
        self,
        state: PublicGameState,
        end_stage: GameStage,
        end_depth: int,
        num_rollouts: int,
    ):
        """
        Parameters
        ----------
        state: GameState
        p_range: np.ndarray
            Player's hand distribution
        o_range: np.ndarray
            Opponent's hand distribution
        end_stage: GameStage
        end_depth: int
        num_rollouts: int
        """
        # ▷ S = current state, r1 = Range of acting player, r2 = Range of other player, T = number of rollouts
        # Root ← GenerateInitialSubtree(S,EndStage,EndDepth)
        subtree = SubTree(state, end_stage, end_depth)
        sigmas = []
        # for t = 1 to T do ▷ T = number of rollouts
        for t in range(num_rollouts):
            # ← SubtreeTraversalRollout(S,r1,r2,EndStage,EndDepth) ▷ Returns evals for P1, P2 at root
            p_value, o_value = SubtreeTraversalRollout.subtree_traversal_rollout(
                state, self.p_range, self.o_range, end_stage, end_depth
            )
            # S ← UpdateStrategy(Root) ▷ Returns current strategy matrix for the root
            sigma = self.update_strategy(
                subtree, p_value, self.p_range, self.o_range, end_stage, end_depth
            )
            sigmas.append(sigma)

        state_manager = StateManager(state)
        all_actions = state_manager.get_legal_actions()

        # ▷ Generate the Average Strategy over all rollouts
        sigmas = np.array(sigmas)
        sigma_flat = np.mean(sigmas, axis=0)
        # ▷ Sample an action based on the average strategy
        action = self.sample_action_average_strategy(sigma_flat, all_actions)

        # ▷ r1(a∗) is presumed normalized.
        p_range_action = self.bayesian_range_update(
            self.p_range, action, all_actions, sigma_flat
        )

        self.p_range = p_range_action
        self.o_range = self.o_range

        return action
