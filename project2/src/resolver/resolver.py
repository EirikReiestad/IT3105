from typing import List
import numpy as np
import copy
from typing import Dict
from src.game_state.game_state import PublicGameState
from src.poker_oracle.oracle import Oracle
from src.game_manager.game_action import Action
from src.game_manager.game_stage import GameStage
from src.state_manager.manager import StateManager
from .subtree_traversal_rollout import SubtreeTraversalRollout
from .node import Node
from src.setup_logger import setup_logger

logger = setup_logger()


class Resolver:
    def __init__(self, total_players: int, networks: Dict = None):
        amount_of_pairs = len(Oracle.generate_all_hole_pairs())
        self.p_range: np.ndarray = np.random.rand(amount_of_pairs)
        self.o_range: np.ndarray = np.random.rand(amount_of_pairs)
        # TODO: REMEMBER TO REMOVE UNCOMMENT
        self.str = SubtreeTraversalRollout(total_players, networks)
        self.count = 0

    @staticmethod
    def bayesian_range_update(
        p_range: np.ndarray,
        action: Action,
        all_actions: list[Action],
        sigma_flat: np.ndarray,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        p_range: np.ndarray
            Player's hand distribution
        action: Action
            The action taken by the acting player
        all_actions: list[Action]
            All possible actions
        sigma_flat: np.ndarray
            The average strategy over all rollouts

        Returns
        -------
        np.ndarray: The updated range for the acting player
        """
        # logger.debug("Bayesian Range Update")
        if np.isnan(np.min(p_range)):
            raise ValueError("Player hand distribution is NaN")
        if np.isnan(np.min(sigma_flat)):
            raise ValueError("Sigma flat distribution is NaN")

        index = all_actions.index(action)
        prob_pair = p_range / np.sum(p_range)
        if np.sum(sigma_flat) == 0:
            raise ValueError("The sum of sigma_flat is 0")
        prob_act = np.sum(sigma_flat[index]) / np.sum(sigma_flat)
        # [:, index] gives the column
        prob_act_given_pair = sigma_flat[:, index] * prob_pair

        updated_prob = (prob_act_given_pair * prob_pair) / prob_act

        if np.isnan(np.min(updated_prob)):
            raise ValueError("Updated hand distribution is NaN")

        updated_prob = updated_prob / sum(updated_prob)

        return updated_prob

    def sample_action_average_strategy(
        self, sigma_flat: np.ndarray, all_actions
    ) -> Action:
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
        # print(action_probabilities)

        # Normalize probabilities to ensure they sum up to 1
        action_probabilities /= np.sum(action_probabilities)
        # print(all_actions)
        # print(action_probabilities)
        # print(action_probabilities.shape, np.argmax(action_probabilities))

        max_prob = np.max(action_probabilities)

        # Find indices of actions with the maximum probability
        max_indices = np.where(action_probabilities == max_prob)[0]

        # Randomly select one of the actions with the maximum probability
        selected_action = np.random.choice(max_indices)

        return all_actions[selected_action]
        # Find the maximum value in action_probabilities
        # return all_actions[np.argmax(action_probabilities)]

    # def updateStrategy(self, node):
    def update_strategy(
        self,
        node: Node,
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
        np.ndarray: The current strategy matrix for the node
        """

        if not isinstance(node, Node):
            raise ValueError("Node is not an instance of Node")
        for i, c in enumerate(node.children):
            node.children[i].strategy = self.update_strategy(
                c,
                end_stage,
                end_depth,
            )

        p_value, o_value, p_values_all_act, o_values_all_act = (
            self.str.subtree_traversal_rollout(
                node, self.p_range, self.o_range, end_stage, end_depth)
        )

        if node.depth >= end_depth or node.state.game_stage == end_stage:
            return node.strategy
        if not node.state_manager.chance_event:
            # P = s
            all_hole_pairs = Oracle.generate_all_hole_pairs(shuffle=False)
            num_all_hole_pairs = len(all_hole_pairs)

            all_actions = node.available_actions
            num_all_actions = len(all_actions)

            # NOTE: Was originally zeros, but that would cause a division by zero error
            R_s = np.zeros((num_all_hole_pairs, num_all_actions))
            R_s_plus = np.zeros((num_all_hole_pairs, num_all_actions))

            for pair in all_hole_pairs:
                for i, action in enumerate(all_actions):
                    index_pair = all_hole_pairs.index(pair)
                    index_action = all_actions.index(action)

                    new_p_value = p_values_all_act[i]
                    # R_s[h][a] = R_s[h][a] + [v_1(s_new)[h] - v_1(s)[h]]

                    R_s[index_pair][index_action] += (
                        new_p_value[index_pair] - p_value[index_pair]
                    )*100000000000000000000

                    if new_p_value[index_pair] - p_value[index_pair] != 0:
                        # print(new_p_value[index_pair] - p_value[index_pair])
                        pass

                    # print(R_s[index_pair][index_action])
                    R_s_plus[index_pair][index_action] = max(
                        0.0000000001,
                        R_s[index_pair][index_action],  # TODO: This is a hack
                    )
            for pair_idx, pair in enumerate(all_hole_pairs):
                for action_idx, action in enumerate(all_actions):
                    # index_pair = all_hole_pairs.index(pair)
                    # index_action = all_actions.index(action)
                    # NOTE: Same as in subtreetraversal, assuming that the pair order is the same as the index
                    # and that the action order is the same in every case
                    R_s_sum = sum(R_s_plus[pair_idx])
                    if R_s_sum == 0:
                        raise ValueError("The sum of R_s_plus is 0")
                    node.strategy[pair_idx][action_idx] = (
                        R_s_plus[pair_idx][action_idx] / R_s_sum
                    )
        return node.strategy

    def resolve(
        self,
        state: PublicGameState,
        end_stage: GameStage,
        end_depth: int,
        num_rollouts: int,
        verbose: bool = False
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

        Returns
        -------
        action: Action
            The action sampled based on the average strategy
        """
        # ▷ S = current state, r1 = Range of acting player, r2 = Range of other player, T = number of rollouts
        # Root ← GenerateInitialSubtree(S,EndStage,EndDepth)
        node = Node(copy.deepcopy(state), end_stage, end_depth, 0)
        sigmas = []  # a list to hold the strategy matrix for each rollout
        # for t = 1 to T do ▷ T = number of rollouts
        print(node.available_actions)

        for t in range(num_rollouts):
            # ← SubtreeTraversalRollout(S,r1,r2,EndStage,EndDepth) ▷ Returns evals for P1, P2 at root
            if verbose:
                print("Rollout:", t)
            # S ← UpdateStrategy(Root) ▷ Returns current strategy matrix for the root
            strategy = self.update_strategy(
                node,
                end_stage,
                end_depth,
            )
            sigmas.append(strategy)

        state_manager = StateManager(copy.deepcopy(state))
        all_actions = state_manager.get_legal_actions()

        # ▷ Generate the Average Strategy over all rollouts
        sigmas = np.array(sigmas)
        sigma_flat = np.mean(sigmas, axis=0)
        # ▷ Sample an action based on the average strategy
        action = self.sample_action_average_strategy(sigma_flat, all_actions)
        # ▷ r1(a∗) is presumed normalized.

        p_range_action = Resolver.bayesian_range_update(
            self.p_range, action, all_actions, sigma_flat
        )

        self.p_range = p_range_action
        self.o_range = self.o_range

        return action
