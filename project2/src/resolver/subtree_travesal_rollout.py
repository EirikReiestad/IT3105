import numpy as np
from typing import List, Tuple
from src.game_state.game_state import PublicGameState
from src.game_manager.game_stage import GameStage
from .neural_network import NeuralNetwork

from src.poker_oracle.oracle import Oracle
from src.state_manager.manager import StateManager
from .node import Node

# TODO: Fix circular import by refactoring instead
from . import resolver


# NOTE: Important! This is for heads up only, for multiple opponents, we need to change the structure of the code
# TODO: Fix that r2 (o_range) is not updated in the loop
class SubtreeTraversalRollout:
    @staticmethod
    def subtree_traversal_rollout(
        node: Node,
        p_range: np.ndarray,
        o_range: np.ndarray,
        end_stage: GameStage,
        end_depth: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parameters
        ----------
        state: GameState
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
        Tuple[np.ndarray, np.ndarray]: The expected value of the game for both players
            p_values
                Player action values
            o_values
                Opponent action values
        """
        if node.game_stage == GameStage.Showdown:
            utility_matrix = Oracle.utility_matrix_generator(node.board.cards)
            p_values = utility_matrix * o_range.T
            o_values = -p_values * utility_matrix
        elif node.game_stage == end_stage and node.depth == end_depth:
            p_values, o_values = NeuralNetwork.run(
                node, node.game_stage, p_range, o_range
            )
        # TODO: Check if chance event (consider adding the chance events to the game state)
        elif SubtreeTraversalRollout.player_state(node):
            # Value vector is the range times the utility matrix (or the hole pairs)
            hole_pairs = Oracle.generate_all_hole_pairs()
            p_values = np.zeros((len(hole_pairs),))
            o_values = np.zeros((len(hole_pairs),))
            state_manager = StateManager(node)
            strategy = node.strategy
            res = resolver.Resolver()
            for action in state_manager.get_legal_actions():
                p_range = resolver.Resolver.bayesian_range_update(
                    p_range, action, state_manager.get_legal_actions(), strategy
                )
                o_range = o_range
                state = state_manager.generate_state(action)
                p_values_new, o_values_new = (
                    SubtreeTraversalRollout.subtree_traversal_rollout(
                        state, p_range, o_range, end_stage, end_depth
                    )
                )
                hole_pairs = Oracle.generate_all_hole_pairs()
                strategy_matrix = resolver.Resolver.generate_strategy_matrix()
                for pair in hole_pairs:
                    # TODO: FIKSE INDEKS
                    p_values[pair] += strategy_matrix[pair,
                                                      action] * p_values_new[pair]
                    o_values[pair] += strategy_matrix[pair,
                                                      action] * o_values_new[pair]
        else:
            print("Else")
            # TODO: Add chance event ?
            hole_pairs = Oracle.generate_all_hole_pairs()
            p_values = np.zeros((len(hole_pairs),))
            o_values = np.zeros((len(hole_pairs),))
            return p_values, o_values
            # TODO: FIX events
            events = state.get_events()
            for event in events:
                # TODO: This will not work, INDEKS
                p_range[event] = p_range
                o_range[event] = o_range
                p_values[event], o_values[event] = (
                    SubtreeTraversalRollout.subtree_traversal_rollout(
                        state, p_range, o_range, end_stage, end_depth
                    )
                )
                for pair in hole_pairs:
                    p_values[pair] += p_values[event][pair] / abs(events)
                    o_values[pair] += o_values[event][pair] / abs(events)
        return p_values, o_values

    @staticmethod
    def player_state(state: PublicGameState) -> bool:
        """
        Parameters
        ----------
        state: GameState
        Returns
        -------
        bool: The state of the player and whether they need to act
        """
        return True
