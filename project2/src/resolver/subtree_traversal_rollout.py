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

from src.setup_logger import setup_logger

logger = setup_logger()


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
        state: GameState p_range: np.ndarray
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
        logger.debug(
            "Subtree Traversal Rollout (depth = {})".format(node.depth))
        if node.state.game_stage == GameStage.Showdown:
            logger.debug("Showdown")
            utility_matrix = Oracle.utility_matrix_generator(
                node.state.board.cards)
            p_values = utility_matrix * o_range.T
            o_values = -p_values * utility_matrix
        elif node.state.game_stage == end_stage or node.depth == end_depth:
            logger.debug("End stage or depth")
            # TODO: Just return some simple heuristic for now
            p_values, o_values = NeuralNetwork.run(
                node, node.state.game_stage, p_range, o_range
            )
        # TODO: Check if chance event (consider adding the chance events to the game state)
        elif node.is_player:
            logger.debug("Player state")
            # Value vector is the range times the utility matrix (or the hole pairs)
            hole_pairs = Oracle.generate_all_hole_pairs()
            p_values = np.zeros((len(hole_pairs),))
            o_values = np.zeros((len(hole_pairs),))
            state_manager = StateManager(node.state)
            all_actions = node.available_actions
            for action_idx, action in enumerate(all_actions):
                if len(all_actions) != node.strategy.shape[1]:
                    raise ValueError(
                        "The number of actions does not match the strategy matrix, {} != {}".format(
                            len(all_actions), node.strategy.shape[1]))
                p_range = resolver.Resolver.bayesian_range_update(
                    p_range, action, all_actions, node.strategy
                )
                o_range = o_range
                state: PublicGameState = state_manager.generate_state(action)
                new_node = Node(state, end_stage, end_depth, node.depth + 1)
                logger.debug("Place 1")
                p_values_new, o_values_new = (
                    SubtreeTraversalRollout.subtree_traversal_rollout(
                        new_node, p_range, o_range, end_stage, end_depth
                    )
                )
                hole_pairs = Oracle.generate_all_hole_pairs()
                for pair_idx, pair in enumerate(hole_pairs):
                    # NOTE: Assuming that the pair order is the same as the index
                    # NOTE: Assuming the action order is the same in every case
                    p_values[pair_idx] += (
                        node.strategy[pair_idx][action_idx] *
                        p_values_new[pair_idx]
                    )
                    o_values[pair_idx] += (
                        node.strategy[pair_idx][action_idx] *
                        o_values_new[pair_idx]
                    )
        else:
            # TODO: Add chance event ?
            pv_values = np.zeros((len(p_range),))
            ov_values = np.zeros((len(o_range),))
            
            # Get all cards which are not on the board
            events = state.get_events()
            hole_pairs = Oracle.generate_all_hole_pairs()
            for event in events:
                pr_range_event = p_range.copy()
                or_range_event = o_range.copy()

                # Setting the range of all hole pairs with this event card to 0
                for pair_idx, pair in enumerate(hole_pairs):
                    if event in pair:
                        pr_range_event[pair_idx] = 0
                        or_range_event[pair_idx] = 0

                p_range_event, o_range_event = (
                    SubtreeTraversalRollout.subtree_traversal_rollout(
                        state, pr_range_event, or_range_event, end_stage, end_depth
                    )
                )
                for pair_idx in range(len(hole_pairs)):
                    pv_values[pair_idx] += p_range_event[pair_idx] / len(events)
                    ov_values[pair_idx] += o_range_event[pair_idx] / len(events)
        return p_values, o_values

    @ staticmethod
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
