import numpy as np
from typing import List, Tuple
from src.game_state.game_state import GameState
from src.game_manager.game_stage import GameStage
from src.neural_network.neural_network import NeuralNetwork
from src.resolver.resolver import Resolver


# NOTE: Important! This is for heads up only, for multiple opponents, we need to change the structure of the code
class SubtreeTraversalRollout:
    @staticmethod
    def subtree_traversal_rollout(self,
                                  state: GameState,
                                  p_dist: np.ndarray,
                                  o_dist: np.ndarray,
                                  end_stage: GameStage,
                                  end_depth: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parameters
        ----------
        state: GameState
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
        Tuple[np.ndarray, np.ndarray]: The expected value of the game for both players
            p_values
                Player action values
            o_values
                Opponent action values
        """
        if SubtreeTraversalRollout.showdown_state(state):
            p_values = SubtreeTraversalRollout.utility_matrix(
                state, p_dist, o_dist) * o_dist.T
            o_values = -p_values * \
                SubtreeTraversalRollout.utility_matrix(state, o_dist, p_dist)
        elif state.game_stage == end_stage and state.depth == end_depth:
            p_values, o_values = NeuralNetwork.run(
                state, state.game_stage, p_dist, o_dist)
        elif SubtreeTraversalRollout.player_state(state):
            p_values = np.zeros((0, 0))
            o_values = np.zeros((0, 0))
            for action in state.get_legal_actions():
                # TODO: This should be index by action (see pseudocode)
                p_dist[action] = Resolver.bayesianRangeUpdate(
                    p_dist, state, action, o_dist)
                # TODO: This should be index by action (see pseudocode)
                o_dist[action] = o_dist
                state = state.get_next_state(action)
                p_values[action], o_values[action] = SubtreeTraversalRollout.subtree_traversal_rollout(
                    state, p_dist, o_dist, end_stage, end_depth)
                hole_pairs = Resolver.generate_all_hole_pairs()  # TODO: Prob wrong, fix
                for pair in hole_pairs:
                    p_values[pair] += Resolver.strategy(
                        pair, action) * p_values[action][pair]
        else:
            p_values = np.zeros((0, 0))
            o_values = np.zeros((0, 0))
            events = state.get_events()
            for event in events:
                # TODO: This will not work
                p_dist[event] = p_dist
                o_dist[event] = o_dist
                p_values[event], o_values[event] = SubtreeTraversalRollout.subtree_traversal_rollout(
                    state, p_dist, o_dist, end_stage, end_depth)
                for pair in hole_pairs:
                    p_values[pair] += p_values[event][pair] / abs(events)
                    o_values[pair] += o_values[event][pair] / abs(events)
        return p_values, o_values

    @ staticmethod
    def showdown_state(self, state: GameState) -> bool:
        pass

    @ staticmethod
    def utility_matrix(self,
                       state: GameState,
                       player_hand_distribution: np.ndarray,
                       opponent_hand_distribution: np.ndarray) -> np.ndarray:
        pass

    @ staticmethod
    def player_state(self, state: GameState) -> np.ndarray:
        """
        Parameters
        ----------
        state: GameState
        Returns
        -------
        np.ndarray[bool]: The state of the players and whether they need to act
        """
        pass


def generateInitialSubtree(state, endStage, endDepth):
    # TODO: EIRIK
    return 0
