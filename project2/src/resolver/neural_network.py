import numpy as np
from typing import Tuple
from src.game_state.game_state import GameState
from src.game_manager.game_stage import GameStage


class NeuralNetwork:
    @staticmethod
    def run(self,
            state: GameState,
            stage: GameStage,
            player_hand_distribution: np.ndarray,
            opponent_hand_distribution: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parameters
        ----------
        state: GameState
        stage: GameStage
        player_hand_distribution: np.ndarray
            hand distribution of player
        opponent_hand_distribution: np.ndarray
            hand distribution of opponent
        Returns
        -------
        np.ndarray: The expected value of the game for the player
        """
        pass
