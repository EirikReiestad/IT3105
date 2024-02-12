import numpy as np
from typing import Tuple
from src.game_state.game_state import PublicGameState
from src.game_manager.game_stage import GameStage


# TODO!! EIRIKKKKKKKKKKKKKKKKOSELIG
class NeuralNetwork:
    @staticmethod
    def run(
        state: PublicGameState,
        stage: GameStage,
        player_hand_distribution: np.ndarray,
        opponent_hand_distribution: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
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
        if np.isnan(np.min(player_hand_distribution)):
            print(player_hand_distribution)
            raise ValueError("Player hand distribution is NaN")
        if np.isnan(np.min(opponent_hand_distribution)):
            raise ValueError("Opponent hand distribution is NaN")
        print(opponent_hand_distribution)
        pass
