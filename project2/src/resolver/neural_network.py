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
        p_range: np.ndarray,
        o_range: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parameters
        ----------
        state: GameState
        stage: GameStage
        p_range: np.ndarray
            hand distribution of player
        o_range: np.ndarray
            hand distribution of opponent

        Returns
        -------
        np.ndarray: The expected value of the game for the player
        """
        if np.isnan(np.min(p_range)):
            print(p_range)
            raise ValueError("Player hand distribution is NaN")
        if np.isnan(np.min(o_range)):
            print(o_range)
            raise ValueError("Opponent hand distribution is NaN")
        return p_range, o_range
