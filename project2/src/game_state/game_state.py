from dataclasses import dataclass
from typing import List
from src.game_manager.game_stage import GameStage
from .player_state import PublicPlayerState
from .board_state import PublicBoardState


@dataclass
class PublicGameState:
    def __init__(
        self,
        player_states: List[PublicPlayerState],
        board_state: PublicBoardState,
        game_stage: GameStage,
    ):
        self.player_states = player_states  # A list of PlayerState instances
        self.board_state = board_state  # An instance of BoardState
        self.game_stage = game_stage  # An instance of GameStage
