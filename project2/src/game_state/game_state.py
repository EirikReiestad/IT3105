from dataclasses import dataclass
from typing import List
from src.game_manager.game_stage import GameStage
from .player_state import PublicPlayerState, PrivatePlayerState
from .board_state import PublicBoardState, PrivateBoardState


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
        self._current_player_index = (
            self.board_state.dealer + 1) % len(self.player_states)

    @property
    def current_player_index(self):
        return self._current_player_index

    @current_player_index.setter
    def current_player_index(self, value):
        self._current_player_index = (value + 1) % len(self.player_states)


@dataclass
class PrivateGameState:
    def __init__(
        self,
        player_states: List[PrivatePlayerState],
        board_state: PrivateBoardState,
        game_stage: GameStage,
    ):
        self.player_states = player_states
        self.board_state = board_state
        self.game_stage = game_stage
        self.current_player_index = (
            self.board_state.dealer + 1) % len(self.player_states)
