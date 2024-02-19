from dataclasses import dataclass
from itertools import combinations
from typing import List
from src.game_manager.game_stage import GameStage
from .player_state import PublicPlayerState, PrivatePlayerState
from .board_state import PublicBoardState, PrivateBoardState
from ..poker_oracle.deck import Deck

from src.config import Config

config = Config()

@dataclass
class PublicGameState:
    def __init__(
        self,
        player_states: List[PublicPlayerState],
        board_state: PublicBoardState,
        game_stage: GameStage,
        current_player_index: int,
        buy_in: int,
        check_count: int,
    ):
        self.player_states = player_states  # A list of PlayerState instances
        self.board_state = board_state  # An instance of BoardState
        self.game_stage = game_stage  # An instance of GameStage
        self.current_player_index = current_player_index
        self.buy_in = buy_in
        self.check_count = check_count

    def get_events(self):
        deck = Deck(shuffle=False)
        
        if self.game_stage == GameStage.Flop:
            events = combinations(deck.stack, 3)
        elif self.game_stage == GameStage.Turn or self.game_stage == GameStage.Flop:
            events = deck.stack

        return events


@dataclass
class PrivateGameState:
    def __init__(
        self,
        player_states: List[PrivatePlayerState],
        board_state: PrivateBoardState,
        game_stage: GameStage,
        current_player_index: int,
        buy_in: int,
    ):
        self.player_states = player_states
        self.board_state = board_state
        self.game_stage = game_stage
        self.current_player_index = current_player_index
        self.buy_in = buy_in
        self.check_count = 0
