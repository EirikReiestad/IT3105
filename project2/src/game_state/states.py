from dataclasses import dataclass
from typing import List
from src.poker_oracle.deck import Card
from src.game_manager.game_stage import GameStage


@dataclass
class PlayerState:
    def __init__(self, chips: int, folded: bool, bet: int):
        self.chips = chips  # An integer
        self.folded = folded  # A boolean
        self.bet = bet  # An integer


@dataclass
class BoardState:
    def __init__(self, cards: List[Card], pot: int, highest_bet: int, dealer: int):
        if not isinstance(cards, list):
            raise TypeError("cards must be a list")
        self.cards = cards  # A list of Card instances
        self.pot = pot  # An integer
        self.highest_bet = highest_bet  # An integer
        self.dealer = dealer  # An integer


@dataclass
class GameState:
    def __init__(
        self,
        player_states: List[PlayerState],
        board_state: BoardState,
        game_stage: GameStage,
    ):
        self.player_states = player_states  # A list of PlayerState instances
        self.board_state = board_state  # An instance of BoardState
        self.game_stage = game_stage  # An instance of GameStage
