from dataclasses import dataclass
from typing import List, Tuple
from src.poker_oracle.deck import Card
from src.game_manager.game_stage import GameStage
from src.poker_oracle.deck import Deck
from src.game_state.player_state import PublicPlayerState


@dataclass
class PublicBoardState:
    def __init__(
        self,
        cards: List[Card],
        pot: int,
        highest_bet: int,
        dealer: int = 0,
        game_stage: GameStage = GameStage.PreFlop,
    ):
        if not isinstance(cards, list):
            raise TypeError("cards must be a list")
        self.cards = cards  # A list of Card instances
        self.pot = pot  # An integer
        self.highest_bet = highest_bet  # An integer
        self.dealer: int = dealer
        self.game_stage: GameStage = game_stage  # An instance of GameStage

    def __repr__(self):
        return f"Pot: {self.pot} Highest bet: {self.highest_bet} Game stage: {self.game_stage} Cards: {self.cards}"


@dataclass
class PrivateBoardState(PublicBoardState):
    flop: Tuple[Card, Card, Card]
    turn: Card
    river: Card

    def __init__(
        self,
        pot: int,
        highest_bet: int,
    ):
        super().__init__(cards=list(), pot=pot, highest_bet=highest_bet)

    def reset_round(self, deck: Deck, dealer: int) -> Deck:
        # Deal flop, turn, and river
        self.flop = (deck.pop(), deck.pop(), deck.pop())
        self.turn = deck.pop()
        self.river = deck.pop()
        self.cards = list()
        self.pot = 0
        self.highest_bet = 0

        for card in self.flop:
            if not isinstance(card, Card):
                raise TypeError("card must be a Card")
        if not isinstance(self.turn, Card):
            raise TypeError("turn must be a Card")
        if not isinstance(self.river, Card):
            raise TypeError("river must be a Card")

        self.dealer = dealer

        return deck

    def to_public(self) -> PublicBoardState:
        self._update_card_state()
        return PublicBoardState(
            cards=self.cards,
            pot=self.pot,
            highest_bet=self.highest_bet,
            dealer=self.dealer,
            game_stage=self.game_stage,
        )

    def update_board_state(self, game_stage: GameStage):
        self.game_stage = game_stage
        self._update_card_state()

    def _update_card_state(self):
        if self.game_stage == GameStage.PreFlop:
            pass  # Doesn't matter because already an empty list
        elif self.game_stage == GameStage.Flop:
            flop = self.flop
            self.cards = [i for i in flop]
        elif self.game_stage == GameStage.Turn:
            flop = self.flop
            turn = self.turn
            self.cards = [flop[0], flop[1], flop[2], turn]
        elif self.game_stage == GameStage.River:
            flop = self.flop
            turn = self.turn
            river = self.river
            self.cards = [flop[0], flop[1], flop[2], turn, river]
