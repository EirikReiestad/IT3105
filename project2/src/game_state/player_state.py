from dataclasses import dataclass
from typing import Tuple
from src.poker_oracle.deck import Card
from src.game_manager.game_action import Action


@dataclass
class PublicPlayerState:
    def __init__(
        self, chips: int, folded: bool = False, bust: bool = False, bet: int = 0
    ):
        self.chips: int = chips
        self.folded: bool = folded
        self.bust: bool = bust
        self.round_bet: int = bet  # Betting for that round # TODO: Maybe change name to bet
        self.betting_history = list()
        self.action_history = list()

    def __repr__(self):
        return f"Chips: {self.chips} Folded: {self.folded} Round bet: {self.round_bet}"


class PrivatePlayerState(PublicPlayerState):
    def __init__(self):
        super().__init__(chips=100)
        self.cards: Tuple[Card, Card] = (None, None)

    def _bet(self, amount) -> bool:
        if self.chips < amount:
            return False
        self.round_bet += amount
        self.chips -= amount
        return True

    def action(self, action: Action, amount: int = 0) -> bool:
        """Returns a boolean indicating whether the action was successful"""
        if self._bet(amount) is False:
            # TODO: Do not need to fold. Can go all-in or check if possible
            print("Not enough money. Folded.")
            self.action_history.append(Action.Fold)
            self.betting_history.append(0)
            return False
        self.action_history.append(action)
        self.betting_history.append(0)
        if action == Action.Fold:
            self.folded = True
        return True

    def to_public(self) -> PublicPlayerState:
        return PublicPlayerState(
            chips=self.chips,
            folded=self.folded,
            bet=self.bet,
        )

    def hand(self) -> Tuple[Card, Card]:
        return self.cards

    def reset_round(self, cards: Tuple[Card, Card]):
        # Enforce type
        if not isinstance(cards, tuple):
            raise TypeError("cards must be a tuple")
        if len(cards) != 2:
            raise ValueError("cards must be a tuple of length 2")
        if self.bust is False:
            self.cards = cards
        self.folded = False
        self.bet = 0
        self.betting_history = list()
        self.action_history = list()

    def __str__(self):
        s = ""
        if self.folded:
            s = "Folded "
        s += f"Cards: {self.cards[0]}, {self.cards[1]} Chips: {self.chips}"
        return s
