from src.card import Card
from typing import List, Tuple, Optional


class Player:
    def __init__(self, cards: Tuple[Card, Card]):
        # Enforce type
        if not isinstance(cards, tuple):
            raise TypeError("cards must be a tuple")
        if len(cards) != 2:
            raise ValueError("cards must be a tuple of length 2")

        self.cards = cards  # A tuple of two Card instances
        self.chips = 100  # An integer
        self.folded = False  # A boolean
        self.bet = 0  # An integer

    def bet(self, amount) -> bool:
        if self.chips < amount:
            return False
        self.bet += amount
        self.chips -= amount
        return True

    def __str__(self):
        return f"Cards: {self.cards[0]}, {self.cards[1]} Chips: {self.chips}"


class Players:
    def __init__(self, players: List[Player]):
        self.players = players  # A list of Player instances

    def __len__(self):
        return len(self.players)

    def __iter__(self):
        return iter(self.players)

    def get_number_of_folded(self) -> int:
        # Note: This is more expensive than a simple counter. However, the number of players is
        # small so this is not an issue, and it is more readable.
        # TODO: Consider using a counter instead.
        return sum(player.folded for player in self.players)

    def get_number_of_active_players(self) -> int:
        return len(self.players) - self.get_number_of_folded()

    def has_folded(self, player: int) -> bool:
        return self.players[player].folded

    def fold(self, player: int):
        self.players[player].folded = True

    def get_bet(self, player: int) -> int:
        return self.players[player].bet

    def place_bet(self, player, bet):
        self.players[player].bet += bet
        self.players[player].bet(bet)
