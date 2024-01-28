from enum import Enum

class Suit(Enum):
    Clubs = "♣"
    Diamonds = "♦"
    Hearts = "♥"
    Spades = "♠"

class Card:
    def __init__(self, suit, rank):
        self.suit = suit
        self.rank = rank

    def __str__(self):
        rank = {
            1: "A",
            11: "J",
            12: "Q",
            13: "K"
        }.get(self.rank, str(self.rank))
        return f"{rank}{self.suit.value}"

    def __eq__(self, other) -> bool:
        return self.rank == other.rank and self.suit == other.suit

    def __lt__(self, other) -> bool:
        return self.rank < other.rank

    def __le__(self, other) -> bool:
        return self.rank <= other.rank

    def __gt__(self, other) -> bool:
        return self.rank > other.rank

    def __ge__(self, other) -> bool:
        return self.rank >= other.rank
