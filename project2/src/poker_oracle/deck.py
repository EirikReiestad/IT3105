import random
from enum import Enum

from src.config import Config

config = Config()


class Suit(Enum):
    Clubs = "♣"
    Diamonds = "♦"
    Hearts = "♥"
    Spades = "♠"
    None_ = ""  # For the strategy


class Card:
    def __init__(self, suit, rank):
        self.suit = suit
        self.rank = rank

    def __str__(self):
        rank = {1: "A", 11: "J", 12: "Q", 13: "K"}.get(
            self.rank, str(self.rank))
        return f"{rank}{self.suit.value}"

    def to_png_str(self) -> str:
        suit = ""
        match self.suit:
            case Suit.Clubs:
                suit = "clubs"
            case Suit.Diamonds:
                suit = "diamonds"
            case Suit.Hearts:
                suit = "hearts"
            case Suit.Spades:
                suit = "spades"
        if self.rank == 1:
            return f"ace_of_{suit}.png"
        if self.rank == 11:
            return f"jack_of_{suit}.png"
        if self.rank == 12:
            return f"queen_of_{suit}.png"
        if self.rank == 13:
            return f"king_of_{suit}.png"
        return f"{self.rank}_of_{suit}.png"

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


class Deck:
    def __init__(self, shuffle=True):
        self.stack = []
        self._reset_stack(shuffle)

    def _reset_stack(self, shuffle):
        self.stack = []
        for suit_idx in range(4):
            suit = None
            if suit_idx == 0:
                suit = Suit.Clubs
            elif suit_idx == 1:
                suit = Suit.Diamonds
            elif suit_idx == 2:
                suit = Suit.Hearts
            elif suit_idx == 3:
                suit = Suit.Spades

            if not config.data['simplify']:
                for rank in range(1, 14):
                    self.stack.append(Card(suit, rank))
            else:
                for rank in range(9, 14):
                    self.stack.append(Card(suit, rank))
                self.stack.append(Card(suit, 1))

        if shuffle:
            random.shuffle(self.stack)

    def remove(self, card):
        self.stack = [c for c in self.stack if c != card]

    def pop(self) -> Card:
        if self.stack:
            return self.stack.pop()
        else:
            return None

    def __len__(self):
        return len(self.stack)

    def __str__(self):
        return str(self.stack)
