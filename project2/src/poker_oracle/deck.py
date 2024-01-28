import random


class Suit:
    Clubs = 0
    Diamonds = 1
    Hearts = 2
    Spades = 3


class Card:
    def __init__(self, suit, rank):
        self.suit = suit
        self.rank = rank


class Deck:
    def __init__(self):
        self.stack = []

    def reset_stack(self):
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

            for rank in range(1, 14):
                self.stack.append(Card(suit, rank))

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


# Unit Test
def test_generate_stack():
    deck = Deck()
    deck.reset_stack()
    assert len(deck.stack) == 52


# Run the test
if __name__ == "__main__":
    test_generate_stack()
    print("All tests passed.")
