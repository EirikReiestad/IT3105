import unittest
from src.poker_oracle.deck import Deck


# Unit Test
class TestDeck(unittest.TestCase):
    def test_generate_stack(self):
        deck = Deck()
        assert len(deck.stack) == 52


# Run the test
if __name__ == "__main__":
    unittest.main()
