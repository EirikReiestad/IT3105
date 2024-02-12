import unittest
from src.poker_oracle.deck import Deck
from src.config import Config

config = Config()


# Unit Test
class TestDeck(unittest.TestCase):
    def test_generate_stack(self):
        deck = Deck()
        if config.data['simplify']:
            assert len(deck.stack) == 6 * 4
        else:
            assert len(deck.stack) == 52


# Run the test
if __name__ == "__main__":
    unittest.main()
