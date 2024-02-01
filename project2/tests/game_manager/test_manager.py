import unittest
from src.game_manager.manager import GameManager
from src.poker_oracle.deck import Deck


class TestGameManager(unittest.TestCase):
    def test_new(self):
        num_players = 2
        deck = Deck()
        deck.reset_stack()
        game_manager = GameManager(num_players, deck)
        self.assertEqual(len(game_manager.players), num_players)


if __name__ == "__main__":
    unittest.main()
