import unittest
from src.game_manager.manager import GameManager


class TestGameManager(unittest.TestCase):
    def test_new(self):
        num_players = 2
        game_manager = GameManager(num_players)
        self.assertEqual(len(game_manager.players), num_players)


if __name__ == "__main__":
    unittest.main()
