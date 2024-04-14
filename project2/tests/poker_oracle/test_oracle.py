import unittest
from src.poker_oracle.oracle import Oracle
from src.poker_oracle.deck import Suit, Card
from src.poker_oracle.hands import Hands
from src.config import Config

config = Config()


class TestOracle(unittest.TestCase):
    def test_generate_all_hole_pairs(self):
        pairs = Oracle.generate_all_hole_pairs()
        if config.data['simplify']:
            self.assertEqual(len(pairs), 276)
        else:
            self.assertEqual(len(pairs), 1326)

    def test_generate_all_hole_pairs_types(self):
        pairs, idx = Oracle.generate_all_hole_pairs_types()
        self.assertEqual(len(pairs), 169)

    def test_hand_classifier_royal_flush(self):
        cards = [
            Card(Suit.Clubs, 13),
            Card(Suit.Clubs, 12),
            Card(Suit.Clubs, 1),
            Card(Suit.Clubs, 2),
            Card(Suit.Clubs, 11),
            Card(Suit.Clubs, 10),
        ]

        result, _ = Oracle.hand_classifier(cards)
        self.assertEqual(result, Hands.RoyalFlush)

    def test_hand_classifier_one_pair(self):
        cards = [
            Card(Suit.Clubs, 13),
            Card(Suit.Spades, 12),
            Card(Suit.Hearts, 1),
            Card(Suit.Clubs, 2),
            Card(Suit.Clubs, 12),
            Card(Suit.Clubs, 10),
        ]

        result, _ = Oracle.hand_classifier(cards)
        self.assertEqual(result, Hands.OnePair)

    def test_hand_evaluator_player_two_wins_higher_draw(self):
        cards_one = [
            Card(Suit.Spades, 10),
            Card(Suit.Diamonds, 9),
            Card(Suit.Spades, 1),
            Card(Suit.Clubs, 9),
            Card(Suit.Diamonds, 12),
            Card(Suit.Clubs, 13),
            Card(Suit.Clubs, 1),
        ]

        cards_two = [
            Card(Suit.Clubs, 12),
            Card(Suit.Diamonds, 13),
            Card(Suit.Spades, 1),
            Card(Suit.Clubs, 9),
            Card(Suit.Diamonds, 12),
            Card(Suit.Clubs, 13),
            Card(Suit.Clubs, 1),
        ]

        result = Oracle.hand_evaluator(cards_one, cards_two)
        self.assertEqual(result, 1)

    def test_hand_evaluator_player_with_A(self):
        cards_one = [
            Card(Suit.Spades, 13),
            Card(Suit.Diamonds, 12),
            Card(Suit.Spades, 11),
            Card(Suit.Clubs, 10),
            Card(Suit.Diamonds, 9),
        ]

        cards_two = [
            Card(Suit.Spades, 13),
            Card(Suit.Diamonds, 12),
            Card(Suit.Spades, 11),
            Card(Suit.Clubs, 1),
            Card(Suit.Diamonds, 10),
        ]

        result = Oracle.hand_evaluator(cards_one, cards_two)
        self.assertEqual(result, -1)

    def test_hand_evaluator_player_with_A_pair(self):
        cards_one = [
            Card(Suit.Spades, 13),
            Card(Suit.Diamonds, 13),
            Card(Suit.Spades, 1),
            Card(Suit.Clubs, 1),
            Card(Suit.Diamonds, 10),
        ]

        cards_two = [
            Card(Suit.Spades, 13),
            Card(Suit.Diamonds, 13),
            Card(Suit.Spades, 1),
            Card(Suit.Clubs, 10),
            Card(Suit.Diamonds, 10),
        ]

        result = Oracle.hand_evaluator(cards_one, cards_two)
        self.assertEqual(result, 1)

    def test_hand_evaluator_player_one_lose(self):
        cards_one = [
            Card(Suit.Clubs, 13),
            Card(Suit.Spades, 12),
            Card(Suit.Hearts, 1),
            Card(Suit.Clubs, 2),
            Card(Suit.Clubs, 12),
            Card(Suit.Clubs, 10),
        ]

        cards_two = [
            Card(Suit.Clubs, 13),
            Card(Suit.Clubs, 12),
            Card(Suit.Clubs, 1),
            Card(Suit.Clubs, 2),
            Card(Suit.Clubs, 11),
            Card(Suit.Clubs, 10),
        ]

        result = Oracle.hand_evaluator(cards_one, cards_two)
        self.assertEqual(result, -1)

    def test_hand_evaluator_player_one_win(self):
        cards_one = [
            Card(Suit.Clubs, 13),
            Card(Suit.Spades, 12),
            Card(Suit.Hearts, 1),
            Card(Suit.Clubs, 2),
            Card(Suit.Clubs, 12),
            Card(Suit.Clubs, 10),
        ]

        cards_two = [
            Card(Suit.Clubs, 13),
            Card(Suit.Clubs, 12),
            Card(Suit.Clubs, 1),
            Card(Suit.Clubs, 2),
            Card(Suit.Clubs, 11),
            Card(Suit.Clubs, 10),
        ]

        result = Oracle.hand_evaluator(cards_two, cards_one)
        self.assertEqual(result, 1)

    def test_hand_evaluator_tie(self):
        cards_one = [
            Card(Suit.Spades, 13),
            Card(Suit.Spades, 12),
            Card(Suit.Spades, 1),
            Card(Suit.Spades, 2),
            Card(Suit.Spades, 11),
            Card(Suit.Spades, 10),
        ]

        cards_two = [
            Card(Suit.Clubs, 13),
            Card(Suit.Clubs, 12),
            Card(Suit.Clubs, 1),
            Card(Suit.Clubs, 2),
            Card(Suit.Clubs, 11),
            Card(Suit.Clubs, 10),
        ]

        result = Oracle.hand_evaluator(cards_two, cards_one)
        self.assertEqual(result, 0)

    def test_hand_evaluator_two_pair_first_pair_tie(self):
        cards_one = [
            Card(Suit.Spades, 13),
            Card(Suit.Diamonds, 11),
            Card(Suit.Spades, 10),
            Card(Suit.Clubs, 13),
            Card(Suit.Hearts, 10),
        ]

        cards_two = [
            Card(Suit.Spades, 13),
            Card(Suit.Diamonds, 11),
            Card(Suit.Spades, 10),
            Card(Suit.Clubs, 13),
            Card(Suit.Hearts, 11),
        ]

        result = Oracle.hand_evaluator(cards_two, cards_one)
        self.assertEqual(result, 1)

    def test_hand_evaluator_two_pairs_first_pair_tie(self):
        cards_one = [
            Card(Suit.Spades, 13),
            Card(Suit.Diamonds, 11),
            Card(Suit.Spades, 10),
            Card(Suit.Clubs, 13),
            Card(Suit.Hearts, 10),
        ]

        cards_two = [
            Card(Suit.Spades, 13),
            Card(Suit.Diamonds, 11),
            Card(Suit.Spades, 10),
            Card(Suit.Clubs, 13),
            Card(Suit.Hearts, 11),
        ]

        result = Oracle.hand_evaluator(cards_two, cards_one)
        self.assertEqual(result, 1)

    def test_hand_evaluator_resolve_tie_straight(self):
        cards_one = [
            Card(Suit.Spades, 13),
            Card(Suit.Spades, 12),
            Card(Suit.Spades, 1),
            Card(Suit.Spades, 2),
            Card(Suit.Spades, 11),
            Card(Suit.Spades, 10),
        ]

        cards_two = [
            Card(Suit.Clubs, 13),
            Card(Suit.Clubs, 12),
            Card(Suit.Clubs, 1),
            Card(Suit.Clubs, 5),
            Card(Suit.Clubs, 11),
            Card(Suit.Clubs, 10),
        ]

        result = Oracle.hand_evaluator(cards_one, cards_two)
        self.assertEqual(result, 0)

    def test_hand_evaluator_resolve_tie_with_extra_card(self):
        cards_one = [
            Card(Suit.Spades, 10),
            Card(Suit.Hearts, 2),
            Card(Suit.Hearts, 3),
            Card(Suit.Spades, 10),
            Card(Suit.Clubs, 5),
            Card(Suit.Spades, 7),
        ]

        cards_two = [
            Card(Suit.Clubs, 10),
            Card(Suit.Hearts, 2),
            Card(Suit.Hearts, 3),
            Card(Suit.Spades, 10),
            Card(Suit.Clubs, 6),
            Card(Suit.Spades, 8),
        ]

        result = Oracle.hand_evaluator(cards_one, cards_two)
        self.assertEqual(result, -1)

    def test_hand_evaluator_win_two_pairs_tiebreak(self):
        cards_one = [
            Card(Suit.Spades, 10),
            Card(Suit.Clubs, 10),
            Card(Suit.Spades, 11),
            Card(Suit.Clubs, 11),
            Card(Suit.Hearts, 12),
            Card(Suit.Spades, 2),
        ]

        cards_two = [
            Card(Suit.Spades, 10),
            Card(Suit.Clubs, 10),
            Card(Suit.Clubs, 11),
            Card(Suit.Spades, 11),
            Card(Suit.Clubs, 9),
            Card(Suit.Hearts, 2),
        ]

        result = Oracle.hand_evaluator(cards_one, cards_two)
        self.assertEqual(result, 1)

    def test_hand_evaluator_win_one_pair_tiebreak(self):
        cards_one = [
            Card(Suit.Spades, 10),
            Card(Suit.Hearts, 10),
            Card(Suit.Spades, 9),
            Card(Suit.Hearts, 8),
            Card(Suit.Clubs, 12),
            Card(Suit.Spades, 2),
        ]

        cards_two = [
            Card(Suit.Clubs, 10),
            Card(Suit.Hearts, 10),
            Card(Suit.Clubs, 9),
            Card(Suit.Hearts, 11),
            Card(Suit.Spades, 8),
            Card(Suit.Clubs, 2),
        ]

        result = Oracle.hand_evaluator(cards_one, cards_two)
        self.assertEqual(result, 1)

    def test_hand_evaluator_win_ace_tiebreak(self):
        cards_one = [
            Card(Suit.Hearts, 10),
            Card(Suit.Hearts, 10),
            Card(Suit.Hearts, 13),
            Card(Suit.Spades, 8),
            Card(Suit.Spades, 12),
            Card(Suit.Spades, 2),
        ]

        cards_two = [
            Card(Suit.Hearts, 10),
            Card(Suit.Hearts, 10),
            Card(Suit.Hearts, 9),
            Card(Suit.Clubs, 1),
            Card(Suit.Clubs, 8),
            Card(Suit.Clubs, 2),
        ]

        result = Oracle.hand_evaluator(cards_one, cards_two)
        self.assertEqual(result, -1)

    def test_hand_evaluator_highest_house(self):
        cards_one = [
            Card(Suit.Spades, 10),
            Card(Suit.Spades, 10),
            Card(Suit.Spades, 13),
            Card(Suit.Spades, 13),
            Card(Suit.Spades, 13),
            Card(Suit.Spades, 1),
        ]

        cards_two = [
            Card(Suit.Clubs, 11),
            Card(Suit.Clubs, 11),
            Card(Suit.Clubs, 13),
            Card(Suit.Clubs, 13),
            Card(Suit.Clubs, 13),
            Card(Suit.Clubs, 2),
        ]

        result = Oracle.hand_evaluator(cards_one, cards_two)
        self.assertEqual(result, -1)

    def test_hand_evaluator_low_straight(self):
        cards_one = [
            Card(Suit.Spades, 1),
            Card(Suit.Spades, 2),
            Card(Suit.Spades, 3),
            Card(Suit.Spades, 4),
            Card(Suit.Spades, 5),
            Card(Suit.Spades, 1),
        ]

        cards_two = [
            Card(Suit.Clubs, 10),
            Card(Suit.Clubs, 11),
            Card(Suit.Clubs, 12),
            Card(Suit.Clubs, 13),
            Card(Suit.Clubs, 1),
            Card(Suit.Clubs, 2),
        ]

        result = Oracle.hand_evaluator(cards_one, cards_two)
        self.assertEqual(result, -1)


if __name__ == "__main__":
    unittest.main()
