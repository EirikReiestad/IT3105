import unittest
from src.poker_oracle.oracle import Oracle
from src.poker_oracle.deck import Suit, Card
from src.poker_oracle.hands import Hands
class TestOracle(unittest.TestCase):
    def test_generate_all_hole_pairs(self):
        pairs = Oracle.generate_all_hole_pairs()
        self.assertEqual(len(pairs), 1326)

    def test_generate_all_hole_pairs_types(self):
        pairs = Oracle.generate_all_hole_pairs_types()
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

    def test_hand_evaluator_resolve_tie_with_extra_cards(self):
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
        self.assertEqual(result, -1)


if __name__ == "__main__":
    unittest.main()
