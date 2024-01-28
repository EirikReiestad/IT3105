import unittest
from src.poker_oracle.deck import Card, Suit
from src.poker_oracle.hands import HandsCheck


class TestRoyalFlush(unittest.TestCase):
    def test_royal_flush_ok(self):
        cards = [
            Card(Suit.Spades, 10),
            Card(Suit.Hearts, 1),
            Card(Suit.Spades, 11),
            Card(Suit.Spades, 12),
            Card(Suit.Spades, 13),
            Card(Suit.Spades, 1),
            Card(Suit.Hearts, 10),
        ]
        result, cards = HandsCheck.is_royal_flush(cards)

        cards.sort(key=lambda x: x.rank)

        self.assertTrue(result)
        self.assertEqual(
            cards,
            [
                Card(Suit.Spades, 1),
                Card(Suit.Spades, 10),
                Card(Suit.Spades, 11),
                Card(Suit.Spades, 12),
                Card(Suit.Spades, 13),
            ],
        )

    def test_royal_flush_ok_2(self):
        cards = [
            Card(Suit.Hearts, 10),
            Card(Suit.Hearts, 1),
            Card(Suit.Spades, 11),
            Card(Suit.Spades, 12),
            Card(Suit.Spades, 13),
            Card(Suit.Spades, 1),
            Card(Suit.Spades, 10),
        ]
        result, cards = HandsCheck.is_royal_flush(cards)

        cards.sort(key=lambda x: x.rank)

        self.assertTrue(result)
        self.assertEqual(
            cards,
            [
                Card(Suit.Spades, 1),
                Card(Suit.Spades, 10),
                Card(Suit.Spades, 11),
                Card(Suit.Spades, 12),
                Card(Suit.Spades, 13),
            ],
        )

    def test_royal_flush_not_ok(self):
        cards = [
            Card(Suit.Hearts, 10),
            Card(Suit.Hearts, 1),
            Card(Suit.Spades, 11),
            Card(Suit.Spades, 8),
            Card(Suit.Spades, 13),
            Card(Suit.Spades, 1),
            Card(Suit.Spades, 10),
        ]
        result, cards = HandsCheck.is_royal_flush(cards)

        cards.sort(key=lambda x: x.rank)

        self.assertFalse(result)
        self.assertEqual(cards, [])


class TestStraightFlush(unittest.TestCase):
    def test_straight_flush_ok(self):
        cards = [
            Card(Suit.Spades, 10),
            Card(Suit.Hearts, 1),
            Card(Suit.Spades, 11),
            Card(Suit.Spades, 12),
            Card(Suit.Spades, 13),
            Card(Suit.Spades, 1),
            Card(Suit.Spades, 10),
        ]
        result, cards = HandsCheck.is_straight_flush(cards)

        cards.sort(key=lambda x: x.rank)

        self.assertTrue(result)
        self.assertEqual(
            cards,
            [
                Card(Suit.Spades, 1),
                Card(Suit.Spades, 10),
                Card(Suit.Spades, 11),
                Card(Suit.Spades, 12),
                Card(Suit.Spades, 13),
            ],
        )

    def test_straight_flush_ok_2(self):
        cards = [
            Card(Suit.Spades, 2),
            Card(Suit.Hearts, 1),
            Card(Suit.Spades, 3),
            Card(Suit.Spades, 4),
            Card(Suit.Spades, 5),
            Card(Suit.Spades, 6),
            Card(Suit.Hearts, 10),
        ]
        result, cards = HandsCheck.is_straight_flush(cards)

        cards.sort(key=lambda x: x.rank)

        self.assertTrue(result)
        self.assertEqual(
            cards,
            [
                Card(Suit.Spades, 2),
                Card(Suit.Spades, 3),
                Card(Suit.Spades, 4),
                Card(Suit.Spades, 5),
                Card(Suit.Spades, 6),
            ],
        )

    def test_straight_flush_not_ok(self):
        cards = [
            Card(Suit.Spades, 2),
            Card(Suit.Hearts, 1),
            Card(Suit.Spades, 3),
            Card(Suit.Hearts, 8),
            Card(Suit.Spades, 5),
            Card(Suit.Spades, 6),
            Card(Suit.Hearts, 10),
        ]
        result, cards = HandsCheck.is_straight_flush(cards)

        cards.sort(key=lambda x: x.rank)

        self.assertFalse(result)
        self.assertEqual(cards, [])

    def test_straight_flush_not_ok_2(self):
        cards = [
            Card(Suit.Spades, 2),
            Card(Suit.Hearts, 1),
            Card(Suit.Spades, 3),
            Card(Suit.Hearts, 4),
            Card(Suit.Spades, 5),
            Card(Suit.Spades, 6),
            Card(Suit.Hearts, 10),
        ]
        result, cards = HandsCheck.is_straight_flush(cards)

        cards.sort(key=lambda x: x.rank)

        self.assertFalse(result)
        self.assertEqual(cards, [])


class TestFourOfAKind(unittest.TestCase):
    def test_four_of_a_kind_ok(self):
        cards = [
            Card(Suit.Hearts, 10),
            Card(Suit.Hearts, 1),
            Card(Suit.Spades, 1),
            Card(Suit.Clubs, 1),
            Card(Suit.Diamonds, 1),
            Card(Suit.Spades, 10),
        ]
        result, cards = HandsCheck.is_four_of_a_kind(cards)

        cards.sort(key=lambda x: x.rank)

        self.assertTrue(result)
        self.assertEqual(len(cards), 4)

    def test_four_of_a_kind_not_ok(self):
        cards = [
            Card(Suit.Hearts, 10),
            Card(Suit.Hearts, 1),
            Card(Suit.Spades, 2),
            Card(Suit.Clubs, 1),
            Card(Suit.Diamonds, 1),
            Card(Suit.Spades, 10),
        ]
        result, cards = HandsCheck.is_four_of_a_kind(cards)

        self.assertFalse(result)
        self.assertEqual(cards, [])


class TestFullHouse(unittest.TestCase):
    def test_full_house_ok(self):
        cards = [
            Card(Suit.Hearts, 10),
            Card(Suit.Clubs, 10),
            Card(Suit.Spades, 1),
            Card(Suit.Clubs, 1),
            Card(Suit.Diamonds, 1),
            Card(Suit.Spades, 11),
        ]
        result, cards = HandsCheck.is_full_house(cards)

        cards.sort(key=lambda x: x.rank)

        self.assertTrue(result)
        self.assertEqual(
            cards,
            [
                Card(Suit.Spades, 1),
                Card(Suit.Clubs, 1),
                Card(Suit.Diamonds, 1),
                Card(Suit.Hearts, 10),
                Card(Suit.Clubs, 10),
            ],
        )

    def test_full_house_not_ok(self):
        cards = [
            Card(Suit.Hearts, 10),
            Card(Suit.Clubs, 9),
            Card(Suit.Spades, 1),
            Card(Suit.Clubs, 1),
            Card(Suit.Diamonds, 1),
            Card(Suit.Hearts, 11),
        ]
        result, cards = HandsCheck.is_full_house(cards)

        cards.sort(key=lambda x: x.rank)

        self.assertFalse(result)
        self.assertEqual(cards, [])


class TestFlush(unittest.TestCase):
    def test_flush_ok(self):
        cards = [
            Card(Suit.Hearts, 10),
            Card(Suit.Hearts, 7),
            Card(Suit.Hearts, 2),
            Card(Suit.Clubs, 1),
            Card(Suit.Hearts, 1),
            Card(Suit.Hearts, 11),
        ]
        result, cards = HandsCheck.is_flush(cards)

        cards.sort(key=lambda x: x.rank)

        self.assertTrue(result)
        self.assertEqual(
            cards,
            [
                Card(Suit.Hearts, 1),
                Card(Suit.Hearts, 2),
                Card(Suit.Hearts, 7),
                Card(Suit.Hearts, 10),
                Card(Suit.Hearts, 11),
            ],
        )

    def test_flush_ok2(self):
        cards = [
            Card(Suit.Hearts, 10),
            Card(Suit.Hearts, 7),
            Card(Suit.Hearts, 2),
            Card(Suit.Hearts, 11),
            Card(Suit.Hearts, 1),
            Card(Suit.Hearts, 13),
        ]
        result, cards = HandsCheck.is_flush(cards)

        cards.sort(key=lambda x: x.rank)

        self.assertTrue(result)
        self.assertEqual(
            cards,
            [
                Card(Suit.Hearts, 2),
                Card(Suit.Hearts, 7),
                Card(Suit.Hearts, 10),
                Card(Suit.Hearts, 11),
                Card(Suit.Hearts, 13),
            ],
        )

    def test_flush_not_ok(self):
        cards = [
            Card(Suit.Hearts, 10),
            Card(Suit.Hearts, 7),
            Card(Suit.Spades, 2),
            Card(Suit.Clubs, 1),
            Card(Suit.Hearts, 1),
            Card(Suit.Hearts, 11),
        ]
        result, cards = HandsCheck.is_flush(cards)

        cards.sort(key=lambda x: x.rank)

        self.assertFalse(result)
        self.assertEqual(cards, [])


class TestStraight(unittest.TestCase):
    def test_straight_ok(self):
        cards = [
            Card(Suit.Hearts, 10),
            Card(Suit.Spades, 7),
            Card(Suit.Hearts, 8),
            Card(Suit.Clubs, 9),
            Card(Suit.Hearts, 13),
            Card(Suit.Hearts, 11),
        ]
        result, cards = HandsCheck.is_straight(cards)

        cards.sort(key=lambda x: x.rank)

        self.assertTrue(result)
        self.assertEqual(
            cards,
            [
                Card(Suit.Spades, 7),
                Card(Suit.Hearts, 8),
                Card(Suit.Clubs, 9),
                Card(Suit.Hearts, 10),
                Card(Suit.Hearts, 11),
            ],
        )

    def test_straight_ok2(self):
        cards = [
            Card(Suit.Hearts, 10),
            Card(Suit.Hearts, 7),
            Card(Suit.Clubs, 8),
            Card(Suit.Hearts, 9),
            Card(Suit.Spades, 13),
            Card(Suit.Clubs, 11),
            Card(Suit.Hearts, 12),
        ]
        result, cards = HandsCheck.is_straight(cards)

        cards.sort(key=lambda x: x.rank)

        self.assertTrue(result)
        self.assertEqual(
            cards,
            [
                Card(Suit.Hearts, 9),
                Card(Suit.Hearts, 10),
                Card(Suit.Clubs, 11),
                Card(Suit.Hearts, 12),
                Card(Suit.Spades, 13),
            ],
        )

    def test_straight_not_ok(self):
        cards = [
            Card(Suit.Hearts, 10),
            Card(Suit.Hearts, 7),
            Card(Suit.Hearts, 8),
            Card(Suit.Hearts, 9),
            Card(Suit.Hearts, 13),
            Card(Suit.Hearts, 11),
        ]
        result, cards = HandsCheck.is_straight(cards)

        cards.sort(key=lambda x: x.rank)

        self.assertFalse(result)
        self.assertEqual(cards, [])


class TestThreeOfAKind(unittest.TestCase):
    def test_three_of_a_kind_ok(self):
        cards = [
            Card(Suit.Hearts, 10),
            Card(Suit.Spades, 10),
            Card(Suit.Hearts, 8),
            Card(Suit.Clubs, 10),
            Card(Suit.Hearts, 13),
            Card(Suit.Hearts, 11),
        ]
        result, cards = HandsCheck.is_three_of_a_kind(cards)

        self.assertTrue(result)
        self.assertEqual(len(cards), 3)

    def test_three_of_a_kind_not_ok(self):
        cards = [
            Card(Suit.Hearts, 10),
            Card(Suit.Spades, 10),
            Card(Suit.Clubs, 13),
            Card(Suit.Clubs, 10),
            Card(Suit.Hearts, 13),
            Card(Suit.Spades, 13),
        ]
        result, cards = HandsCheck.is_three_of_a_kind(cards)

        self.assertFalse(result)
        self.assertEqual(len(cards), 0)


class TestTwoPair(unittest.TestCase):
    def test_two_pair_ok(self):
        cards = [
            Card(Suit.Hearts, 10),
            Card(Suit.Spades, 10),
            Card(Suit.Hearts, 8),
            Card(Suit.Clubs, 8),
            Card(Suit.Hearts, 13),
            Card(Suit.Clubs, 13),
        ]
        result, cards = HandsCheck.is_two_pairs(cards)

        self.assertTrue(result)
        self.assertEqual(len(cards), 4)
        self.assertTrue(any(card.rank == 13 for card in cards))
        self.assertTrue(any(card.rank == 10 for card in cards))
        self.assertTrue(all(card.rank != 8 for card in cards))

    def test_two_pair_not_ok(self):
        cards = [
            Card(Suit.Hearts, 10),
            Card(Suit.Spades, 10),
            Card(Suit.Hearts, 7),
            Card(Suit.Clubs, 8),
            Card(Suit.Hearts, 13),
            Card(Suit.Hearts, 11),
        ]
        result, cards = HandsCheck.is_two_pairs(cards)

        self.assertFalse(result)
        self.assertEqual(len(cards), 0)


class TestOnePair(unittest.TestCase):
    def test_one_pair_ok(self):
        cards = [
            Card(Suit.Hearts, 10),
            Card(Suit.Spades, 10),
            Card(Suit.Hearts, 8),
            Card(Suit.Clubs, 8),
            Card(Suit.Hearts, 13),
            Card(Suit.Hearts, 11),
        ]
        result, cards = HandsCheck.is_one_pair(cards)

        self.assertTrue(result)
        self.assertEqual(len(cards), 2)

    def test_one_pair_not_ok(self):
        cards = [
            Card(Suit.Hearts, 10),
            Card(Suit.Spades, 7),
            Card(Suit.Hearts, 8),
            Card(Suit.Clubs, 6),
            Card(Suit.Hearts, 13),
            Card(Suit.Hearts, 11),
        ]
        result, cards = HandsCheck.is_one_pair(cards)

        self.assertFalse(result)
        self.assertEqual(len(cards), 0)


if __name__ == "__main__":
    unittest.main()
