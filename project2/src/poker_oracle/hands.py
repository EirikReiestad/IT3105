from itertools import combinations
from typing import List, Tuple
from enum import Enum
from collections import defaultdict, Counter
from dataclasses import dataclass
from card import Card, Suit

import itertools
from typing import List, Tuple

class Hands:
    RoyalFlush = 1
    StraightFlush = 2
    FourOfAKind = 3
    FullHouse = 4
    Flush = 5
    Straight = 6
    ThreeOfAKind = 7
    TwoPairs = 8
    OnePair = 9
    HighCard = 10

class HandsCheck:
    @staticmethod
    def is_royal_flush(cards: List[Tuple[int, int]]) -> Tuple[bool, List[Tuple[int, int]]]:
        combinations = list(itertools.combinations(cards, 5))
        target_ranks = {1, 10, 11, 12, 13}

        for combination in combinations:
            unique_suits = {}
            for card in combination:
                if card[1] not in unique_suits:
                    unique_suits[card[1]] = []
                unique_suits[card[1]].append(card[0])
            for suit, ranks in unique_suits.items():
                unique_ranks = set(ranks)
                if unique_ranks == target_ranks:
                    new_cards = [(rank, suit) for rank in unique_ranks]
                    return (True, new_cards)
        return (False, [])
    
    @staticmethod
    def is_straight_flush(cards):
        result, new_cards = HandsCheck.is_royal_flush(cards)
        sorted_cards = sorted(cards, key=lambda x: x[0], reverse=True)
        combinations = list(itertools.combinations(sorted_cards, 5))

        if result:
            return (result, new_cards)

        for combination in combinations:
            unique_suits = set()
            unique_ranks = set()
            first_suit = combination[0][1]
            for card in combination:
                unique_suits.add(card[1])
                unique_ranks.add(card[0])

            ranks_vec = sorted(list(unique_ranks))

            is_one_apart = all((ranks_vec[i+1] - ranks_vec[i]) == 1 for i in range(len(ranks_vec)-1))

            if len(unique_suits) == 1 and len(ranks_vec) == 5 and is_one_apart:
                new_cards = [(rank, first_suit) for rank in ranks_vec]
                return (True, new_cards)
        return (False, [])

    @staticmethod
    def is_four_of_a_kind(cards):
        combinations = list(itertools.combinations(cards, 5))

        for combination in combinations:
            unique_ranks = {}
            for card in combination:
                if card[0] not in unique_ranks:
                    unique_ranks[card[0]] = []
                unique_ranks[card[0]].append(card[1])
            for rank, suits in unique_ranks.items():
                unique_suits = set(suits)
                if len(unique_suits) == 4:
                    new_cards = [(rank, suit) for suit in unique_suits]
                    return (True, new_cards)
        return (False, [])
    
    @staticmethod
    def is_full_house(cards):
        sorted_cards = sorted(cards, key=lambda x: x[0], reverse=True)
        combinations = list(itertools.combinations(sorted_cards, 5))

        for combination in combinations:
            rank_counts = {}
            for card in combination:
                if card[0] not in rank_counts:
                    rank_counts[card[0]] = []
                rank_counts[card[0]].append(card[1])
            cards_len_2 = []
            cards_len_3 = []
            for rank, suits in rank_counts.items():
                if len(suits) == 2:
                    cards_len_2.extend([(rank, suit) for suit in suits])
                elif len(suits) == 3:
                    cards_len_3.extend([(rank, suit) for suit in suits])
            if cards_len_2 and cards_len_3:
                return (True, cards_len_2 + cards_len_3)
        return (False, [])
    
    @staticmethod
    def is_flush(cards):
        sorted_cards = sorted(cards, key=lambda x: x[0], reverse=True)
        combinations = list(itertools.combinations(sorted_cards, 5))

        for combination in combinations:
            unique_suits = set()
            unique_ranks = set()
            first_suit = combination[0][1]
            for card in combination:
                unique_suits.add(card[1])
                unique_ranks.add(card[0])

            ranks_vec = sorted(list(unique_ranks))

            is_one_apart = all((ranks_vec[i+1] - ranks_vec[i]) == 1 for i in range(len(ranks_vec)-1))

            if len(unique_suits) == 1 and len(ranks_vec) == 5 and not is_one_apart:
                new_cards = [(rank, first_suit) for rank in ranks_vec]
                return (True, new_cards)
        return (False, [])
    
    @staticmethod
    def is_straight(cards):
        sorted_cards = sorted(cards, key=lambda x: x[0], reverse=True)
        combinations = list(itertools.combinations(sorted_cards, 5))
        target_ranks = {1, 10, 11, 12, 13}

        for combination in combinations:
            unique_suits = set()
            unique_ranks = set()

            for card in combination:
                unique_suits.add(card[1])
                unique_ranks.add(card[0])
            royal = unique_ranks == target_ranks
            ranks_vec = sorted(list(unique_ranks))

            is_one_apart = all((ranks_vec[i+1] - ranks_vec[i]) == 1 for i in range(len(ranks_vec)-1))

            if len(unique_suits) != 1 and len(ranks_vec) == 5 and (is_one_apart or royal):
                new_cards = [card for card in combination]
                return (True, new_cards)
        return (False, [])
    
    @staticmethod
    def is_three_of_a_kind(cards):
        sorted_cards = sorted(cards, key=lambda x: x[0], reverse=True)
        combinations = list(itertools.combinations(sorted_cards, 5))

        for combination in combinations:
            unique_ranks = {}
            for card in combination:
                if card[0] not in unique_ranks:
                    unique_ranks[card[0]] = []
                unique_ranks[card[0]].append(card[1])
            length_unique_ranks = len(unique_ranks)
            for rank, suits in unique_ranks.items():
                unique_suits = set(suits)
                if len(unique_suits) == 3 and length_unique_ranks > 2:
                    new_cards = [(rank, suit) for suit in unique_suits]
                    return (True, new_cards)
        return (False, [])

    @staticmethod
    def is_two_pairs(cards):
        sorted_cards = sorted(cards, key=lambda x: x[0], reverse=True)
        combinations = list(itertools.combinations(sorted_cards, 5))

        for combination in combinations:
            unique_ranks = {}
            for card in combination:
                if card[0] not in unique_ranks:
                    unique_ranks[card[0]] = []
                unique_ranks[card[0]].append(card[1])
            if len(unique_ranks) == 3:
                filtered_map = {rank: suits for rank, suits in unique_ranks.items() if len(suits) == 2}
                if len(filtered_map) != 2:
                    continue
                new_cards = [(rank, suit) for rank, suits in filtered_map.items() for suit in suits]
                return (True, new_cards)
        return (False, [])

    @staticmethod
    def is_one_pair(cards):
        sorted_cards = sorted(cards, key=lambda x: x[0], reverse=True)
        combinations = list(itertools.combinations(sorted_cards, 5))

        for combination in combinations:
            unique_ranks = {}
            for card in combination:
                if card[0] not in unique_ranks:
                    unique_ranks[card[0]] = []
                unique_ranks[card[0]].append(card[1])
            if len(unique_ranks) == 4:
                filtered_map = {rank: suits for rank, suits in unique_ranks.items() if len(suits) == 2}
                if filtered_map:
                    first_key, first_value = next(iter(filtered_map.items()))
                    new_cards = [(rank, suit) for suit in first_value for rank in [first_key]]
                    return (True, new_cards)
        return (False, [])




import unittest

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
            ]
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
            ]
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
            ]
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
            ]
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
            ]
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
            ]
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
            ]
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
            ]
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
            ]
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


if __name__ == '__main__':
    unittest.main()
