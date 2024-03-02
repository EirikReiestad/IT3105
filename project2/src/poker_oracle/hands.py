import os
import sys

module_path = os.path.abspath(os.path.join("./"))
if module_path not in sys.path:
    sys.path.append(module_path)

from itertools import combinations
from typing import List, Tuple
from enum import Enum
from collections import defaultdict, Counter
from dataclasses import dataclass
from src.poker_oracle.deck import Card, Suit

import itertools
from typing import List, Tuple


class Hands(Enum):
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

    def __lt__(self, other):
        return self.value < other.value

    def __le__(self, other):
        return self.value <= other.value

    def __eq__(self, other):
        if isinstance(other, Hands):
            return self.value == other.value
        return NotImplemented

    def __ne__(self, other):
        if isinstance(other, Hands):
            return self.value != other.value
        return NotImplemented

    def __gt__(self, other):
        return self.value > other.value

    def __ge__(self, other):
        return self.value >= other.value


class HandsCheck:
    @staticmethod
    def is_royal_flush(cards):
        target_ranks = set([1, 10, 11, 12, 13])

        for combination in combinations(cards, 5):
            unique_suits = defaultdict(list)
            for card in combination:
                unique_suits[card.suit].append(card.rank)
            for suit, ranks in unique_suits.items():
                unique_ranks = set(ranks)
                if unique_ranks == target_ranks:
                    new_cards = [Card(suit, rank) for rank in unique_ranks]
                    return (True, new_cards)
        return (False, [])

    @staticmethod
    def is_straight_flush(cards):
        result, new_cards = HandsCheck.is_royal_flush(cards)

        sorted_cards = sorted(cards, key=lambda x: x.rank, reverse=True)
        combinations = list(itertools.combinations(sorted_cards, 5))

        if result:
            return (result, new_cards)

        for combination in combinations:
            unique_suits = set()
            unique_ranks = set()
            first_suit = combination[0].suit
            for card in combination:
                unique_suits.add(card.suit)
                unique_ranks.add(card.rank)

            ranks_vec = sorted(list(unique_ranks))

            is_one_apart = all(
                (ranks_vec[i + 1] - ranks_vec[i]) == 1
                for i in range(len(ranks_vec) - 1)
            )

            if len(unique_suits) == 1 and len(ranks_vec) == 5 and is_one_apart:
                new_cards = [Card(first_suit, rank) for rank in ranks_vec]
                return (True, new_cards)
        return (False, [])

    @staticmethod
    def is_four_of_a_kind(cards):
        combinations = list(itertools.combinations(cards, 5))

        for combination in combinations:
            unique_ranks = {}
            for card in combination:
                if card.rank not in unique_ranks:
                    unique_ranks[card.rank] = []
                unique_ranks[card.rank].append(card.suit)
            for rank, suits in unique_ranks.items():
                unique_suits = set(suits)
                if len(unique_suits) == 4:
                    new_cards = [Card(suit, rank) for suit in unique_suits]
                    return (True, new_cards)
        return (False, [])

    @staticmethod
    def is_full_house(cards):
        sorted_cards = sorted(cards, key=lambda x: x.rank, reverse=True)
        combinations = list(itertools.combinations(sorted_cards, 5))

        for combination in combinations:
            rank_counts = {}
            for card in combination:
                if card.rank not in rank_counts:
                    rank_counts[card.rank] = []
                rank_counts[card.rank].append(card.suit)
            cards_len_2 = []
            cards_len_3 = []
            for rank, suits in rank_counts.items():
                if len(suits) == 2:
                    cards_len_2.extend([Card(suit, rank) for suit in suits])
                elif len(suits) == 3:
                    cards_len_3.extend([Card(suit, rank) for suit in suits])
            if cards_len_2 and cards_len_3:
                return (True, cards_len_2 + cards_len_3)
        return (False, [])

    @staticmethod
    def is_flush(cards):
        sorted_cards = sorted(cards, key=lambda x: x.rank, reverse=True)
        combinations = list(itertools.combinations(sorted_cards, 5))

        for combination in combinations:
            unique_suits = set()
            unique_ranks = set()
            first_suit = combination[0].suit
            for card in combination:
                unique_suits.add(card.suit)
                unique_ranks.add(card.rank)

            ranks_vec = sorted(list(unique_ranks))

            is_one_apart = all(
                (ranks_vec[i + 1] - ranks_vec[i]) == 1
                for i in range(len(ranks_vec) - 1)
            )

            if len(unique_suits) == 1 and len(ranks_vec) == 5 and not is_one_apart:
                new_cards = [Card(first_suit, rank) for rank in ranks_vec]
                return (True, new_cards)
        return (False, [])

    @staticmethod
    def is_straight(cards):
        sorted_cards = sorted(cards, key=lambda x: x.rank, reverse=True)
        combinations = list(itertools.combinations(sorted_cards, 5))
        target_ranks = {1, 10, 11, 12, 13}

        for combination in combinations:
            unique_suits = set()
            unique_ranks = set()

            for card in combination:
                unique_suits.add(card.suit)
                unique_ranks.add(card.rank)
            royal = unique_ranks == target_ranks
            ranks_vec = sorted(list(unique_ranks))

            is_one_apart = all(
                (ranks_vec[i + 1] - ranks_vec[i]) == 1
                for i in range(len(ranks_vec) - 1)
            )

            if (
                len(unique_suits) != 1
                and len(ranks_vec) == 5
                and (is_one_apart or royal)
            ):
                new_cards = [card for card in combination]
                return (True, new_cards)
        return (False, [])

    @staticmethod
    def is_three_of_a_kind(cards):
        sorted_cards = sorted(cards, key=lambda x: x.rank, reverse=True)
        combinations = list(itertools.combinations(sorted_cards, 5))

        for combination in combinations:
            unique_ranks = {}
            for card in combination:
                if card.rank not in unique_ranks:
                    unique_ranks[card.rank] = []
                unique_ranks[card.rank].append(card.suit)
            length_unique_ranks = len(unique_ranks)
            for rank, suits in unique_ranks.items():
                unique_suits = set(suits)
                if len(unique_suits) == 3 and length_unique_ranks > 2:
                    new_cards = [Card(suit, rank) for suit in unique_suits]
                    return (True, new_cards)
        return (False, [])

    @staticmethod
    def is_two_pairs(cards):
        sorted_cards = sorted(cards, key=lambda x: x.rank, reverse=True)
        combinations = list(itertools.combinations(sorted_cards, 5))

        for combination in combinations:
            unique_ranks = {}
            for card in combination:
                if card.rank not in unique_ranks:
                    unique_ranks[card.rank] = []
                unique_ranks[card.rank].append(card.suit)
            if len(unique_ranks) == 3:
                filtered_map = {
                    rank: suits
                    for rank, suits in unique_ranks.items()
                    if len(suits) == 2
                }
                if len(filtered_map) != 2:
                    continue
                new_cards = [
                    Card(suit, rank)
                    for rank, suits in filtered_map.items()
                    for suit in suits
                ]
                return (True, new_cards)
        return (False, [])

    @staticmethod
    def is_one_pair(cards):
        sorted_cards = sorted(cards, key=lambda x: x.rank, reverse=True)
        combinations = list(itertools.combinations(sorted_cards, 5))

        for combination in combinations:
            unique_ranks = {}
            for card in combination:
                if card.rank not in unique_ranks:
                    unique_ranks[card.rank] = []
                unique_ranks[card.rank].append(card.suit)
            if len(unique_ranks) == 4:
                filtered_map = {
                    rank: suits
                    for rank, suits in unique_ranks.items()
                    if len(suits) == 2
                }
                if filtered_map:
                    first_key, first_value = next(iter(filtered_map.items()))
                    new_cards = [
                        Card(suit, rank) for suit in first_value for rank in [first_key]
                    ]
                    return (True, new_cards)
        return (False, [])


if __name__ == "__main__":
    cards_one = [
            Card(Suit.Spades, 13),
            Card(Suit.Spades, 12),
            Card(Suit.Spades, 1),
            Card(Suit.Spades, 2),
            Card(Suit.Spades, 11),
            Card(Suit.Spades, 10),
        ]


    ues, cards_back = HandsCheck.is_two_pairs(cards_one)

    for card in cards_back:
        print(card)

    