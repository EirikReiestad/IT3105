import numpy as np
from .deck import Card, Suit
from src.poker_oracle.hands import HandsCheck, Hands
from src.poker_oracle.deck import Deck
from itertools import combinations
from typing import List, Tuple
import os
import sys

from enum import Enum

module_path = os.path.abspath(os.path.join("./"))
if module_path not in sys.path:
    sys.path.append(module_path)

class HoleType(Enum):
    SameRank = 0
    Unsuited = 1
    Suited = 2

class Oracle:
    def __init__(self, num_players=None):
        self.hole_pair_types, self.hole_pair_types_idx  = Oracle.generate_all_hole_pairs_types()
        self.hole_pairs = Oracle.generate_all_hole_pairs()

        self._cache = {
            "utility_matrix": None,
            "public_cards": None
        }
        if num_players is not None:
            self.cheat_sheet = self.cheat_sheet_generator(num_players-1, 100)

    @staticmethod
    def hand_classifier(cards: List[Card]) -> Tuple[Hands, List[Card]]:
        if not (5 <= len(cards) <= 7):
            raise ValueError("Invalid number of cards: {}".format(len(cards)))

        hand_checks = [
            (HandsCheck.is_royal_flush, Hands.RoyalFlush),
            (HandsCheck.is_straight_flush, Hands.StraightFlush),
            (HandsCheck.is_four_of_a_kind, Hands.FourOfAKind),
            (HandsCheck.is_full_house, Hands.FullHouse),
            (HandsCheck.is_flush, Hands.Flush),
            (HandsCheck.is_straight, Hands.Straight),
            (HandsCheck.is_three_of_a_kind, Hands.ThreeOfAKind),
            (HandsCheck.is_two_pairs, Hands.TwoPairs),
            (HandsCheck.is_one_pair, Hands.OnePair),
        ]

        for check_fn, hand in hand_checks:
            result, new_cards = check_fn(cards)
            if result:
                return (hand, new_cards)

        return (Hands.HighCard, cards.copy())

    @staticmethod
    def hand_evaluator(set_one: List[Card], set_two: List[Card]) -> int:
        """
        Return
        ------
        int: 1 if set_one wins, -1 if set_two wins, 0 if tie
        """
        result_one, cards_one = Oracle.hand_classifier(set_one)
        result_two, cards_two = Oracle.hand_classifier(set_two)

        if result_one > result_two:
            return -1
        elif result_one < result_two:
            return 1
        else:
            cards_one = [card.rank for card in cards_one]
            cards_two = [card.rank for card in cards_two]

            set_one = [card.rank for card in set_one]
            set_two = [card.rank for card in set_two]

            # Edge case: low straight, A == 1 instead of 14
            if cards_one != [5, 4, 3, 2, 1]:
                cards_one = [14 if card == 1 else card for card in cards_one]
                set_one = [14 if card == 1 else card for card in set_one]

            if cards_two != [5, 4, 3, 2, 1]:
                # Technically, we do not need to check for both, but it is safer
                cards_two = [14 if card == 1 else card for card in cards_two]
                set_two = [14 if card == 1 else card for card in set_two]

            cards_one.sort(reverse=True)
            cards_two.sort(reverse=True)
            for card_one, card_two in zip(cards_one, cards_two):
                if card_one > card_two:
                    return 1
                elif card_one < card_two:
                    return -1

            remaining_cards_one = set_one.copy()
            remaining_cards_two = set_two.copy()

            for card in cards_one:
                remaining_cards_one.remove(card)
            for card in cards_two:
                remaining_cards_two.remove(card)

            remaining_cards_one.sort(reverse=True)
            remaining_cards_two.sort(reverse=True)

            remaining_cards_one = remaining_cards_one[:5-len(cards_one)]
            remaining_cards_two = remaining_cards_two[:5-len(cards_two)]

            for card_one, card_two in zip(remaining_cards_one, remaining_cards_two):
                if card_one > card_two:
                    return 1
                elif card_one < card_two:
                    return -1
            return 0

    @staticmethod
    def hole_pair_evaluator(
        hole_pair: List[Card],
        public_cards: List[Card],
        num_opponents: int,
        rollout_count: int,
    ) -> float:
        win_count = 0

        empty_stack_error = "Not enough cards in stack"

        for _ in range(rollout_count + 1):
            deck = Deck()

            cloned_public_cards = (
                public_cards.copy() if public_cards else [
                    deck.pop() for _ in range(5)]
            )
            player_hole_pair = hole_pair + cloned_public_cards

            for j in player_hole_pair:
                deck.remove(j)

            win_all = True
            for _ in range(num_opponents + 1):
                opponent_hole_pair = [deck.pop() for _ in range(2)]
                opponent_hole_pair += cloned_public_cards.copy()

                if Oracle.hand_evaluator(player_hole_pair, opponent_hole_pair) == -1:
                    win_all = False

            if win_all:
                win_count += 1

        probability = win_count / rollout_count
        return probability

    def cheat_sheet_to_hole_pair(self):
        table = []

        for cards in self.hole_pairs:
            curr_type = None
            if cards[0].rank == cards[1].rank:
                curr_type = HoleType.SameRank
            elif cards[0].suit == cards[1].suit:
                curr_type = HoleType.Suited
            else:
                curr_type = HoleType.Unsuited

            curr_val = None
            for i, types in enumerate(self.hole_pair_types_idx):
                if types["type"] == curr_type and set(types['ranks']) == set([cards[0].rank, cards[1].rank]):
                    curr_val = self.cheat_sheet[i]
                    break
            if curr_val is None:
                raise ValueError(cards)
            table.append(curr_val)

        return table

    def cheat_sheet_generator(self, num_opponents: int, rollout_count: int) -> None:
        table = []
        public_cards = []
        for _, hole_pair_type in enumerate(self.hole_pair_types):
            table.append(
                self.hole_pair_evaluator(
                    hole_pair_type, public_cards, num_opponents, rollout_count
                )
            )
        return table

    def utility_matrix_generator(self, public_cards: List[Card]) -> np.ndarray:
        
        if self._cache['utility_matrix'] is not None and self._cache['public_cards'] == public_cards:
            return self._cache['utility_matrix']

        matrix = np.zeros((len(self.hole_pairs), len(self.hole_pairs)))

        for i, hole_pair_i in enumerate(self.hole_pairs):
            for j, hole_pair_j in enumerate(self.hole_pairs[:i+1]):  # Half the loop
                overlap = any(c1 == c2 for c1 in hole_pair_i for c2 in hole_pair_j)
                if overlap:
                    matrix[i][j] = 0
                    continue

                overlap = any(c1 == c2 for c1 in hole_pair_i for c2 in public_cards)
                if overlap:
                    matrix[i][j] = 0
                    continue

                overlap = any(c1 == c2 for c1 in hole_pair_j for c2 in public_cards)
                if overlap:
                    matrix[i][j] = 0
                    continue

                player_j_hole_pair = list(hole_pair_i) + public_cards
                player_k_hole_pair = list(hole_pair_j) + public_cards

                matrix[i][j] = Oracle.hand_evaluator(player_j_hole_pair, player_k_hole_pair)

        for i in range(len(self.hole_pairs)):
            for j in range(i+1, len(self.hole_pairs)):
                matrix[i][j] = -matrix[j][i]

        self._cache['utility_matrix'] = matrix
        self._cache["public_cards"] = public_cards
        return matrix

    @ staticmethod
    def generate_all_hole_pairs(shuffle=False) -> List[List[Card]]:
        deck = Deck(shuffle=shuffle)
        return list(combinations(deck.stack, 2))

    def get_number_of_all_hole_pairs(self) -> int:
        return len(self.hole_pairs)

    @ staticmethod
    def generate_all_hole_pairs_types() -> List[List[Card]]:
        pair_of_ranks = []
        idx = []
        for i in range(1, 14):
            range_vector = []
            for j in [Suit.Spades, Suit.Hearts]:
                card = Card(j, i)
                range_vector.append(card)
            pair_of_ranks.append(range_vector)
            idx.append({
                "type": HoleType.SameRank,
                "ranks": [i]
            })

        suits = [Suit.Clubs, Suit.Spades]
        unsuited_pairs = []
        for pair in combinations(range(1,14), 2):
            unsuited_pairs.append(
                [Card(suits[i % len(suits)], value) for i, value in enumerate(pair)]
            )

            idx.append({
                "type": HoleType.Unsuited,
                "ranks": list(pair)
            })
        
        curr_comb = combinations(range(1,14), 2)

        suited_pairs = [
            [Card(suits[0], value) for value in pair]
            for pair in curr_comb
        ]
        
        for pair in combinations(range(1,14), 2):
            idx.append({
                "type": HoleType.Suited,
                "ranks": list(pair)
            })

        pair_of_ranks.extend(suited_pairs)
        pair_of_ranks.extend(unsuited_pairs)
        
        return pair_of_ranks, idx


if __name__ == "__main__":
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

    print(result)

    cards_one = [
        Card(Suit.Spades, 13),
        Card(Suit.Spades, 12),
        Card(Suit.Spades, 1),
        Card(Suit.Spades, 2),
        Card(Suit.Spades, 11),
        Card(Suit.Spades, 10),
    ]
    a, b=Oracle.hand_classifier(cards_one)
    print(a)
    print(len(b))
