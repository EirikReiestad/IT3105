from typing import List, Tuple, Optional
from enum import Enum
from itertools import combinations
import unittest
from src.poker_oracle.deck import Deck
from src.poker_oracle.hands import HandsCheck, Hands
from .deck import Card, Suit


class Oracle:

    @staticmethod
    def hand_classifier(cards: List[Card]) -> Tuple[Hands, List[Card]]:
        if not (5 < len(cards) < 7):
            return None

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
        result_one, cards_one = Oracle.hand_classifier(set_one)
        result_two, cards_two = Oracle.hand_classifier(set_two)

        if result_one > result_two:
            return -1
        elif result_one < result_two:
            return 1
        else:
            unique_vec1 = [x for x in set_one if x not in cards_one]
            unique_vec2 = [x for x in set_two if x not in cards_two]

            if not unique_vec1 and not unique_vec2:
                return 0
            else:
                max_card_one = max(
                    unique_vec1, key=lambda x: x.rank, default=None)
                max_card_two = max(
                    unique_vec2, key=lambda x: x.rank, default=None)

                if max_card_one and max_card_two:
                    comparison_result = max_card_one.rank - max_card_two.rank
                    if comparison_result > 0:
                        return 1
                    elif comparison_result < 0:
                        return -1
                    else:
                        return 0
                elif max_card_one:
                    return 1
                elif max_card_two:
                    return -1
                else:
                    return 0

    @staticmethod
    def hole_pair_evaluator(
        self,
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

    @staticmethod
    def cheat_sheet_generator(self, num_opponents: int, rollout_count: int) -> None:
        hole_pair_types = Oracle.generate_all_hole_pairs_types()

        table = []
        public_cards = []
        for i in range(len(hole_pair_types)):
            table.append(
                self.hole_pair_evaluator(
                    hole_pair_types[i], public_cards, num_opponents, rollout_count
                )
            )

    @staticmethod
    def utility_matrix_generator(public_cards: List[Card]) -> List[List[int]]:
        hole_pairs = Oracle.generate_all_hole_pairs()
        matrix = [[0] * len(hole_pairs) for _ in range(len(hole_pairs))]

        for i, hole_pair_i in enumerate(hole_pairs):
            for j, hole_pair_j in enumerate(hole_pairs):
                overlap = any(
                    c1 == c2 for c1 in hole_pair_i for c2 in hole_pair_j)
                if overlap:
                    matrix[i][j] = 0
                    continue

                overlap = any(
                    c1 == c2 for c1 in hole_pair_i for c2 in public_cards)
                if overlap:
                    matrix[i][j] = 0
                    continue

                overlap = any(
                    c1 == c2 for c1 in hole_pair_j for c2 in public_cards)
                if overlap:
                    matrix[i][j] = 0
                    continue

                player_j_hole_pair = hole_pair_i + public_cards
                player_k_hole_pair = hole_pair_j + public_cards

                matrix[i][j] = Oracle.hand_evaluator(
                    player_j_hole_pair, player_k_hole_pair
                )

        return matrix

    @staticmethod
    def generate_all_hole_pairs() -> List[List[Card]]:
        deck = Deck()
        return list(combinations(deck.stack, 2))

    @staticmethod
    def get_number_of_all_hole_pairs() -> int:
        return len(Oracle.generate_all_hole_pairs())

    @staticmethod
    def generate_all_hole_pairs_types() -> List[List[Card]]:
        pair_of_ranks = []
        for i in range(13):
            range_vector = []
            for j in [Suit.Spades, Suit.Hearts]:
                card = Card(j, i)
                range_vector.append(card)
            pair_of_ranks.append(range_vector)

        suits = [Suit.Clubs, Suit.Spades]
        suited_pairs = []
        for pair in combinations(range(13), 2):
            suited_pairs.append(
                [Card(suits[value % len(suits)], value) for value in pair]
            )
        unsuited_pairs = [
            [Card(suits[0], value) for value in pair]
            for pair in combinations(range(13), 2)
        ]

        pair_of_ranks.extend(suited_pairs)
        pair_of_ranks.extend(unsuited_pairs)
        return pair_of_ranks
