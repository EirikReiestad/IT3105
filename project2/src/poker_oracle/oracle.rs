use std::vec;
extern crate ndarray;
use ndarray::Array2;

use crate::card::{Card, Suit};
use crate::poker_oracle::hands::{HandsCheck, Hands};
use crate::poker_oracle::deck::Deck;
use itertools::Itertools;

pub struct Oracle;

impl Oracle {
    pub fn hand_classifier(cards: &Vec<Card>) -> Option<(Hands, Vec<Card>)> {
        if !(5 < cards.len() && cards.len() < 7) {
            return None;
        }

        let hand_checks: Vec<(&dyn Fn(&Vec<Card>) -> (bool, Vec<Card>), Hands)> = vec![
            (&HandsCheck::is_royal_flush, Hands::RoyalFlush),
            (&HandsCheck::is_straight_flush, Hands::StraightFlush),
            (&HandsCheck::is_four_of_a_kind, Hands::FourOfAKind),
            (&HandsCheck::is_full_house, Hands::FullHouse),
            (&HandsCheck::is_flush, Hands::Flush),
            (&HandsCheck::is_straight, Hands::Straight),
            (&HandsCheck::is_three_of_a_kind, Hands::ThreeOfAKind),
            (&HandsCheck::is_two_pairs, Hands::TwoPairs),
            (&HandsCheck::is_one_pair, Hands::OnePair),
        ];

        for (check_fn, hand) in hand_checks {
            let (result, new_cards) = check_fn(cards);
            if result {
                return Some((hand, new_cards));
            }
        }

        Some((Hands::HighCard, cards.to_vec()))
    }

    pub fn hand_evaluator(set_one: &Vec<Card>, set_two: &Vec<Card>) -> isize {
        let (result_one, cards_one) = Oracle::hand_classifier(set_one).unwrap();
        let (result_two, cards_two) = Oracle::hand_classifier(set_two).unwrap();

        if result_one > result_two {
            -1
        } else if result_one < result_two {
            1
        } else {
            let unique_vec1: Vec<&Card> =
                set_one.iter().filter(|&x| !cards_one.contains(x)).collect();

            let unique_vec2: Vec<&Card> =
                set_two.iter().filter(|&x| !cards_two.contains(x)).collect();

            if unique_vec1.is_empty() && unique_vec2.is_empty() {
                0
            } else {
                let max_card_one: Option<&Card> =
                    unique_vec1.iter().max_by_key(|card| &card.rank).cloned();
                let max_card_two: Option<&Card> =
                    unique_vec2.iter().max_by_key(|card| &card.rank).cloned();

                match (max_card_one, max_card_two) {
                    (Some(card1), Some(card2)) => {
                        let comparison_result = card1.cmp(card2);
                        match comparison_result {
                            std::cmp::Ordering::Greater => 1,
                            std::cmp::Ordering::Less => -1,
                            std::cmp::Ordering::Equal => 0,
                        }
                    }
                    (Some(_), None) => 1,
                    (None, Some(_)) => -1,
                    (None, None) => 0,
                }
            }
        }
    }

    pub fn hole_pair_evaluator(
        &mut self,
        hole_pair: &[Card],
        public_cards: &Vec<Card>,
        num_opponents: usize,
        rollout_count: usize,
    ) -> f32 {
        let mut win_count = 0;

        let empty_stack_error = "Not enough cards in stack";

        for _ in 0..=rollout_count {
            let mut deck = Deck::new();
            deck.reset_stack();

            let cloned_public_cards: Vec<Card> = if public_cards.is_empty() {
                vec![deck.pop().expect(empty_stack_error); 5]
            } else {
                public_cards.clone()
            };

            let player_hole_pair = &hole_pair
                .iter()
                .chain(cloned_public_cards.iter())
                .cloned()
                .collect();

            for j in player_hole_pair {
                deck.remove(j);
            }
            // CREATE HOLE FOR EACH OPPONENTS
            let mut win_all: bool = true;
            for _ in 0..=num_opponents {
                let mut opponent_hole_pair: Vec<Card> = vec![
                    deck.pop().expect(empty_stack_error),
                    deck.pop().expect(empty_stack_error),
                ];
                opponent_hole_pair.extend(cloned_public_cards.iter().cloned());

                if Oracle::hand_evaluator(player_hole_pair, &opponent_hole_pair) == -1 {
                    win_all = false;
                }
            }

            if win_all {
                win_count += 1;
            }
        }
        let probability: f32 = win_count as f32 / rollout_count as f32;
        probability
    }

    pub fn cheat_sheet_generator(&mut self, num_opponents: usize, rollout_count: usize) {
        // TODO: task says jth hole-pair but isnt it hard to find it again? maybe use a hashmap to easier find which hole pair type
        let hole_pair_types = Oracle::generate_all_hole_pairs_types();

        let mut table: Vec<f32> = Vec::with_capacity(hole_pair_types.len());
        let public_cards: Vec<Card> = Vec::new();
        for i in 0..hole_pair_types.len() {
            table.push(self.hole_pair_evaluator(
                &hole_pair_types[i],
                &public_cards,
                num_opponents,
                rollout_count,
            ));
        }
    }

    pub fn utility_matrix_generator(public_cards: Vec<Card>) -> Array2<isize> {
        let hole_pairs = Oracle::generate_all_hole_pairs();

        let mut matrix: Array2<isize> = Array2::zeros((hole_pairs.len(), hole_pairs.len()));

        for i in 0..hole_pairs.len() {
            for j in 0..hole_pairs.len() {
                // two hole pairs overlap with each other
                let overlap = hole_pairs[i].iter().any(|c1| {
                    hole_pairs[j]
                        .iter()
                        .any(|c2| c1.rank == c2.rank && c1.suit == c2.suit)
                });
                if overlap {
                    matrix[[i, j]] = 0;
                    continue;
                }

                // overlap with public cards
                let overlap = hole_pairs[i].iter().any(|c1| {
                    public_cards
                        .iter()
                        .any(|c2| c1.rank == c2.rank && c1.suit == c2.suit)
                });
                if overlap {
                    matrix[[i, j]] = 0;
                    continue;
                }

                let overlap = hole_pairs[j].iter().any(|c1| {
                    public_cards
                        .iter()
                        .any(|c2| c1.rank == c2.rank && c1.suit == c2.suit)
                });
                if overlap {
                    matrix[[i, j]] = 0;
                    continue;
                }

                // merge player hole pairs to use in hand evaluator
                let player_j_hole_pair: Vec<Card> = hole_pairs[i]
                    .iter()
                    .chain(public_cards.iter())
                    .cloned()
                    .collect();

                let player_k_hole_pair: Vec<Card> = hole_pairs[j]
                    .iter()
                    .chain(public_cards.iter())
                    .cloned()
                    .collect();

                matrix[[i, j]] = Oracle::hand_evaluator(&player_j_hole_pair, &player_k_hole_pair);
            }
        }

        matrix
    }

    pub fn generate_all_hole_pairs() -> Vec<Vec<Card>> {
        let mut deck = Deck::new();
        deck.reset_stack();
        deck.stack.into_iter().combinations(2).collect()
    }

    pub fn generate_all_hole_pairs_types() -> Vec<Vec<Card>> {
        let mut pair_of_ranks: Vec<Vec<Card>> = Vec::new();

        for i in 0..=12 {
            let mut range_vector: Vec<Card> = Vec::new();
            for j in vec![Suit::Spades, Suit::Hearts] {
                let card = Card { suit: j, rank: i };
                range_vector.push(card);
            }
            pair_of_ranks.push(range_vector);
        }

        let suits = &[Suit::Clubs, Suit::Spades];
        let suited_pairs: Vec<Vec<Card>> = (0..=12)
            .combinations(2)
            .map(|pair| {
                pair.iter()
                    .map(|&value| Card {
                        suit: suits[value % suits.len()],
                        rank: value,
                    })
                    .collect()
            })
            .collect();
        let unsuited_pairs: Vec<Vec<Card>> = (0..=12)
            .combinations(2)
            .map(|pair| {
                pair.iter()
                    .map(|&value| Card {
                        suit: suits[0],
                        rank: value,
                    })
                    .collect()
            })
            .collect();

        pair_of_ranks.extend(suited_pairs);
        pair_of_ranks.extend(unsuited_pairs);
        pair_of_ranks
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_all_hole_pairs() {
        let pairs = Oracle::generate_all_hole_pairs();
        assert_eq!(pairs.len(), 1326);
    }

    #[test]
    fn generate_all_hole_pairs_types() {
        let pairs = Oracle::generate_all_hole_pairs_types();
        assert_eq!(pairs.len(), 169);
    }

    #[test]
    fn hand_classifier_royal_flush() {
        let cards = vec![
            Card::new(Suit::Clubs, 13),
            Card::new(Suit::Clubs, 12),
            Card::new(Suit::Clubs, 1),
            Card::new(Suit::Clubs, 2),
            Card::new(Suit::Clubs, 11),
            Card::new(Suit::Clubs, 10),
        ];

        let (result, _) = Oracle::hand_classifier(&cards).unwrap();
        assert_eq!(result, Hands::RoyalFlush);
    }

    #[test]
    fn hand_classifier_one_pair() {
        let cards = vec![
            Card::new(Suit::Clubs, 13),
            Card::new(Suit::Spades, 12),
            Card::new(Suit::Hearts, 1),
            Card::new(Suit::Clubs, 2),
            Card::new(Suit::Clubs, 12),
            Card::new(Suit::Clubs, 10),
        ];

        let (result, _) = Oracle::hand_classifier(&cards).unwrap();
        assert_eq!(result, Hands::OnePair);
    }

    #[test]
    fn hand_evaluator_player_one_lose() {
        let cards_one = vec![
            Card::new(Suit::Clubs, 13),
            Card::new(Suit::Spades, 12),
            Card::new(Suit::Hearts, 1),
            Card::new(Suit::Clubs, 2),
            Card::new(Suit::Clubs, 12),
            Card::new(Suit::Clubs, 10),
        ];

        let cards_two = vec![
            Card::new(Suit::Clubs, 13),
            Card::new(Suit::Clubs, 12),
            Card::new(Suit::Clubs, 1),
            Card::new(Suit::Clubs, 2),
            Card::new(Suit::Clubs, 11),
            Card::new(Suit::Clubs, 10),
        ];

        let result = Oracle::hand_evaluator(&cards_one, &cards_two);
        assert_eq!(result, -1);
    }

    #[test]
    fn hand_evaluator_player_one_win() {
        let cards_one = vec![
            Card::new(Suit::Clubs, 13),
            Card::new(Suit::Spades, 12),
            Card::new(Suit::Hearts, 1),
            Card::new(Suit::Clubs, 2),
            Card::new(Suit::Clubs, 12),
            Card::new(Suit::Clubs, 10),
        ];

        let cards_two = vec![
            Card::new(Suit::Clubs, 13),
            Card::new(Suit::Clubs, 12),
            Card::new(Suit::Clubs, 1),
            Card::new(Suit::Clubs, 2),
            Card::new(Suit::Clubs, 11),
            Card::new(Suit::Clubs, 10),
        ];

        let result = Oracle::hand_evaluator(&cards_two, &cards_one);
        assert_eq!(result, 1);
    }

    #[test]
    fn hand_evaluator_tie() {
        let cards_one = vec![
            Card::new(Suit::Spades, 13),
            Card::new(Suit::Spades, 12),
            Card::new(Suit::Spades, 1),
            Card::new(Suit::Spades, 2),
            Card::new(Suit::Spades, 11),
            Card::new(Suit::Spades, 10),
        ];

        let cards_two = vec![
            Card::new(Suit::Clubs, 13),
            Card::new(Suit::Clubs, 12),
            Card::new(Suit::Clubs, 1),
            Card::new(Suit::Clubs, 2),
            Card::new(Suit::Clubs, 11),
            Card::new(Suit::Clubs, 10),
        ];

        let result = Oracle::hand_evaluator(&cards_two, &cards_one);
        assert_eq!(result, 0);
    }

    #[test]
    fn hand_evaluator_resolve_tie_with_extra_cards() {
        let cards_one = vec![
            Card::new(Suit::Spades, 13),
            Card::new(Suit::Spades, 12),
            Card::new(Suit::Spades, 1),
            Card::new(Suit::Spades, 2),
            Card::new(Suit::Spades, 11),
            Card::new(Suit::Spades, 10)
        ];

        let cards_two = vec![
            Card::new(Suit::Clubs, 13),
            Card::new(Suit::Clubs, 12),
            Card::new(Suit::Clubs, 1),
            Card::new(Suit::Clubs, 5),
            Card::new(Suit::Clubs, 11),
            Card::new(Suit::Clubs, 10),
        ];

        let result = Oracle::hand_evaluator(&cards_one, &cards_two);
        assert_eq!(result, -1);
    }
}
