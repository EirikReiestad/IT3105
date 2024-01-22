use std::vec;

use crate::card::{Card, Deck, Suit};
use crate::hands::{Hands, HandsCheck};
use itertools::Itertools;

pub struct Oracle {}

impl Oracle {
    pub fn hand_classifier(&mut self, cards: &Vec<Card>) -> (Hands, Vec<Card>) {
        if !(5 < cards.len() && cards.len() > 7) {
            return (Hands::None, Vec::new());
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
            let (result, new_cards) = check_fn(&cards);
            if result {
                return (hand, new_cards);
            }
        }

        (Hands::HighCard, cards.to_vec())
    }

    pub fn hand_evaluator(&mut self, set_one: &Vec<Card>, set_two: &Vec<Card>) -> isize {
        let (result_one, cards_one) = self.hand_classifier(set_one);
        let (result_two, cards_two) = self.hand_classifier(set_two);

        if result_one > result_two {
            1
        } else if result_one < result_two {
            -1
        } else {
            let unique_vec1: Vec<_> = set_one
                .iter()
                .filter(|&x| !cards_one.contains(x))
                .cloned()
                .collect();

            let unique_vec2: Vec<_> = set_two
                .iter()
                .filter(|&x| !cards_two.contains(x))
                .cloned()
                .collect();

            if unique_vec1.len() == 0 && unique_vec2.len() == 0 {
                0
            } else {
                let max_card_one: Option<&Card> = unique_vec1.iter().max_by_key(|card| &card.rank);
                let max_card_two: Option<&Card> = unique_vec1.iter().max_by_key(|card| &card.rank);

                match (max_card_one, max_card_two) {
                    (Some(card1), Some(card2)) => {
                        if card1.rank > card2.rank {
                            1
                        } else if card1.rank < card2.rank {
                            -1
                        } else {
                            0
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
        hole_pair: &Vec<Card>,
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
                .cloned()
                .chain(cloned_public_cards.iter().cloned())
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
                opponent_hole_pair.extend(public_cards.iter().cloned());

                if self.hand_evaluator(&player_hole_pair, &opponent_hole_pair) == -1 {
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
        // for all the pairs use the method below
        let hole_pair_types = self.generate_all_hole_pairs_types();
        let num_hole_pairs = hole_pair_types.len();

        let mut table: Vec<f32> = vec![0.0; num_hole_pairs];
        let public_cards: Vec<Card> = Vec::new();
        for i in 0..num_hole_pairs {
            table[i] = self.hole_pair_evaluator(
                &hole_pair_types[i],
                &public_cards,
                num_opponents,
                rollout_count,
            );
        }
    }

    pub fn utility_matrix_generator(&mut self, public_cards: Vec<Card>) -> Vec<Vec<isize>> {
        let hole_pairs = self.generate_all_hole_pairs();

        let num_hole_pairs = hole_pairs.len();
        let mut matrix = vec![vec![0; num_hole_pairs]; num_hole_pairs];

        for i in 0..num_hole_pairs {
            for j in 0..num_hole_pairs {
                let overlap = hole_pairs[i].iter().any(|c1| {
                    hole_pairs[j]
                        .iter()
                        .any(|c2| c1.rank == c2.rank && c1.suit == c2.suit)
                });
                if overlap {
                    matrix[i][j] = 0;
                    continue;
                }

                let overlap = hole_pairs[i].iter().any(|c1| {
                    public_cards
                        .iter()
                        .any(|c2| c1.rank == c2.rank && c1.suit == c2.suit)
                });
                if overlap {
                    matrix[i][j] = 0;
                    continue;
                }

                let overlap = hole_pairs[j].iter().any(|c1| {
                    public_cards
                        .iter()
                        .any(|c2| c1.rank == c2.rank && c1.suit == c2.suit)
                });
                if overlap {
                    matrix[i][j] = 0;
                    continue;
                }

                let player_j_hole_pair: Vec<_> = hole_pairs[i]
                    .iter()
                    .cloned()
                    .chain(public_cards.iter().cloned())
                    .collect();

                let player_k_hole_pair: Vec<_> = hole_pairs[j]
                    .iter()
                    .cloned()
                    .chain(public_cards.iter().cloned())
                    .collect();
                matrix[i][j] = self.hand_evaluator(&player_j_hole_pair, &player_k_hole_pair);
            }
        }

        matrix
    }

    pub fn generate_all_hole_pairs(&mut self) -> Vec<Vec<Card>> {
        let deck = Deck::new();
        deck.stack.into_iter().combinations(2).collect()
    }

    pub fn generate_all_hole_pairs_types(&mut self) -> Vec<Vec<Card>> {
        let mut pair_of_ranks: Vec<Vec<Card>> = Vec::new();

        for i in 0..=12 {
            let mut range_vector: Vec<Card> = Vec::new();
            for j in vec![Suit::Spades, Suit::Hearts] {
                let card = Card {
                    suit: j,
                    rank: i
                };
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
                        suit: suits[(value % suits.len()) as usize],
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
