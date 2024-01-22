use crate::card::{Card, Suit};
use crate::hands::{Hands, HandsCheck};
use itertools::Itertools;
use rand::prelude::SliceRandom;
use rand::thread_rng;
use rand::Rng;
use std::ptr::null;

pub struct Deck {
    pub stack: Vec<Card>,
}

impl Deck {
    pub fn new() -> Self {
        Deck { stack: Vec::new() }
    }

    pub fn reset_stack(&mut self) -> () {
        let mut stack = Vec::with_capacity(52);
        for (i, suit) in (0..4).enumerate() {
            let suit = match suit {
                0 => Suit::CLUBS,
                1 => Suit::DIAMONDS,
                2 => Suit::HEARTS,
                3 => Suit::SPADES,
                _ => panic!("Invalid suit"),
            };

            for rank in 0..13 {
                stack.push(Card { suit, rank });
            }
        }
        let mut rng = thread_rng();
        stack.shuffle(&mut rng);
        self.stack = stack;
    }

    pub fn remove(&mut self, card: Card) -> () {
        self.stack.retain(|&i| i != card);
    }

    pub fn pop(&mut self) -> Option<Card> {
        self.stack.pop()
    }
}

pub struct Oracle {}

impl Oracle {
    pub fn hand_classifier(&mut self, cards: Vec<Card>) -> Hands {
        if !(5 <= cards.len() && cards.len() <= 7) {
            return Hands::None;
        }

        if HandsCheck::is_royal_flush(&cards) {
            return Hands::RoyalFlush;
        } else if HandsCheck::is_straight_flush(&cards) {
            return Hands::StraightFlush;
        } else if HandsCheck::is_four_of_a_kind(&cards) {
            return Hands::FourOfAKind;
        } else if HandsCheck::is_full_house(&cards) {
            return Hands::FullHouse;
        } else if HandsCheck::is_flush(&cards) {
            return Hands::Flush;
        } else if HandsCheck::is_straight(&cards) {
            return Hands::Straight;
        } else if HandsCheck::is_three_of_a_kind(&cards) {
            return Hands::ThreeOfAKind;
        } else if HandsCheck::is_two_pairs(&cards) {
            return Hands::TwoPairs;
        } else if HandsCheck::is_one_pair(&cards) {
            return Hands::OnePair;
        } else {
            return Hands::HighCard;
        }
    }

    pub fn hand_evaluator(&mut self, set_one: Vec<Card>, set_two: Vec<Card>) -> isize {
        let result_one: Hands = self.hand_classifier(set_one);
        let result_two: Hands = self.hand_classifier(set_two);

        if result_one > result_two {
            1
        } else if result_one < result_two {
            -1
        } else {
            0
        }
    }

    pub fn hole_pair_evaluator(
        &mut self,
        hole_pair: Vec<Card>,
        public_cards: Vec<Card>,
        num_opponents: usize,
        rollout_count: usize,
    ) -> f32 {
        let win_count = 0;
        let player_hole_pair: Vec<_> = hole_pair
            .iter()
            .cloned()
            .chain(public_cards.iter().cloned())
            .collect();
        let empty_stack_error = "Not enough cards in stack";

        for i in 0..=rollout_count {
            let deck = Deck::new();
            deck.reset_stack();

            if public_cards.len() == 0 {
                public_cards = vec![deck.pop().expect(empty_stack_error); 5];
                player_hole_pair = hole_pair
                    .iter()
                    .cloned()
                    .chain(public_cards.iter().cloned())
                    .collect();
            }

            for j in player_hole_pair {
                deck.remove(j);
            }
            // CREATE HOLE FOR EACH OPPONENTS
            let mut win_all: i32 = 1;
            for j in 0..=num_opponents {
                let opponent_hole_pair = vec![
                    deck.pop().expect(empty_stack_error),
                    deck.pop().expect(empty_stack_error),
                ];
                opponent_hole_pair.extend(public_cards);

                if self.hand_evaluator(player_hole_pair, opponent_hole_pair) == -1 {
                    win_all = 0
                }
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
                hole_pair_types[i],
                public_cards,
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
                matrix[i][j] = self.hand_evaluator(player_j_hole_pair, player_k_hole_pair);
            }
        }

        matrix
    }

    pub fn generate_all_hole_pairs(&mut self) -> Vec<Vec<Card>> {
        let deck = Deck::new();
        deck.stack.into_iter().combinations(2).collect()
    }
}
