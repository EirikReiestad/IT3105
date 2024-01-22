use crate::card::Card;
use itertools::Itertools;
use std::collections::HashMap;
use std::collections::HashSet;

#[derive(PartialEq, PartialOrd)]
enum Hands {
    RoyalFlush,
    StraightFlush,
    FourOfAKind,
    FullHouse,
    Flush,
    Straight,
    ThreeOfAKind,
    TwoPairs,
    OnePair,
    HighCard,
}

pub struct HandsCheck {}

impl HandsCheck {
    pub fn is_royal_flush(&mut self, cards: &Vec<Card>) -> bool {
        let cards = cards.clone();
        let combinations: Vec<Vec<Card>> = cards.into_iter().combinations(5).collect();
        let target_ranks: HashSet<usize> = [10, 11, 12, 13, 14].iter().cloned().collect();

        for combination in combinations {
            let mut unique_ranks = HashSet::new();
            let unique_cards: Vec<Card> = combination
                .into_iter()
                .filter(|card| unique_ranks.insert(card.rank))
                .collect();

            if let Some(first_suit) = unique_cards.first().map(|first_card| first_card.suit) {
                return unique_ranks == target_ranks
                    && unique_cards.iter().all(|card| card.suit == first_suit);
            }
        }
        false
    }

    pub fn is_straight_flush(&mut self, cards: &Vec<Card>) -> bool {
        let cards = cards.clone();
        let combinations: Vec<Vec<Card>> = cards.into_iter().combinations(5).collect();

        for combination in combinations {
            let mut unique_suits = HashSet::new();
            let mut unique_cards: Vec<Card> = combination
                .into_iter()
                .filter(|card| unique_suits.insert(card.suit))
                .collect();

            unique_cards.sort_by(|a, b| b.rank.cmp(&a.rank));
            let is_one_apart = unique_cards
                .windows(2)
                .all(|window| (window[0].rank - window[1].rank) == 1);

            if unique_suits.len() == 1 && is_one_apart {
                true;
            }
        }
        false
    }
    pub fn is_four_of_a_kind(&mut self, cards: &Vec<Card>) -> bool {
        let cards = cards.clone();
        let combinations: Vec<Vec<Card>> = cards.into_iter().combinations(5).collect();

        for combination in combinations {
            let mut rank_counts: HashMap<usize, usize> = HashMap::new();

            // Iterate over each instance and update the count in the HashMap
            for card in combination {
                let count = rank_counts.entry(card.rank).or_insert(0);
                *count += 1;
            }

            let has_count_4 = rank_counts.values().any(|&count| count == 4);

            return has_count_4;
        }
        false
    }
    pub fn is_full_house(&mut self, cards: &Vec<Card>) -> bool {
        let cards = cards.clone();
        let combinations: Vec<Vec<Card>> = cards.into_iter().combinations(5).collect();

        for combination in combinations {
            let mut rank_counts: HashMap<usize, usize> = HashMap::new();

            // Iterate over each instance and update the count in the HashMap
            for card in combination {
                let count = rank_counts.entry(card.rank).or_insert(0);
                *count += 1;
            }

            let has_count_2 = rank_counts.values().any(|&count| count == 2);
            let has_count_3 = rank_counts.values().any(|&count| count == 3);

            return has_count_2 && has_count_3;
        }
        false
    }

    pub fn is_flush(&mut self, cards: &Vec<Card>) -> bool {
        let cards = cards.clone();
        let combinations: Vec<Vec<Card>> = cards.into_iter().combinations(5).collect();

        for combination in combinations {
            let mut unique_suits = HashSet::new();
            let mut unique_cards: Vec<Card> = combination
                .into_iter()
                .filter(|card| unique_suits.insert(card.suit))
                .collect();

            unique_cards.sort_by(|a, b| b.rank.cmp(&a.rank));
            let is_one_apart = unique_cards
                .windows(2)
                .all(|window| (window[0].rank - window[1].rank) == 1);

            if unique_suits.len() == 1 && !is_one_apart {
                true;
            }
        }
        false
    }

    pub fn is_straight(&mut self, cards: &Vec<Card>) -> bool {
        let cards = cards.clone();
        let combinations: Vec<Vec<Card>> = cards.into_iter().combinations(5).collect();

        for mut combination in combinations {
            combination.sort_by(|a, b| b.rank.cmp(&a.rank));
            let is_one_apart = combination
                .windows(2)
                .all(|window| (window[0].rank - window[1].rank) == 1);
            if is_one_apart {
                true;
            }
        }
        false
    }
    pub fn is_three_of_a_kind(&mut self, cards: &Vec<Card>) -> bool {
        let cards = cards.clone();
        let combinations: Vec<Vec<Card>> = cards.into_iter().combinations(5).collect();

        for combination in combinations {
            let mut rank_counts: HashMap<usize, usize> = HashMap::new();

            // Iterate over each instance and update the count in the HashMap
            for card in combination {
                let count = rank_counts.entry(card.rank).or_insert(0);
                *count += 1;
            }

            let has_count_1 = rank_counts.values().any(|&count| count == 1);
            let has_count_3 = rank_counts.values().any(|&count| count == 3);

            return has_count_1 && has_count_3;
        }
        false
    }
    pub fn is_two_pairs(&mut self, cards: &Vec<Card>) -> bool {
        let cards = cards.clone();
        let combinations: Vec<Vec<Card>> = cards.into_iter().combinations(5).collect();

        for combination in combinations {
            let mut rank_counts: HashMap<usize, usize> = HashMap::new();

            // Iterate over each instance and update the count in the HashMap
            for card in combination {
                let count = rank_counts.entry(card.rank).or_insert(0);
                *count += 1;
            }

            let count_2_entries: Vec<_> =
                rank_counts.values().filter(|&count| *count == 2).collect();
            return count_2_entries.len() == 2;
        }
        false
    }

    pub fn is_one_pair(&mut self, cards: &Vec<Card>) -> bool {
        let combinations: Vec<Vec<Card>> = cards.into_iter().cloned().combinations(5).collect();
        for combination in combinations {
            let mut unique_ranks = HashSet::new();
            let _: Vec<Card> = combination
                .into_iter()
                .filter(|card| unique_ranks.insert(card.rank))
                .collect();

            if unique_ranks.len() == 4 {
                true;
            }
        }
        false
    }
}
