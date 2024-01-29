use crate::card::{Card, Suit};
use itertools::Itertools;
use std::collections::HashMap;
use std::collections::HashSet;

#[derive(PartialEq, PartialOrd, Debug)]
pub enum Hands {
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

pub struct HandsCheck;

impl HandsCheck {
    pub fn is_royal_flush(cards: &Vec<Card>) -> (bool, Vec<Card>) {
        let combinations: Vec<Vec<&Card>> = cards.into_iter().combinations(5).collect();
        let target_ranks: HashSet<&usize> = [1, 10, 11, 12, 13].iter().collect();

        for combination in combinations {
            let mut unique_suits = HashMap::new();
            for card in combination {
                unique_suits
                    .entry(card.suit)
                    .or_insert(Vec::new())
                    .push(&card.rank);
            }
            for (suit, ranks) in unique_suits {
                let unique_ranks: HashSet<&usize> = ranks.into_iter().collect();

                if unique_ranks == target_ranks {
                    let new_cards: Vec<Card> = unique_ranks
                        .into_iter()
                        .map(|rank| Card::new(suit, *rank))
                        .collect();
                    return (true, new_cards);
                }
            }
        }
        (false, vec![])
    }

    pub fn is_straight_flush(cards: &Vec<Card>) -> (bool, Vec<Card>) {
        let (result, new_cards) = HandsCheck::is_royal_flush(cards);
        let mut sorted_cards = cards.clone();
        sorted_cards.sort_by(|a, b| b.rank.cmp(&a.rank));
        let combinations: Vec<Vec<&Card>> = sorted_cards.iter().combinations(5).collect();

        if result {
            return (result, new_cards);
        }

        for combination in combinations {
            let mut unique_suits = HashSet::new();
            let mut unique_ranks = HashSet::new();
            let first_suit = combination[0].suit;
            for card in combination {
                unique_suits.insert(card.suit);
                unique_ranks.insert(card.rank);
            }

            let mut ranks_vec: Vec<usize> = unique_ranks.into_iter().collect();
            ranks_vec.sort();

            let is_one_apart = ranks_vec
                .windows(2)
                .all(|window| (window[1] - window[0]) == 1);

            if unique_suits.len() == 1 && ranks_vec.len() == 5 && is_one_apart {
                let new_cards: Vec<Card> = ranks_vec
                    .into_iter()
                    .map(|rank: usize| Card::new(first_suit, rank))
                    .collect();
                return (true, new_cards);
            }
        }
        (false, vec![])
    }

    pub fn is_four_of_a_kind(cards: &Vec<Card>) -> (bool, Vec<Card>) {
        let combinations: Vec<Vec<&Card>> = cards.into_iter().combinations(5).collect();

        for combination in combinations {
            let mut unique_ranks = HashMap::new();
            for card in combination {
                unique_ranks
                    .entry(card.rank)
                    .or_insert(Vec::new())
                    .push(card.suit);
            }
            for (rank, suits) in unique_ranks {
                let unique_suits: HashSet<_> = suits.into_iter().collect();
                if unique_suits.len() == 4 {
                    let new_cards: Vec<Card> = unique_suits
                        .into_iter()
                        .map(|suit| Card::new(suit, rank))
                        .collect();
                    return (true, new_cards);
                }
            }
        }
        (false, vec![])
    }

    pub fn is_full_house(cards: &Vec<Card>) -> (bool, Vec<Card>) {
        let mut sorted_cards = cards.clone();
        sorted_cards.sort_by(|a, b| b.rank.cmp(&a.rank));
        let combinations: Vec<Vec<&Card>> = sorted_cards.iter().combinations(5).collect();

        for combination in combinations {
            let mut rank_counts: HashMap<usize, Vec<Suit>> = HashMap::new();

            for card in &combination {
                rank_counts.entry(card.rank).or_default().push(card.suit);
            }

            let mut cards_len_2: Vec<Card> = Vec::new();
            let mut cards_len_3: Vec<Card> = Vec::new();

            for (rank, suits) in rank_counts.iter() {
                match suits.len() {
                    2 => cards_len_2.extend(suits.iter().map(|suit| Card::new(*suit, *rank))),
                    3 => cards_len_3.extend(suits.iter().map(|suit| Card::new(*suit, *rank))),
                    _ => {}
                }
            }

            if !cards_len_2.is_empty() && !cards_len_3.is_empty() {
                cards_len_2.extend(cards_len_3);
                return (true, cards_len_2);
            }
        }

        (false, vec![])
    }

    pub fn is_flush(cards: &Vec<Card>) -> (bool, Vec<Card>) {
        let mut sorted_cards = cards.clone();
        sorted_cards.sort_by(|a, b| b.rank.cmp(&a.rank));
        let combinations: Vec<Vec<&Card>> = sorted_cards.iter().combinations(5).collect();

        for combination in combinations {
            let mut unique_suits = HashSet::new();
            let mut unique_ranks = HashSet::new();
            let mut first_suit = Suit::Spades;
            for card in combination {
                first_suit = card.suit;
                unique_suits.insert(card.suit);
                unique_ranks.insert(card.rank);
            }

            let mut ranks_vec: Vec<usize> = unique_ranks.into_iter().collect();
            ranks_vec.sort();

            let is_one_apart = ranks_vec
                .windows(2)
                .all(|window| (window[1] - window[0]) == 1);

            if unique_suits.len() == 1 && ranks_vec.len() == 5 && !is_one_apart {
                let new_cards: Vec<Card> = ranks_vec
                    .into_iter()
                    .map(|rank: usize| Card::new(first_suit, rank))
                    .collect();
                return (true, new_cards);
            }
        }
        (false, vec![])
    }

    pub fn is_straight(cards: &Vec<Card>) -> (bool, Vec<Card>) {
        let mut sorted_cards = cards.clone();
        sorted_cards.sort_by(|a, b| b.rank.cmp(&a.rank));
        let combinations: Vec<Vec<&Card>> = sorted_cards.iter().combinations(5).collect();
        let target_ranks: HashSet<&usize> = [1, 10, 11, 12, 13].iter().collect();

        for combination in combinations {
            let mut unique_suits = HashSet::new();
            let mut unique_ranks = HashSet::new();

            for card in combination.clone() {
                unique_suits.insert(card.suit);
                unique_ranks.insert(&card.rank);
            }
            let royal = unique_ranks == target_ranks;
            let mut ranks_vec: Vec<&usize> = unique_ranks.into_iter().collect();
            ranks_vec.sort();

            let is_one_apart = ranks_vec
                .windows(2)
                .all(|window| (window[1] - window[0]) == 1);

            if unique_suits.len() != 1 && ranks_vec.len() == 5 && (is_one_apart || royal) {
                let new_cards: Vec<Card> = combination.iter().map(|&card| card.clone()).collect();
                return (true, new_cards);
            }
        }
        (false, vec![])
    }

    pub fn is_three_of_a_kind(cards: &Vec<Card>) -> (bool, Vec<Card>) {
        let mut sorted_cards = cards.clone();
        sorted_cards.sort_by(|a, b| b.rank.cmp(&a.rank));
        let combinations: Vec<Vec<&Card>> = sorted_cards.iter().combinations(5).collect();

        for combination in combinations {
            let mut unique_ranks = HashMap::new();
            for card in combination {
                unique_ranks
                    .entry(card.rank)
                    .or_insert(Vec::new())
                    .push(card.suit);
            }

            let length_unique_ranks = unique_ranks.len();
            for (rank, suits) in unique_ranks {
                let unique_suits: HashSet<_> = suits.into_iter().collect();
                if unique_suits.len() == 3 && length_unique_ranks > 2 {
                    let new_cards: Vec<Card> = unique_suits
                        .into_iter()
                        .map(|suit| Card::new(suit, rank))
                        .collect();
                    return (true, new_cards);
                }
            }
        }
        (false, vec![])
    }
    pub fn is_two_pairs(cards: &Vec<Card>) -> (bool, Vec<Card>) {
        let mut sorted_cards = cards.clone();
        sorted_cards.sort_by(|a, b| b.rank.cmp(&a.rank));
        let combinations: Vec<Vec<&Card>> = sorted_cards.iter().combinations(5).collect();

        for combination in combinations {
            let mut unique_ranks = HashMap::new();
            for card in combination {
                unique_ranks
                    .entry(card.rank)
                    .or_insert(Vec::new())
                    .push(card.suit);
            }
            if unique_ranks.len() == 3 {
                let filtered_map: HashMap<usize, Vec<Suit>> = unique_ranks
                    .into_iter()
                    .filter(|(_, v)| v.len() == 2)
                    .collect();
                if filtered_map.len() != 2 {
                    continue;
                }

                let new_cards: Vec<Card> = filtered_map
                    .iter()
                    .flat_map(|(rank, suits)| suits.iter().map(move |suit| Card::new(*suit, *rank)))
                    .collect();

                return (true, new_cards);
            }
        }
        (false, vec![])
    }

    pub fn is_one_pair(cards: &Vec<Card>) -> (bool, Vec<Card>) {
        let mut sorted_cards = cards.clone();
        sorted_cards.sort_by(|a, b| b.rank.cmp(&a.rank));
       let combinations: Vec<Vec<&Card>> = sorted_cards.iter().combinations(5).collect();

        for combination in combinations {
            let mut unique_ranks = HashMap::new();
            for card in combination {
                unique_ranks
                    .entry(card.rank)
                    .or_insert(Vec::new())
                    .push(card.suit);
            }
            if unique_ranks.len() == 4 {
                let filtered_map: HashMap<usize, Vec<Suit>> = unique_ranks
                    .into_iter()
                    .filter(|(_, v)| v.len() == 2)
                    .collect();

                if let Some((first_key, first_value)) = filtered_map.iter().next() {
                    let new_cards: Vec<Card> = first_value
                        .into_iter()
                        .map(|suit| Card::new(*suit, *first_key))
                        .collect();
                    return (true, new_cards);
                }
            }
        }
        (false, vec![])
    }
} 

#[cfg(test)]
mod tests {
    use super::*;
    mod royal_flush {
        use super::*;
        #[test]
        fn test_royal_flush_ok() {
            let cards = vec![
                Card::new(Suit::Spades, 10),
                Card::new(Suit::Hearts, 1),
                Card::new(Suit::Spades, 11),
                Card::new(Suit::Spades, 12),
                Card::new(Suit::Spades, 13),
                Card::new(Suit::Spades, 1),
                Card::new(Suit::Hearts, 10),
            ];
            let (result, mut cards): (bool, Vec<Card>) = HandsCheck::is_royal_flush(&cards);

            cards.sort_by(|a, b| a.rank.cmp(&b.rank));

            assert!(result);
            assert_eq!(
                cards,
                vec![
                    Card::new(Suit::Spades, 1),
                    Card::new(Suit::Spades, 10),
                    Card::new(Suit::Spades, 11),
                    Card::new(Suit::Spades, 12),
                    Card::new(Suit::Spades, 13),
                ]
            )
        }

        #[test]
        fn test_royal_flush_ok_2() {
            let cards = vec![
                Card::new(Suit::Hearts, 10),
                Card::new(Suit::Hearts, 1),
                Card::new(Suit::Spades, 11),
                Card::new(Suit::Spades, 12),
                Card::new(Suit::Spades, 13),
                Card::new(Suit::Spades, 1),
                Card::new(Suit::Spades, 10),
            ];
            let (result, mut cards): (bool, Vec<Card>) = HandsCheck::is_royal_flush(&cards);

            cards.sort_by(|a, b| a.rank.cmp(&b.rank));

            assert!(result);
            assert_eq!(
                cards,
                vec![
                    Card::new(Suit::Spades, 1),
                    Card::new(Suit::Spades, 10),
                    Card::new(Suit::Spades, 11),
                    Card::new(Suit::Spades, 12),
                    Card::new(Suit::Spades, 13),
                ]
            );
        }

        #[test]
        fn test_royal_flush_not_ok() {
            let cards = vec![
                Card::new(Suit::Hearts, 10),
                Card::new(Suit::Hearts, 1),
                Card::new(Suit::Spades, 11),
                Card::new(Suit::Spades, 8),
                Card::new(Suit::Spades, 13),
                Card::new(Suit::Spades, 1),
                Card::new(Suit::Spades, 10),
            ];
            let (result, mut cards): (bool, Vec<Card>) = HandsCheck::is_royal_flush(&cards);

            cards.sort_by(|a, b| a.rank.cmp(&b.rank));

            assert!(!result);
            assert_eq!(cards, vec![]);
        }
    }

    mod straight_flush {
        use super::*;
        #[test]
        fn test_straight_flush_ok() {
            let cards = vec![
                Card::new(Suit::Spades, 10),
                Card::new(Suit::Hearts, 1),
                Card::new(Suit::Spades, 11),
                Card::new(Suit::Spades, 12),
                Card::new(Suit::Spades, 13),
                Card::new(Suit::Spades, 1),
                Card::new(Suit::Spades, 10),
            ];
            let (result, mut cards): (bool, Vec<Card>) = HandsCheck::is_straight_flush(&cards);

            cards.sort_by(|a, b| a.rank.cmp(&b.rank));

            assert!(result);
            assert_eq!(
                cards,
                vec![
                    Card::new(Suit::Spades, 1),
                    Card::new(Suit::Spades, 10),
                    Card::new(Suit::Spades, 11),
                    Card::new(Suit::Spades, 12),
                    Card::new(Suit::Spades, 13),
                ]
            )
        }

        #[test]
        fn test_straight_flush_ok_2() {
            let cards = vec![
                Card::new(Suit::Spades, 2),
                Card::new(Suit::Hearts, 1),
                Card::new(Suit::Spades, 3),
                Card::new(Suit::Spades, 4),
                Card::new(Suit::Spades, 5),
                Card::new(Suit::Spades, 6),
                Card::new(Suit::Hearts, 10),
            ];
            let (result, mut cards): (bool, Vec<Card>) = HandsCheck::is_straight_flush(&cards);

            cards.sort_by(|a, b| a.rank.cmp(&b.rank));

            assert!(result);
            assert_eq!(
                cards,
                vec![
                    Card::new(Suit::Spades, 2),
                    Card::new(Suit::Spades, 3),
                    Card::new(Suit::Spades, 4),
                    Card::new(Suit::Spades, 5),
                    Card::new(Suit::Spades, 6),
                ]
            )
        }

        #[test]
        fn test_straight_flush_ok_3() {
            let cards = vec![
                Card::new(Suit::Spades, 2),
                Card::new(Suit::Hearts, 1),
                Card::new(Suit::Spades, 3),
                Card::new(Suit::Spades, 4),
                Card::new(Suit::Spades, 5),
                Card::new(Suit::Spades, 6),
                Card::new(Suit::Spades, 7),
            ];
            let (result, mut cards): (bool, Vec<Card>) = HandsCheck::is_straight_flush(&cards);

            cards.sort_by(|a, b| a.rank.cmp(&b.rank));

            assert_eq!(result, true);
            assert_eq!(
                cards,
                vec![
                    Card::new(Suit::Spades, 3),
                    Card::new(Suit::Spades, 4),
                    Card::new(Suit::Spades, 5),
                    Card::new(Suit::Spades, 6),
                    Card::new(Suit::Spades, 7)
                ]
            )
        }

        #[test]
        fn test_straight_flush_not_ok() {
            let cards = vec![
                Card::new(Suit::Spades, 2),
                Card::new(Suit::Hearts, 1),
                Card::new(Suit::Spades, 3),
                Card::new(Suit::Hearts, 8),
                Card::new(Suit::Spades, 5),
                Card::new(Suit::Spades, 6),
                Card::new(Suit::Hearts, 10),
            ];
            let (result, mut cards): (bool, Vec<Card>) = HandsCheck::is_straight_flush(&cards);

            cards.sort_by(|a, b| a.rank.cmp(&b.rank));

            assert!(!result);
            assert_eq!(cards, vec![])
        }

        #[test]
        fn test_straight_flush_not_ok_2() {
            let cards = vec![
                Card::new(Suit::Spades, 2),
                Card::new(Suit::Hearts, 1),
                Card::new(Suit::Spades, 3),
                Card::new(Suit::Hearts, 4),
                Card::new(Suit::Spades, 5),
                Card::new(Suit::Spades, 6),
                Card::new(Suit::Hearts, 10),
            ];
            let (result, mut cards): (bool, Vec<Card>) = HandsCheck::is_straight_flush(&cards);

            cards.sort_by(|a, b| a.rank.cmp(&b.rank));

            assert!(!result);
            assert_eq!(cards, vec![])
        }
    }

    mod four_of_a_kind {
        use super::*;
        #[test]
        fn test_four_of_a_kind_ok() {
            let cards = vec![
                Card::new(Suit::Hearts, 10),
                Card::new(Suit::Hearts, 1),
                Card::new(Suit::Spades, 1),
                Card::new(Suit::Clubs, 1),
                Card::new(Suit::Diamonds, 1),
                Card::new(Suit::Spades, 10),
            ];
            let (result, mut cards): (bool, Vec<Card>) = HandsCheck::is_four_of_a_kind(&cards);

            cards.sort_by(|a, b| a.rank.cmp(&b.rank));

            assert!(result);
            assert_eq!(cards.len(), 4);
        }

        #[test]
        fn test_four_of_a_kind_not_ok() {
            let cards = vec![
                Card::new(Suit::Hearts, 10),
                Card::new(Suit::Hearts, 1),
                Card::new(Suit::Spades, 2),
                Card::new(Suit::Clubs, 1),
                Card::new(Suit::Diamonds, 1),
                Card::new(Suit::Spades, 10),
            ];
            let (result, cards): (bool, Vec<Card>) = HandsCheck::is_four_of_a_kind(&cards);

            assert!(!result);
            assert_eq!(cards, vec![]);
        }

        
    }
    mod full_house {
        use super::*;
        #[test]
        fn test_full_house_ok() {
            let cards = vec![
                Card::new(Suit::Hearts, 10),
                Card::new(Suit::Clubs, 10),
                Card::new(Suit::Spades, 1),
                Card::new(Suit::Clubs, 1),
                Card::new(Suit::Diamonds, 1),
                Card::new(Suit::Spades, 11),
            ];
            let (result, mut cards): (bool, Vec<Card>) = HandsCheck::is_full_house(&cards);

            cards.sort_by(|a, b| a.rank.cmp(&b.rank));

            assert!(result);
            assert_eq!(
                cards,
                vec![
                    Card::new(Suit::Spades, 1),
                    Card::new(Suit::Clubs, 1),
                    Card::new(Suit::Diamonds, 1),
                    Card::new(Suit::Hearts, 10),
                    Card::new(Suit::Clubs, 10),
                ]
            );
        }

        #[test]
        fn test_full_house_not_ok() {
            let cards = vec![
                Card::new(Suit::Hearts, 10),
                Card::new(Suit::Clubs, 9),
                Card::new(Suit::Spades, 1),
                Card::new(Suit::Clubs, 1),
                Card::new(Suit::Diamonds, 1),
                Card::new(Suit::Hearts, 11),
            ];
            let (result, mut cards): (bool, Vec<Card>) = HandsCheck::is_full_house(&cards);

            cards.sort_by(|a, b| a.rank.cmp(&b.rank));

            assert!(!result);
            assert_eq!(cards, vec![]);
        }
    }

    mod flush {
        use super::*;
        
        #[test]
        fn test_flush_ok() {
            let cards = vec![
                Card::new(Suit::Hearts, 10),
                Card::new(Suit::Hearts, 7),
                Card::new(Suit::Hearts, 2),
                Card::new(Suit::Clubs, 1),
                Card::new(Suit::Hearts, 1),
                Card::new(Suit::Hearts, 11),
            ];
            let (result, mut cards): (bool, Vec<Card>) = HandsCheck::is_flush(&cards);

            cards.sort_by(|a, b| a.rank.cmp(&b.rank));

            assert!(result);
            assert_eq!(
                cards,
                vec![
                    Card::new(Suit::Hearts, 1),
                    Card::new(Suit::Hearts, 2),
                    Card::new(Suit::Hearts, 7),
                    Card::new(Suit::Hearts, 10),
                    Card::new(Suit::Hearts, 11),
                ]
            );
        }

        #[test]
        fn test_flush_ok2() {
            let cards = vec![
                Card::new(Suit::Hearts, 10),
                Card::new(Suit::Hearts, 7),
                Card::new(Suit::Hearts, 2),
                Card::new(Suit::Hearts, 11),
                Card::new(Suit::Hearts, 1),
                Card::new(Suit::Hearts, 13),
            ];
            let (result, mut cards): (bool, Vec<Card>) = HandsCheck::is_flush(&cards);

            cards.sort_by(|a, b| a.rank.cmp(&b.rank));

            assert!(result);
            assert_eq!(
                cards,
                vec![
                    Card::new(Suit::Hearts, 2),
                    Card::new(Suit::Hearts, 7),
                    Card::new(Suit::Hearts, 10),
                    Card::new(Suit::Hearts, 11),
                    Card::new(Suit::Hearts, 13),
                ]
            );
        }

        #[test]
        fn test_flush_not_ok() {
            let cards = vec![
                Card::new(Suit::Hearts, 10),
                Card::new(Suit::Hearts, 7),
                Card::new(Suit::Spades, 2),
                Card::new(Suit::Clubs, 1),
                Card::new(Suit::Hearts, 1),
                Card::new(Suit::Hearts, 11),
            ];
            let (result, mut cards): (bool, Vec<Card>) = HandsCheck::is_flush(&cards);

            cards.sort_by(|a, b| a.rank.cmp(&b.rank));

            assert!(!result);
            assert_eq!(cards, vec![]);
        }
    }
    mod straight {
        use super::*;
        #[test]
        fn test_straight_ok() {
            let cards = vec![
                Card::new(Suit::Hearts, 10),
                Card::new(Suit::Spades, 7),
                Card::new(Suit::Hearts, 8),
                Card::new(Suit::Clubs, 9),
                Card::new(Suit::Hearts, 13),
                Card::new(Suit::Hearts, 11),
            ];
            let (result, mut cards): (bool, Vec<Card>) = HandsCheck::is_straight(&cards);

            cards.sort_by(|a, b| a.rank.cmp(&b.rank));

            assert!(result);
            assert_eq!(
                cards,
                vec![
                    Card::new(Suit::Spades, 7),
                    Card::new(Suit::Hearts, 8),
                    Card::new(Suit::Clubs, 9),
                    Card::new(Suit::Hearts, 10),
                    Card::new(Suit::Hearts, 11),
                ]
            );
        }

        #[test]
        fn test_straight_ok2() {
            let cards = vec![
                Card::new(Suit::Hearts, 10),
                Card::new(Suit::Hearts, 7),
                Card::new(Suit::Clubs, 8),
                Card::new(Suit::Hearts, 9),
                Card::new(Suit::Spades, 13),
                Card::new(Suit::Clubs, 11),
                Card::new(Suit::Hearts, 12),
            ];
            let (result, mut cards): (bool, Vec<Card>) = HandsCheck::is_straight(&cards);

            cards.sort_by(|a, b| a.rank.cmp(&b.rank));

            assert!(result);
            assert_eq!(
                cards,
                vec![
                    Card::new(Suit::Hearts, 9),
                    Card::new(Suit::Hearts, 10),
                    Card::new(Suit::Clubs, 11),
                    Card::new(Suit::Hearts, 12),
                    Card::new(Suit::Spades, 13),
                ]
            );
        }

        #[test]
        fn test_straight_not_ok() {
            let cards = vec![
                Card::new(Suit::Hearts, 10),
                Card::new(Suit::Hearts, 7),
                Card::new(Suit::Hearts, 8),
                Card::new(Suit::Hearts, 9),
                Card::new(Suit::Hearts, 13),
                Card::new(Suit::Hearts, 11),
            ];
            let (result, mut cards): (bool, Vec<Card>) = HandsCheck::is_straight(&cards);

            cards.sort_by(|a, b| a.rank.cmp(&b.rank));

            assert!(!result);
            assert_eq!(cards, vec![]);
        }
    }
    mod three_of_a_kind {
        use super::*;
        #[test]
        fn test_three_of_a_kind_ok() {
            let cards = vec![
                Card::new(Suit::Hearts, 10),
                Card::new(Suit::Spades, 10),
                Card::new(Suit::Hearts, 8),
                Card::new(Suit::Clubs, 10),
                Card::new(Suit::Hearts, 13),
                Card::new(Suit::Hearts, 11),
            ];
            let (result, cards): (bool, Vec<Card>) = HandsCheck::is_three_of_a_kind(&cards);

            assert!(result);
            assert_eq!(3, cards.len());
        }

        #[test]
        fn test_three_of_a_kind_not_ok() {
            let cards = vec![
                Card::new(Suit::Hearts, 10),
                Card::new(Suit::Spades, 10),
                Card::new(Suit::Clubs, 13),
                Card::new(Suit::Clubs, 10),
                Card::new(Suit::Hearts, 13),
                Card::new(Suit::Spades, 13),
            ];
            let (result, cards): (bool, Vec<Card>) = HandsCheck::is_three_of_a_kind(&cards);

            assert!(!result);
            assert_eq!(0, cards.len());
        }
    }

    mod two_pair {
        use super::*;
        #[test]
        fn test_two_pair_ok() {
            let cards = vec![
                Card::new(Suit::Hearts, 10),
                Card::new(Suit::Spades, 10),
                Card::new(Suit::Hearts, 8),
                Card::new(Suit::Clubs, 8),
                Card::new(Suit::Hearts, 13),
                Card::new(Suit::Clubs, 13),
            ];
            let (result, cards): (bool, Vec<Card>) = HandsCheck::is_two_pairs(&cards);

            assert!(result);
            assert_eq!(4, cards.len());
            assert_eq!(true, cards.iter().any(|x| x.rank == 13));
            assert_eq!(true, cards.iter().any(|x| x.rank == 10));
            assert_eq!(true, cards.iter().all(|x| x.rank != 8));
        }

        #[test]
        fn test_two_pair_not_ok() {
            let cards = vec![
                Card::new(Suit::Hearts, 10),
                Card::new(Suit::Spades, 10),
                Card::new(Suit::Hearts, 7),
                Card::new(Suit::Clubs, 8),
                Card::new(Suit::Hearts, 13),
                Card::new(Suit::Hearts, 11),
            ];
            let (result, cards): (bool, Vec<Card>) = HandsCheck::is_two_pairs(&cards);

            assert!(!result);
            assert_eq!(0, cards.len());
        }
    }

    mod one_pair {
        use super::*;
        #[test]
        fn test_one_pair_ok() {
            let cards = vec![
                Card::new(Suit::Hearts, 10),
                Card::new(Suit::Spades, 10),
                Card::new(Suit::Hearts, 8),
                Card::new(Suit::Clubs, 8),
                Card::new(Suit::Hearts, 13),
                Card::new(Suit::Hearts, 11),
            ];
            let (result, cards): (bool, Vec<Card>) = HandsCheck::is_one_pair(&cards);

            assert!(result);
            assert_eq!(2, cards.len());
        }

        #[test]
        fn test_one_pair_not_ok() {
            let cards = vec![
                Card::new(Suit::Hearts, 10),
                Card::new(Suit::Spades, 7),
                Card::new(Suit::Hearts, 8),
                Card::new(Suit::Clubs, 6),
                Card::new(Suit::Hearts, 13),
                Card::new(Suit::Hearts, 11),
            ];
            let (result, cards): (bool, Vec<Card>) = HandsCheck::is_one_pair(&cards);

            assert!(!result);
            assert_eq!(0, cards.len());
        }
    }
}
