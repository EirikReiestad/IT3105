use crate::card::{Card, Suit};
use itertools::Itertools;
use std::collections::HashMap;
use std::collections::HashSet;

#[derive(PartialEq, PartialOrd)]
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
    None,
}

pub struct HandsCheck {}

impl HandsCheck {
    pub fn is_royal_flush(cards: &Vec<Card>) -> (bool, Vec<Card>) {
        let cards = cards.clone();
        let combinations: Vec<Vec<Card>> = cards.into_iter().combinations(5).collect();
        let target_ranks: HashSet<usize> = [1, 10, 11, 12, 13].iter().cloned().collect();

        for combination in combinations {
            let mut unique_suits = HashMap::new();
            for card in combination {
                unique_suits
                    .entry(card.suit)
                    .or_insert(Vec::new())
                    .push(card.rank);
            }
            for (suit, ranks) in unique_suits {
                let unique_ranks: HashSet<_> = ranks.into_iter().collect();

                if unique_ranks == target_ranks {
                    let new_cards: Vec<Card> = unique_ranks
                        .into_iter()
                        .map(|rank| Card {
                            suit: suit,
                            rank: rank,
                        })
                        .collect();
                    return (true, new_cards);
                }
            }
        }
        (false, vec![])
    }

    pub fn is_straight_flush(cards: &Vec<Card>) -> (bool, Vec<Card>) {
        let cards = cards.clone();
        let (result, new_cards) = HandsCheck::is_royal_flush(&cards);

        let combinations: Vec<Vec<Card>> = cards.into_iter().combinations(5).collect();

        if result {
            return (result, new_cards);
        }

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

            if unique_suits.len() == 1 && ranks_vec.len() == 5 && is_one_apart {
                let new_cards: Vec<Card> = ranks_vec
                    .into_iter()
                    .map(|rank: usize| Card {
                        suit: first_suit,
                        rank: rank,
                    })
                    .collect();
                return (true, new_cards);
            }
        }
        (false, vec![])
    }

    pub fn is_four_of_a_kind(cards: &Vec<Card>) -> (bool, Vec<Card>) {
        let cards = cards.clone();
        let combinations: Vec<Vec<Card>> = cards.into_iter().combinations(5).collect();

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
                        .map(|suit| Card {
                            suit: suit,
                            rank: rank,
                        })
                        .collect();
                    return (true, new_cards);
                }
            }
        }
        (false, vec![])
    }

    pub fn is_full_house(cards: &Vec<Card>) -> (bool, Vec<Card>) {
        let cards = cards.clone();
        for combination in cards.iter().combinations(5) {
            let mut rank_counts: HashMap<usize, Vec<Suit>> = HashMap::new();

            for card in &combination {
                rank_counts
                    .entry(card.rank)
                    .or_insert_with(Vec::new)
                    .push(card.suit);
            }

            let mut cards_len_2: Vec<Card> = Vec::new();
            let mut cards_len_3: Vec<Card> = Vec::new();

            for (rank, suits) in rank_counts.iter() {
                match suits.len() {
                    2 => cards_len_2
                        .extend(suits.iter().cloned().map(|suit| Card { rank: *rank, suit })),
                    3 => cards_len_3
                        .extend(suits.iter().cloned().map(|suit| Card { rank: *rank, suit })),
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
        let cards = cards.clone();
        let (result, new_cards) = HandsCheck::is_royal_flush(&cards);

        let combinations: Vec<Vec<Card>> = cards.into_iter().combinations(5).collect();

        if result {
            return (result, new_cards);
        }

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
                    .map(|rank: usize| Card {
                        suit: first_suit,
                        rank: rank,
                    })
                    .collect();
                return (true, new_cards);
            }
        }
        (false, vec![])
    }

    pub fn is_straight(cards: &Vec<Card>) -> (bool, Vec<Card>) {
        let cards = cards.clone();
        let (result, new_cards) = HandsCheck::is_royal_flush(&cards);

        let combinations: Vec<Vec<Card>> = cards.into_iter().combinations(5).collect();
        let target_ranks: HashSet<usize> = [1, 10, 11, 12, 13].iter().cloned().collect();
        if result {
            return (result, new_cards);
        }

        let original_combinations = combinations.clone();

        for combination in combinations {
            let mut unique_suits = HashSet::new();
            let mut unique_ranks = HashSet::new();

            for card in combination.clone() {
                unique_suits.insert(card.suit);
                unique_ranks.insert(card.rank);
            }
            let royal = unique_ranks == target_ranks;
            let mut ranks_vec: Vec<usize> = unique_ranks.into_iter().collect();
            ranks_vec.sort();

            let is_one_apart = ranks_vec
                .windows(2)
                .all(|window| (window[1] - window[0]) == 1);

            if unique_suits.len() != 1 && ranks_vec.len() == 5 && (is_one_apart || royal) {
                return (true, combination);
            }
        }
        (false, vec![])
    }

    pub fn is_three_of_a_kind(cards: &Vec<Card>) -> (bool, Vec<Card>) {
        let cards = cards.clone();
        let combinations: Vec<Vec<Card>> = cards.into_iter().combinations(5).collect();

        for combination in combinations {
            let mut unique_ranks = HashMap::new();
            for card in combination {
                unique_ranks
                    .entry(card.rank)
                    .or_insert(Vec::new())
                    .push(card.suit);
            }
            for (rank, suits) in unique_ranks.clone() {
                let unique_suits: HashSet<_> = suits.into_iter().collect();
                if unique_suits.len() == 3 && unique_ranks.len() > 2 {
                    let new_cards: Vec<Card> = unique_suits
                        .into_iter()
                        .map(|suit| Card {
                            suit: suit,
                            rank: rank,
                        })
                        .collect();
                    return (true, new_cards);
                }
            }
        }
        (false, vec![])
    }
    pub fn is_two_pairs(cards: &Vec<Card>) -> (bool, Vec<Card>) {
        let combinations: Vec<Vec<Card>> = cards.into_iter().cloned().combinations(5).collect();
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
                    .flat_map(|(rank, suits)| {
                        suits.iter().map(move |suit| Card {
                            suit: *suit,
                            rank: *rank,
                        })
                    })
                    .collect();

                return (true, new_cards);
            }
        }
        (false, vec![])
    }

    pub fn is_one_pair(cards: &Vec<Card>) -> (bool, Vec<Card>) {
        let combinations: Vec<Vec<Card>> = cards.into_iter().cloned().combinations(5).collect();
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
                        .map(|suit| Card {
                            suit: *suit,
                            rank: *first_key,
                        })
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

    #[test]
    fn test_royal_flush_OK() {
        let mut cards = vec![
            Card {
                suit: Suit::Spades,
                rank: 10,
            },
            Card {
                suit: Suit::Hearts,
                rank: 1,
            },
            Card {
                suit: Suit::Spades,
                rank: 11,
            },
            Card {
                suit: Suit::Spades,
                rank: 12,
            },
            Card {
                suit: Suit::Spades,
                rank: 13,
            },
            Card {
                suit: Suit::Spades,
                rank: 1,
            },
            Card {
                suit: Suit::Hearts,
                rank: 10,
            },
        ];
        let (result, mut cards): (bool, Vec<Card>) = HandsCheck::is_royal_flush(&cards);

        cards.sort_by(|a, b| a.rank.cmp(&b.rank));

        assert_eq!(result, true);
        assert_eq!(
            cards,
            vec![
                Card {
                    suit: Suit::Spades,
                    rank: 1,
                },
                Card {
                    suit: Suit::Spades,
                    rank: 10,
                },
                Card {
                    suit: Suit::Spades,
                    rank: 11,
                },
                Card {
                    suit: Suit::Spades,
                    rank: 12,
                },
                Card {
                    suit: Suit::Spades,
                    rank: 13,
                }
            ]
        )
    }

    #[test]
    fn test_royal_flush_OK_2() {
        let mut cards = vec![
            Card {
                suit: Suit::Hearts,
                rank: 10,
            },
            Card {
                suit: Suit::Hearts,
                rank: 1,
            },
            Card {
                suit: Suit::Spades,
                rank: 11,
            },
            Card {
                suit: Suit::Spades,
                rank: 12,
            },
            Card {
                suit: Suit::Spades,
                rank: 13,
            },
            Card {
                suit: Suit::Spades,
                rank: 1,
            },
            Card {
                suit: Suit::Spades,
                rank: 10,
            },
        ];
        let (result, mut cards): (bool, Vec<Card>) = HandsCheck::is_royal_flush(&cards);

        cards.sort_by(|a, b| a.rank.cmp(&b.rank));

        assert_eq!(result, true);
        assert_eq!(
            cards,
            vec![
                Card {
                    suit: Suit::Spades,
                    rank: 1,
                },
                Card {
                    suit: Suit::Spades,
                    rank: 10,
                },
                Card {
                    suit: Suit::Spades,
                    rank: 11,
                },
                Card {
                    suit: Suit::Spades,
                    rank: 12,
                },
                Card {
                    suit: Suit::Spades,
                    rank: 13,
                }
            ]
        );
    }

    #[test]
    fn test_royal_flush_NOT_OK() {
        let mut cards = vec![
            Card {
                suit: Suit::Hearts,
                rank: 10,
            },
            Card {
                suit: Suit::Hearts,
                rank: 1,
            },
            Card {
                suit: Suit::Spades,
                rank: 11,
            },
            Card {
                suit: Suit::Spades,
                rank: 8,
            },
            Card {
                suit: Suit::Spades,
                rank: 13,
            },
            Card {
                suit: Suit::Spades,
                rank: 1,
            },
            Card {
                suit: Suit::Spades,
                rank: 10,
            },
        ];
        let (result, mut cards): (bool, Vec<Card>) = HandsCheck::is_royal_flush(&cards);

        cards.sort_by(|a, b| a.rank.cmp(&b.rank));

        assert_eq!(result, false);
        assert_eq!(cards, vec![]);
    }

    #[test]
    fn test_straight_flush_OK() {
        let mut cards = vec![
            Card {
                suit: Suit::Spades,
                rank: 10,
            },
            Card {
                suit: Suit::Hearts,
                rank: 1,
            },
            Card {
                suit: Suit::Spades,
                rank: 11,
            },
            Card {
                suit: Suit::Spades,
                rank: 12,
            },
            Card {
                suit: Suit::Spades,
                rank: 13,
            },
            Card {
                suit: Suit::Spades,
                rank: 1,
            },
            Card {
                suit: Suit::Hearts,
                rank: 10,
            },
        ];
        let (result, mut cards): (bool, Vec<Card>) = HandsCheck::is_straight_flush(&cards);

        cards.sort_by(|a, b| a.rank.cmp(&b.rank));

        assert_eq!(result, true);
        assert_eq!(
            cards,
            vec![
                Card {
                    suit: Suit::Spades,
                    rank: 1,
                },
                Card {
                    suit: Suit::Spades,
                    rank: 10,
                },
                Card {
                    suit: Suit::Spades,
                    rank: 11,
                },
                Card {
                    suit: Suit::Spades,
                    rank: 12,
                },
                Card {
                    suit: Suit::Spades,
                    rank: 13,
                }
            ]
        )
    }

    #[test]
    fn test_straight_flush_OK_2() {
        let mut cards = vec![
            Card {
                suit: Suit::Spades,
                rank: 2,
            },
            Card {
                suit: Suit::Hearts,
                rank: 1,
            },
            Card {
                suit: Suit::Spades,
                rank: 3,
            },
            Card {
                suit: Suit::Spades,
                rank: 4,
            },
            Card {
                suit: Suit::Spades,
                rank: 5,
            },
            Card {
                suit: Suit::Spades,
                rank: 6,
            },
            Card {
                suit: Suit::Hearts,
                rank: 10,
            },
        ];
        let (result, mut cards): (bool, Vec<Card>) = HandsCheck::is_straight_flush(&cards);

        cards.sort_by(|a, b| a.rank.cmp(&b.rank));

        assert_eq!(result, true);
        assert_eq!(
            cards,
            vec![
                Card {
                    suit: Suit::Spades,
                    rank: 2,
                },
                Card {
                    suit: Suit::Spades,
                    rank: 3,
                },
                Card {
                    suit: Suit::Spades,
                    rank: 4,
                },
                Card {
                    suit: Suit::Spades,
                    rank: 5,
                },
                Card {
                    suit: Suit::Spades,
                    rank: 6,
                }
            ]
        )
    }

    #[test]
    fn test_straight_flush_NOT_OK() {
        let mut cards = vec![
            Card {
                suit: Suit::Spades,
                rank: 2,
            },
            Card {
                suit: Suit::Hearts,
                rank: 1,
            },
            Card {
                suit: Suit::Spades,
                rank: 3,
            },
            Card {
                suit: Suit::Hearts,
                rank: 8,
            },
            Card {
                suit: Suit::Spades,
                rank: 5,
            },
            Card {
                suit: Suit::Spades,
                rank: 6,
            },
            Card {
                suit: Suit::Hearts,
                rank: 10,
            },
        ];
        let (result, mut cards): (bool, Vec<Card>) = HandsCheck::is_straight_flush(&cards);

        cards.sort_by(|a, b| a.rank.cmp(&b.rank));

        assert_eq!(result, false);
        assert_eq!(cards, vec![])
    }

    #[test]
    fn test_straight_flush_NOT_OK_2() {
        let mut cards = vec![
            Card {
                suit: Suit::Spades,
                rank: 2,
            },
            Card {
                suit: Suit::Hearts,
                rank: 1,
            },
            Card {
                suit: Suit::Spades,
                rank: 3,
            },
            Card {
                suit: Suit::Hearts,
                rank: 4,
            },
            Card {
                suit: Suit::Spades,
                rank: 5,
            },
            Card {
                suit: Suit::Spades,
                rank: 6,
            },
            Card {
                suit: Suit::Hearts,
                rank: 10,
            },
        ];
        let (result, mut cards): (bool, Vec<Card>) = HandsCheck::is_straight_flush(&cards);

        cards.sort_by(|a, b| a.rank.cmp(&b.rank));

        assert_eq!(result, false);
        assert_eq!(cards, vec![])
    }

    #[test]
    fn test_four_of_a_kind_OK() {
        let mut cards = vec![
            Card {
                suit: Suit::Hearts,
                rank: 10,
            },
            Card {
                suit: Suit::Hearts,
                rank: 1,
            },
            Card {
                suit: Suit::Spades,
                rank: 1,
            },
            Card {
                suit: Suit::Clubs,
                rank: 1,
            },
            Card {
                suit: Suit::Diamonds,
                rank: 1,
            },
            Card {
                suit: Suit::Spades,
                rank: 10,
            },
        ];
        let (result, mut cards): (bool, Vec<Card>) = HandsCheck::is_four_of_a_kind(&cards);

        cards.sort_by(|a, b| a.rank.cmp(&b.rank));

        assert_eq!(result, true);
        assert_eq!(cards.len(), 4);
    }

    #[test]
    fn test_four_of_a_kind_NOT_OK() {
        let mut cards = vec![
            Card {
                suit: Suit::Hearts,
                rank: 10,
            },
            Card {
                suit: Suit::Hearts,
                rank: 1,
            },
            Card {
                suit: Suit::Spades,
                rank: 2,
            },
            Card {
                suit: Suit::Clubs,
                rank: 1,
            },
            Card {
                suit: Suit::Diamonds,
                rank: 1,
            },
            Card {
                suit: Suit::Spades,
                rank: 10,
            },
        ];
        let (result, mut cards): (bool, Vec<Card>) = HandsCheck::is_four_of_a_kind(&cards);

        // cards.sort_by(|a, b| a.rank.cmp(&b.rank));

        assert_eq!(result, false);
        assert_eq!(cards, vec![]);
    }

    #[test]
    fn test_full_house_OK() {
        let mut cards = vec![
            Card {
                suit: Suit::Hearts,
                rank: 10,
            },
            Card {
                suit: Suit::Clubs,
                rank: 10,
            },
            Card {
                suit: Suit::Spades,
                rank: 1,
            },
            Card {
                suit: Suit::Clubs,
                rank: 1,
            },
            Card {
                suit: Suit::Diamonds,
                rank: 1,
            },
            Card {
                suit: Suit::Spades,
                rank: 11,
            },
        ];
        let (result, mut cards): (bool, Vec<Card>) = HandsCheck::is_full_house(&cards);

        cards.sort_by(|a, b| a.rank.cmp(&b.rank));

        assert_eq!(result, true);
        assert_eq!(
            cards,
            vec![
                Card {
                    suit: Suit::Spades,
                    rank: 1,
                },
                Card {
                    suit: Suit::Clubs,
                    rank: 1,
                },
                Card {
                    suit: Suit::Diamonds,
                    rank: 1,
                },
                Card {
                    suit: Suit::Hearts,
                    rank: 10,
                },
                Card {
                    suit: Suit::Clubs,
                    rank: 10,
                }
            ]
        );
    }

    #[test]
    fn test_full_house_NOT_OK() {
        let mut cards = vec![
            Card {
                suit: Suit::Hearts,
                rank: 10,
            },
            Card {
                suit: Suit::Clubs,
                rank: 9,
            },
            Card {
                suit: Suit::Spades,
                rank: 1,
            },
            Card {
                suit: Suit::Clubs,
                rank: 1,
            },
            Card {
                suit: Suit::Diamonds,
                rank: 1,
            },
            Card {
                suit: Suit::Spades,
                rank: 11,
            },
        ];
        let (result, mut cards): (bool, Vec<Card>) = HandsCheck::is_full_house(&cards);

        cards.sort_by(|a, b| a.rank.cmp(&b.rank));

        assert_eq!(result, false);
        assert_eq!(cards, vec![]);
    }

    #[test]
    fn test_flush_OK() {
        let mut cards = vec![
            Card {
                suit: Suit::Hearts,
                rank: 10,
            },
            Card {
                suit: Suit::Hearts,
                rank: 7,
            },
            Card {
                suit: Suit::Hearts,
                rank: 2,
            },
            Card {
                suit: Suit::Clubs,
                rank: 1,
            },
            Card {
                suit: Suit::Hearts,
                rank: 1,
            },
            Card {
                suit: Suit::Hearts,
                rank: 11,
            },
        ];
        let (result, mut cards): (bool, Vec<Card>) = HandsCheck::is_flush(&cards);

        cards.sort_by(|a, b| a.rank.cmp(&b.rank));

        assert_eq!(result, true);
        assert_eq!(
            cards,
            vec![
                Card {
                    suit: Suit::Hearts,
                    rank: 1,
                },
                Card {
                    suit: Suit::Hearts,
                    rank: 2,
                },
                Card {
                    suit: Suit::Hearts,
                    rank: 7,
                },
                Card {
                    suit: Suit::Hearts,
                    rank: 10,
                },
                Card {
                    suit: Suit::Hearts,
                    rank: 11,
                },
            ]
        );
    }

    #[test]
    fn test_flush_NOT_OK() {
        let mut cards = vec![
            Card {
                suit: Suit::Hearts,
                rank: 10,
            },
            Card {
                suit: Suit::Hearts,
                rank: 7,
            },
            Card {
                suit: Suit::Spades,
                rank: 2,
            },
            Card {
                suit: Suit::Clubs,
                rank: 1,
            },
            Card {
                suit: Suit::Hearts,
                rank: 1,
            },
            Card {
                suit: Suit::Hearts,
                rank: 11,
            },
        ];
        let (result, mut cards): (bool, Vec<Card>) = HandsCheck::is_flush(&cards);

        cards.sort_by(|a, b| a.rank.cmp(&b.rank));

        assert_eq!(result, false);
        assert_eq!(cards, vec![]);
    }

    #[test]
    fn test_straight_OK() {
        let mut cards = vec![
            Card {
                suit: Suit::Hearts,
                rank: 10,
            },
            Card {
                suit: Suit::Spades,
                rank: 7,
            },
            Card {
                suit: Suit::Hearts,
                rank: 8,
            },
            Card {
                suit: Suit::Clubs,
                rank: 9,
            },
            Card {
                suit: Suit::Hearts,
                rank: 13,
            },
            Card {
                suit: Suit::Hearts,
                rank: 11,
            },
        ];
        let (result, mut cards): (bool, Vec<Card>) = HandsCheck::is_straight(&cards);

        cards.sort_by(|a, b| a.rank.cmp(&b.rank));

        assert_eq!(result, true);
        assert_eq!(
            cards,
            vec![
                Card {
                    suit: Suit::Spades,
                    rank: 7,
                },
                Card {
                    suit: Suit::Hearts,
                    rank: 8,
                },
                Card {
                    suit: Suit::Clubs,
                    rank: 9,
                },
                Card {
                    suit: Suit::Hearts,
                    rank: 10,
                },
                Card {
                    suit: Suit::Hearts,
                    rank: 11,
                },
            ]
        );
    }

    #[test]
    fn test_straight_NOT_OK() {
        let mut cards = vec![
            Card {
                suit: Suit::Hearts,
                rank: 10,
            },
            Card {
                suit: Suit::Hearts,
                rank: 7,
            },
            Card {
                suit: Suit::Hearts,
                rank: 8,
            },
            Card {
                suit: Suit::Hearts,
                rank: 9,
            },
            Card {
                suit: Suit::Hearts,
                rank: 13,
            },
            Card {
                suit: Suit::Hearts,
                rank: 11,
            },
        ];
        let (result, mut cards): (bool, Vec<Card>) = HandsCheck::is_straight(&cards);

        cards.sort_by(|a, b| a.rank.cmp(&b.rank));

        assert_eq!(result, false);
        assert_eq!(cards, vec![]);
    }

    #[test]
    fn test_three_of_a_kind_OK() {
        let mut cards = vec![
            Card {
                suit: Suit::Hearts,
                rank: 10,
            },
            Card {
                suit: Suit::Spades,
                rank: 10,
            },
            Card {
                suit: Suit::Hearts,
                rank: 8,
            },
            Card {
                suit: Suit::Clubs,
                rank: 10,
            },
            Card {
                suit: Suit::Hearts,
                rank: 13,
            },
            Card {
                suit: Suit::Hearts,
                rank: 11,
            },
        ];
        let (result, mut cards): (bool, Vec<Card>) = HandsCheck::is_three_of_a_kind(&cards);

        assert_eq!(result, true);
        assert_eq!(3, cards.len());
    }

    #[test]
    fn test_three_of_a_kind_NOT_OK() {
        let mut cards = vec![
            Card {
                suit: Suit::Hearts,
                rank: 10,
            },
            Card {
                suit: Suit::Spades,
                rank: 10,
            },
            Card {
                suit: Suit::Clubs,
                rank: 13,
            },
            Card {
                suit: Suit::Clubs,
                rank: 10,
            },
            Card {
                suit: Suit::Hearts,
                rank: 13,
            },
            Card {
                suit: Suit::Spades,
                rank: 13,
            },
        ];
        let (result, mut cards): (bool, Vec<Card>) = HandsCheck::is_three_of_a_kind(&cards);

        assert_eq!(result, false);
        assert_eq!(0, cards.len());
    }

    #[test]
    fn test_two_pair_OK() {
        let mut cards = vec![
            Card {
                suit: Suit::Hearts,
                rank: 10,
            },
            Card {
                suit: Suit::Spades,
                rank: 10,
            },
            Card {
                suit: Suit::Hearts,
                rank: 8,
            },
            Card {
                suit: Suit::Clubs,
                rank: 8,
            },
            Card {
                suit: Suit::Hearts,
                rank: 13,
            },
            Card {
                suit: Suit::Hearts,
                rank: 11,
            },
        ];
        let (result, mut cards): (bool, Vec<Card>) = HandsCheck::is_two_pairs(&cards);

        assert_eq!(result, true);
        assert_eq!(4, cards.len());
    }

    #[test]
    fn test_two_pair_NOT_OK() {
        let mut cards = vec![
            Card {
                suit: Suit::Hearts,
                rank: 10,
            },
            Card {
                suit: Suit::Spades,
                rank: 10,
            },
            Card {
                suit: Suit::Hearts,
                rank: 7,
            },
            Card {
                suit: Suit::Clubs,
                rank: 8,
            },
            Card {
                suit: Suit::Hearts,
                rank: 13,
            },
            Card {
                suit: Suit::Hearts,
                rank: 11,
            },
        ];
        let (result, mut cards): (bool, Vec<Card>) = HandsCheck::is_two_pairs(&cards);

        assert_eq!(result, false);
        assert_eq!(0, cards.len());
    }


    #[test]
    fn test_one_pair_OK() {
        let mut cards = vec![
            Card {
                suit: Suit::Hearts,
                rank: 10,
            },
            Card {
                suit: Suit::Spades,
                rank: 10,
            },
            Card {
                suit: Suit::Hearts,
                rank: 8,
            },
            Card {
                suit: Suit::Clubs,
                rank: 8,
            },
            Card {
                suit: Suit::Hearts,
                rank: 13,
            },
            Card {
                suit: Suit::Hearts,
                rank: 11,
            },
        ];
        let (result, mut cards): (bool, Vec<Card>) = HandsCheck::is_one_pair(&cards);

        assert_eq!(result, true);
        assert_eq!(2, cards.len());
    }

    #[test]
    fn test_one_pair_NOT_OK() {
        let mut cards = vec![
            Card {
                suit: Suit::Hearts,
                rank: 10,
            },
            Card {
                suit: Suit::Spades,
                rank: 7,
            },
            Card {
                suit: Suit::Hearts,
                rank: 8,
            },
            Card {
                suit: Suit::Clubs,
                rank: 6,
            },
            Card {
                suit: Suit::Hearts,
                rank: 13,
            },
            Card {
                suit: Suit::Hearts,
                rank: 11,
            },
        ];
        let (result, mut cards): (bool, Vec<Card>) = HandsCheck::is_one_pair(&cards);

        assert_eq!(result, false);
        assert_eq!(0, cards.len());
    }
}
