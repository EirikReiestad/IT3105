use crate::card::Card;
use std::{collections::HashSet, intrinsics::mir::Return};

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
    pub fn is_royal_flush(&mut self, cards: Vec<Card>) -> bool {
        let combinations: Vec<Vec<_>> = cards.iter().combinations(5).collect();
        let target_ranks: HashSet<usize> = [10, 11, 12, 13, 14].iter().cloned().collect();

        for combination in combinations {
            let mut unique_ranks = HashSet::new();
            let unique_cards: Vec<_> = combination
                .into_iter()
                .filter(|card| unique_ranks.insert(card.rank))
                .collect();

            let first_suit = match cards.first() {
                Some(first_card) => first_card.suit,
                None => return,
            };

            let same_suit = cards.iter().all(|card| card.suit == first_suit);
            if unique_ranks == target_ranks && same_suit {
                return true;
            }
        }
        return false;
    }
}
