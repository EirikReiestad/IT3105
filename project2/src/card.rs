use rand::prelude::SliceRandom;
use rand::thread_rng;
use std::cmp::Ordering;


#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Card {
    pub suit: Suit,
    pub rank: usize,
}

impl std::fmt::Display for Card {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let rank = match self.rank {
            1 => "A".to_string(),
            2..=10 => self.rank.to_string(),
            11 => "J".to_string(),
            12 => "Q".to_string(),
            13 => "K".to_string(),
            _ => panic!("Invalid rank"),
        };
        let suit = match self.suit {
            Suit::Clubs => "♣",
            Suit::Diamonds => "♦",
            Suit::Hearts => "♥",
            Suit::Spades => "♠",
        };
        write!(f, "{}{}", rank, suit)
    }
}

impl Ord for Card {
    fn cmp(&self, other: &Self) -> Ordering {
        self.rank.cmp(&other.rank)
    }
}

impl PartialOrd for Card {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, Eq, Hash)]
pub enum Suit {
    Clubs,
    Diamonds,
    Hearts,
    Spades,
}


pub struct Deck {
    pub stack: Vec<Card>,
}

impl Deck {
    pub fn new() -> Self {
        Deck { stack: Vec::new() }
    }

    pub fn reset_stack(&mut self) {
        let mut stack = Vec::with_capacity(52);
        for (_, suit) in (0..4).enumerate() {
            let suit = match suit {
                0 => Suit::Clubs,
                1 => Suit::Diamonds,
                2 => Suit::Hearts,
                3 => Suit::Spades,
                _ => panic!("Invalid suit"),
            };

            for rank in 1..=13 {
                stack.push(Card { suit, rank });
            }
        }
        let mut rng = thread_rng();
        stack.shuffle(&mut rng);
        self.stack = stack;
    }

    pub fn remove(&mut self, card: &Card) -> () {
        self.stack.retain(|i| i != card);
    }

    pub fn pop(&mut self) -> Option<Card> {
        self.stack.pop()
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_stack() {
        let mut deck = Deck::new();
        deck.reset_stack();
        assert_eq!(deck.stack.len(), 52);
    }
}
