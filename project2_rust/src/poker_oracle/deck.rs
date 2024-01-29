use crate::card::{Card, Suit};
use rand::prelude::SliceRandom;
use rand::thread_rng;

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
