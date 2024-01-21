#[derive(Clone)]
pub struct Card {
    pub suit: Suit,
    pub rank: usize,
}

impl std::fmt::Display for Card {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let rank = match self.rank {
            0 => "A".to_string(),
            1..=9 => self.rank.to_string(),
            10 => "J".to_string(),
            11 => "Q".to_string(),
            12 => "K".to_string(),
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

#[derive(Clone, Copy, PartialEq, PartialOrd, Eq, Hash)]
pub enum Suit {
    Clubs,
    Diamonds,
    Hearts,
    Spades,
}

pub struct Deck;

impl Deck {
    pub fn generate_deck() -> Vec<Card> {
        let mut deck = Vec::with_capacity(52);
        for suit in [Suit::Clubs, Suit::Diamonds, Suit::Hearts, Suit::Spades].iter() {
            for rank in 0..13 {
                deck.push(Card { suit: *suit, rank });
            }
        }
        deck
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_stack() {
        let stack = Deck::generate_deck();
        assert_eq!(stack.len(), 52);
    }
}
