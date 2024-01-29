use std::cmp::Ordering; #[derive(Clone, Debug, PartialEq, Eq)] pub struct Card { pub suit: Suit, pub rank: usize, } impl Card { pub fn new(suit: Suit, rank: usize) -> Card { Card { suit, rank } } } impl std::fmt::Display for Card { fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result { let rank = match self.rank {
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

