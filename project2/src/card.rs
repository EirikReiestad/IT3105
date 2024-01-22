#[derive(Clone, PartialEq)]
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
            Suit::CLUBS => "♣",
            Suit::DIAMONDS => "♦",
            Suit::HEARTS => "♥",
            Suit::SPADES => "♠",
        };
        write!(f, "{}{}", rank, suit)
    }
}

#[derive(Clone, Copy, PartialEq, PartialOrd, Eq, Hash)]
pub enum Suit {
    CLUBS,
    DIAMONDS,
    HEARTS,
    SPADES,
}
