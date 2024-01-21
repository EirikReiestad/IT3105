pub struct Card {
    pub suit: Suit,
    pub rank: usize,
}

impl std::fmt::Display for Card {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let rank = match self.rank {
            0 => "A",
            1..=9 => &self.rank.to_string(),
            10 => "J",
            11 => "Q",
            12 => "K",
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

pub enum Suit {
    CLUBS,
    DIAMONDS,
    HEARTS,
    SPADES,
}
