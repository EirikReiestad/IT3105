pub struct Card {
    pub suit: Suit,
    pub rank: usize,
}

pub enum Suit {
    CLUBS,
    DIAMONDS,
    HEARTS,
    SPADES,
}
