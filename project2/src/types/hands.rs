#[derive(PartialEq, PartialOrd)]
enum Hands {
    ROYAL_FLUSH,
    STRAIGHT_FLUSH,
    FOUR_OF_A_KIND,
    FULL_HOUSE,
    FLUSH,
    STRAIGHT,
    THREE_OF_A_KIND,
    TWO_PAIRS,
    ONE_PAIR,
    HIGH_CARD,
}

pub struct HandsCheck {}

impl HandsCheck {
    pub fn is_royal_flush(&mut self, cards: Vec<Card>) -> (Hands, Vec<Card>) {
        let cards = cards.clone();
        cards.sort_by(|a, b| b.rank.cmp(a.rank));

        
    }
    pub fn is_straight_flush(&mut self, cards: Vec<Card>) -> (Hands, Vec<Card>) {
        let cards = cards.clone();
        cards.sort_by(|a, b| b.rank.cmp(a.rank))
    }
    pub fn is_four_of_a_kind(&mut self, cards: Vec<Card>) -> (Hands, Vec<Card>) {
        let cards = cards.clone();
        cards.sort_by(|a, b| b.rank.cmp(a.rank));
    }
    pub fn is_full_house(&mut self, cards: Vec<Card>) -> (Hands, Vec<Card>) {
        let cards = cards.clone();
        cards.sort_by(|a, b| b.rank.cmp(a.rank));
    }
    pub fn is_flush(&mut self, cards: Vec<Card>) -> (Hands, Vec<Card>) {
        let cards = cards.clone();
        cards.sort_by(|a, b| b.rank.cmp(a.rank));
    }
    pub fn is_straight(&mut self, cards: Vec<Card>) -> (Hands, Vec<Card>) {
        let cards = cards.clone();
        cards.sort_by(|a, b| b.rank.cmp(a.rank));
    }
    pub fn is_three_of_a_kind(&mut self, cards: Vec<Card>) -> (Hands, Vec<Card>) {
        let cards = cards.clone();
        cards.sort_by(|a, b| b.rank.cmp(a.rank));
    }
    pub fn is_two_pairs(&mut self, cards: Vec<Card>) -> (Hands, Vec<Card>) {
        let cards = cards.clone();
        cards.sort_by(|a, b| b.rank.cmp(a.rank));
    }
    pub fn is_one_pair(&mut self, cards: Vec<Card>) -> (Hands, Vec<Card>) {
        let cards = cards.clone();
        cards.sort_by(|a, b| b.rank.cmp(a.rank));
    }
    pub fn is_high_card(&mut self, cards: Vec<Card>) -> (Hands, Vec<Card>) {
        let cards = cards.clone();
        cards.sort_by(|a, b| b.rank.cmp(a.rank));
    }
}
