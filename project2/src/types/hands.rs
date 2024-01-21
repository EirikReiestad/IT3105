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
        let mut unique_ranks = HashSet::new();
        let unique_cards: Vec<_> = cards
            .into_iter()
            .filter(|card| unique_ranks.insert(card.rank))
            .collect();
        let target_ranks: HashSet<i32> = [10, 11, 12, 13, 14].iter().cloned().collect();
        if unique_ranks == target_ranks {
            return 
        }

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
