mod oracle;

use types::Card;
use types::Hands;
use types::HandsCheck;

pub struct Oracle {}

impl Oracle {
    pub fn hand_classifier(&mut self, cards: Vec<Card>) -> (Hands, Vec<Card>) {
        if !(5 <= cards.len() <= 7) {
            return;
        }
        
        if let Some(hand) = HandsCheck.is_royal_flush(cards)
            .or(HandsCheck.is_straight_flush(cards))
            .or(HandsCheck.is_four_of_a_kind(cards))
            .or(HandsCheck.is_full_house(cards))
            .or(HandsCheck.is_flush(cards))
            .or(HandsCheck.is_straight(cards))
            .or(HandsCheck.is_three_of_a_kind(cards))
            .or(HandsCheck.is_two_pairs(cards))
            .or(HandsCheck.is_one_pair(cards))
            .or(HandsCheck.is_high_card(cards))
        {
            return hand;
        }
    }
    pub fn hole_pair_evaluator(
        &mut self,
        hole_pair: Vec<u8>,
        public_cards: Vec<u8>,
        num_opponents: i32,
        rollout_count: i32,
    ) -> f32 {
    }
    pub fn cheat_sheet_generator(&mut self) {
        self.hole_pair_evaluator()
    }
    pub fn utility_matrix_generator(&mut self) -> Vec<Vec<i32>> {}
}
