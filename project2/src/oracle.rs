mod oracle;

use types::Card;
use types::Hands;
use types::HandsCheck;

pub struct Oracle {}

impl Oracle {
    pub fn hand_classifier(&mut self, cards: Vec<Card>) -> Hands {
        if !(5 <= cards.len() <= 7) {
            return;
        }
        
        if HandsCheck.is_royal_flush(cards) {
            return Hands.RoyalFlush;
        } else if HandsCheck.is_straight_flush(cards) {
            return Hands.StraightFlush;
        } else if HandsCheck.is_four_of_a_kind(cards) {
            return Hands.FourOfAKind;
        } else if HandsCheck.is_full_house(cards) {
            return Hands.FullHouse;
        } else if HandsCheck.is_flush(cards) {
            return Hands.Flush;
        } else if HandsCheck.is_straight(cards) {
            return Hands.Straight;
        } else if HandsCheck.is_three_of_a_kind(cards) {
            return Hands.ThreeOfAKind;
        } else if HandsCheck.is_two_pairs(cards) {
            return Hands.TwoPairs;
        } else if HandsCheck.is_one_pair(cards) {
            return Hands.OnePair;
        } else {
            return Hands.HighCard;
        }
    }

    pub fn hand_evaluator(&mut self, set_one: Vec<Card>, set_two: Vec<Card>) -> bool {
        let result_one: Hands = self.hand_classifier(set_one);
        let result_two: Hands = self.hand_classifier(set_two);

        if result_one > result_two {
            true;
        } else if result_one < result_two{
            false;
        } else  {
            // TODO add tie breaker
        }
    }

    pub fn hole_pair_evaluator(
        &mut self,
        hole_pair: Vec<Card>,
        public_cards: Vec<Card>,
        num_opponents: usize,
        rollout_count: usize,
    ) -> f32 {
        let win_count = 0;
        for i in 0..=rollout_count {
            // CREATE HOLE FOR EACH OPPONENTS
            let mut win_all = 1;
            for j in 0..=num_opponents {
                // COMPARE, if win all -> THEN MEANING WE WIN
                // our card == hole_pair + public cards
                // opponent get cars that are left in the bunk
                // have to remove public cards from bunk too
                if !self.hand_evaluator() {
                    win_all = 0
                }
            }

        }
        probability = win_count/rollout_count
    }
    pub fn cheat_sheet_generator(&mut self, public_cards: Vec<Card>, num_opponents: usize) {
        // for all the pairs use the method below
        self.hole_pair_evaluator()
    }
    pub fn utility_matrix_generator(&mut self, public_cards: Vec<Card>) -> Vec<Vec<i32>> {
        // for each hole pair 
        self.hole_pair_evaluator()
    }
}
