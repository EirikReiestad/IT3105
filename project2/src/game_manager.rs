use crate::card::{Card, Suit};
use rand::seq::SliceRandom;
use rand::thread_rng;

pub struct Player {
    cards: (Card, Card),
}

impl Player {
    pub fn new(cards: (Card, Card)) -> Player {
        Player { cards }
    }
}

impl std::fmt::Display for Player {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Cards: {}, {}", self.cards.0, self.cards.1)
    }
}

pub struct GameManager {
    players: Vec<Player>,
    stack: Vec<Card>,
    flop: (Card, Card, Card),
    turn: Card,
    river: Card,
}

impl GameManager {
    pub fn new(num_players: u32) -> GameManager {
        // Generate a stack of cards and shuffle them
        let mut stack = GameManager::generate_stack();
        let mut rng = thread_rng();
        stack.shuffle(&mut rng);

        // Error if there are not enough cards for the game
        let empty_stack_error = "Not enough cards in stack";

        // Deal cards to players
        let mut players = Vec::with_capacity(num_players as usize);

        for _ in 0..num_players {
            let first_card = stack.pop().expect(empty_stack_error);
            let second_card = stack.pop().expect(empty_stack_error);

            players.push(Player::new((first_card, second_card)));
        }

        // Deal flop, turn, and river
        let flop = (
            stack.pop().expect(empty_stack_error),
            stack.pop().expect(empty_stack_error),
            stack.pop().expect(empty_stack_error),
        );
        let turn = stack.pop().expect(empty_stack_error);
        let river = stack.pop().expect(empty_stack_error);

        GameManager {
            players,
            stack,
            flop,
            turn,
            river,
        }
    }

    fn generate_stack() -> Vec<Card> {
        let mut stack = Vec::with_capacity(52);
        for (i, suit) in (0..4).enumerate() {
            let suit = match suit {
                0 => Suit::CLUBS,
                1 => Suit::DIAMONDS,
                2 => Suit::HEARTS,
                3 => Suit::SPADES,
                _ => panic!("Invalid suit"),
            };

            for rank in 0..13 {
                stack.push(Card { suit, rank });
            }
        }
        stack
    }

    pub fn run(&self) {
        println!("Game is running");
    }
}

impl std::fmt::Display for GameManager {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        writeln!(
            f,
            "=============================== GAME MANAGER ==============================="
        )?;
        writeln!(f, "Flop: {} {} {}", self.flop.0, self.flop.1, self.flop.2)?;
        writeln!(f, "Turn: {}", self.turn)?;
        writeln!(f, "River: {}", self.river)?;
        writeln!(f, "(1) Players:")?;
        for (i, player) in self.players.iter().enumerate() {
            write!(f, "Player {}: {}", i, player)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_stack() {
        let stack = GameManager::generate_stack();
        assert_eq!(stack.len(), 52);
    }

    #[test]
    fn test_new() {
        let num_players = 2;
        let game_manager = GameManager::new(num_players);
        assert_eq!(game_manager.players.len(), num_players as usize);
        assert_eq!(
            game_manager.stack.len(),
            52 - 2 * num_players as usize - 3 - 1 - 1 // 52 cards - 2 cards per player - flop - turn - river
        );
    }
}
