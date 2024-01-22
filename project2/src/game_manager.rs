use crate::{card::{Card, Suit}, oracle};
use rand::seq::SliceRandom;
use rand::thread_rng;
use crate::oracle::{Oracle, Deck};

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
    deck: Deck,
    flop: (Card, Card, Card),
    turn: Card,
    river: Card,
}

impl GameManager {
    pub fn new(num_players: u32) -> GameManager {
        // Generate a stack of cards and shuffle them
        let mut deck = oracle::Deck::new();
        // stack.shuffle(&mut rng);

        // Error if there are not enough cards for the game
        let empty_stack_error = "Not enough cards in stack";

        // Deal cards to players
        let mut players = Vec::with_capacity(num_players as usize);

        for _ in 0..num_players {
            let first_card = deck.pop().expect(empty_stack_error);
            let second_card = deck.pop().expect(empty_stack_error);

            players.push(Player::new((first_card, second_card)));
        }

        // Deal flop, turn, and river
        let flop = (
            deck.pop().expect(empty_stack_error),
            deck.pop().expect(empty_stack_error),
            deck.pop().expect(empty_stack_error),
        );
        let turn = deck.pop().expect(empty_stack_error);
        let river = deck.pop().expect(empty_stack_error);

        GameManager {
            players,
            deck,
            flop,
            turn,
            river,
        }
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
        let mut deck: Deck = oracle::Deck::new();
        deck.reset_stack();
        assert_eq!(deck.stack.len(), 52);
    }

    #[test]
    fn test_new() {
        let num_players = 2;
        let game_manager = GameManager::new(num_players);
        assert_eq!(game_manager.players.len(), num_players as usize);
        assert_eq!(
            game_manager.deck.stack.len(),
            52 - 2 * num_players as usize - 3 - 1 - 1 // 52 cards - 2 cards per player - flop - turn - river
        );
    }
}
