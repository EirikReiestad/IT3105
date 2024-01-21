use crate::card::{Card, Suit};
use rand::seq::SliceRandom;
use rand::{thread_rng, Rng};
use std::collections::HashMap;
use std::io;

pub struct Player {
    cards: (Card, Card),
    value: u32,
}

impl Player {
    pub fn new(cards: (Card, Card)) -> Player {
        let value = 100; // TODO: Implement hand value
        Player { cards, value }
    }

    pub fn bet(&mut self, amount: u32) {
        self.value -= amount;
    }
}

impl std::fmt::Display for Player {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Cards: {}, {}", self.cards.0, self.cards.1)
    }
}

#[derive(PartialEq)]
enum GameStage {
    PreFlop,
    Flop,
    Turn,
    River,
}

impl std::fmt::Display for GameStage {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let stage = match self {
            GameStage::PreFlop => "PreFlop",
            GameStage::Flop => "Flop",
            GameStage::Turn => "Turn",
            GameStage::River => "River",
        };
        write!(f, "{}", stage)
    }
}

struct Action {
    fold: bool,
    check: bool,
    call: bool,
    raise: bool,
    all_in: bool,
}

pub struct GameManager {
    players: Vec<Player>,
    flop: (Card, Card, Card),
    turn: Card,
    river: Card,
    game_stage: GameStage,
    dealer: usize,
    minimum_bet: u32,
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

        // Determine dealer
        let dealer = rng.gen_range(0..num_players as usize);

        // Determine minimum bet
        let minimum_bet = 10;

        GameManager {
            players,
            flop,
            turn,
            river,
            game_stage: GameStage::PreFlop,
            dealer,
            minimum_bet,
        }
    }

    fn generate_stack() -> Vec<Card> {
        let mut stack = Vec::with_capacity(52);
        for i in 0..4 {
            let suit = match i {
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

    pub fn run(&mut self) {
        match self.game_stage {
            GameStage::PreFlop => self.run_preflop(),
            GameStage::Flop => self.run_flop(),
            GameStage::Turn => self.run_turn(),
            GameStage::River => self.run_river(),
        }
    }

    fn run_preflop(&mut self) {
        let mut pot = 0;
        let mut folders: HashMap<usize, bool> = HashMap::new();
        let mut highest_bet = 0;
        let mut bets: HashMap<usize, u32> = HashMap::new();

        for i in 0..self.players.len() {
            let turn = (self.dealer + i + 1) % self.players.len();
            if folders.contains_key(&turn) {
                continue;
            }
            println!(" ");
            println!("--- Player {}'s turn ---", turn);

            let mut user_input = String::new();
            if i == 0 {
                println!(
                    "Player {} is the small blind and must bet {}",
                    turn, self.minimum_bet
                );
                pot += self.minimum_bet;
                self.players[turn].bet(self.minimum_bet);
            } else if i == 1 {
                // Big blind
                println!(
                    "Player {} is the big blind and must bet {}",
                    turn,
                    self.minimum_bet * 2
                );
                pot += self.minimum_bet;
                self.players[turn].bet(self.minimum_bet);
            } else {
                let player_bet = bets.get(&turn).unwrap_or(&0);
                let options = if player_bet < &highest_bet {
                    "Fold(0) Call(1) Raise(2) All In(3)"
                } else {
                    "Fold(0) Check(1) Raise(2) All In(3)"
                };
                println!("Options: {}", options);
                loop {
                    match io::stdin().read_line(&mut user_input) {
                        Ok(_) => {
                            let trimmed_input = user_input.trim();

                            match trimmed_input {
                                "0" => {}
                                    folders.insert(turn, true);
                                    println!("Player {} folds", turn);
                                    break;
                                }
                        _  => {}
                            }
                        }
                        Err(error) => println!("error: {}", error),
                    }
                }
            }
        }
    }

    fn run_flop(&self) {
        println!("Flop");
    }

    fn run_turn(&self) {
        println!("Turn");
    }

    fn run_river(&self) {
        println!("River");
    }
}

impl std::fmt::Display for GameManager {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        writeln!(
            f,
            "
============================================================================
=============================== GAME MANAGER ===============================
============================================================================
        "
        )?;
        writeln!(f, "===== {} =====", self.game_stage)?;

        if [GameStage::Flop, GameStage::Turn, GameStage::River].contains(&self.game_stage) {
            writeln!(f, "Flop: {} {} {}", self.flop.0, self.flop.1, self.flop.2)?;
        }
        if [GameStage::Turn, GameStage::River].contains(&self.game_stage) {
            writeln!(f, "Turn: {}", self.turn)?;
        }
        if [GameStage::River].contains(&self.game_stage) {
            writeln!(f, "River: {}", self.river)?;
        }
        writeln!(f, " ")?;
        writeln!(f, "===== PLAYERS =====")?;
        for (i, player) in self.players.iter().enumerate() {
            if i == self.dealer {
                writeln!(f, "Player {} (Dealer): {}", i, player)?;
            } else if i == (self.dealer + 1) % self.players.len() {
                writeln!(f, "Player {} (Small Blind): {}", i, player)?;
            } else if i == (self.dealer + 2) % self.players.len() {
                writeln!(f, "Player {} (Big Blind): {}", i, player)?;
            } else {
                writeln!(f, "Player {}: {}", i, player)?;
            }
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
    }
}
