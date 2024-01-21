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

    pub fn bet(&mut self, amount: u32) -> Result<(), String> {
        if self.value < amount {
            return Err("Not enough money".to_string());
        }
        self.value -= amount;
        Ok(())
    }
}

impl std::fmt::Display for Player {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Cards: {}, {} ", self.cards.0, self.cards.1)?;
        write!(f, "Value: {}", self.value)
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

#[derive(Debug)]
enum Action {
    Fold,
    CallOrCheck, // This is used because the actions are the same for both (kinda)
    _Call(u32),
    _Check,
    Raise(u32),
    _AllIn(u32),
}

pub struct GameManager {
    players: Vec<Player>,
    minimum_bet: u32,
    flop: (Card, Card, Card),
    turn: Card,
    river: Card,
    game_stage: GameStage,
    dealer: usize,
    folded: HashMap<usize, bool>,
    pot: u32,
    highest_bet: u32,
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

        GameManager {
            players,
            minimum_bet: 10,
            // Hand info (round)
            flop,
            turn,
            river,
            game_stage: GameStage::PreFlop,
            dealer,
            folded: HashMap::new(),
            pot: 0,
            highest_bet: 0,
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
        loop {
            match self.game_stage {
                GameStage::PreFlop => {
                    self.run_preflop();
                    self.game_stage = GameStage::Flop
                }
                GameStage::Flop => {
                    self.run_flop();
                    self.game_stage = GameStage::Turn
                }
                GameStage::Turn => {
                    self.run_turn();
                    self.game_stage = GameStage::River
                }
                GameStage::River => self.run_river(),
            }
        }
    }

    fn run_preflop(&mut self) {
        let mut bets: HashMap<usize, u32> = HashMap::new();
        let mut check_count = 0;

        println!("{}", self);

        while check_count != self.players.len() - self.folded.len() {
            for i in 0..self.players.len() {
                let turn = (self.dealer + i + 1) % self.players.len();

                if self.folded.contains_key(&turn) {
                    continue;
                }

                let player_bet = bets.get(&turn).unwrap_or(&0);

                println!(" ");
                println!("--- Player {}'s turn ---", turn);
                println!("{}", self.players[turn]);

                if i == 0 && player_bet == &0 {
                    // Small blind
                    println!(
                        "Player {} is the small blind and must bet {}",
                        turn, self.minimum_bet
                    );
                    self.make_bet(turn, self.minimum_bet);
                } else if i == 1 && player_bet == &0 {
                    // Big blind
                    println!(
                        "Player {} is the big blind and must bet {}",
                        turn,
                        self.minimum_bet * 2
                    );
                    self.make_bet(turn, self.minimum_bet * 2);
                    check_count += 1;
                } else {
                    let action = self.get_action();
                    println!("Action: {:?}", action);
                    match action {
                        Action::Fold => {
                            self.folded.insert(turn, true);
                            continue;
                        }
                        Action::CallOrCheck => {
                            let bet = self.highest_bet - player_bet;
                            self.make_bet(turn, bet);
                            check_count += 1;
                        }
                        Action::Raise(x) => {
                            let raise = self.highest_bet - player_bet + x;
                            self.make_bet(turn, raise);
                            check_count = 1;
                        }
                        _ => panic!("Invalid action"),
                    }
                }
                bets.insert(turn, self.highest_bet); // Set the bet to the highest bet, either it have
                                                     // Called, Checked, or Raised which means it is the highest bet. Or it has folded
                                                     // which means it does not matter.
                                                     // Or small or big blind, which is the highest bet

                println!("{}", self.highest_bet);
                println!("{}", self.players.len() - self.folded.len() - check_count);
                // If everyone has checked, break
                if check_count == self.players.len() - self.folded.len() {
                    break;
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

    fn make_bet(&mut self, player: usize, bet: u32) {
        if self.players[player].bet(bet).is_err() {
            println!("Not enough money. Folded.");
            self.folded.insert(player, true);
        } else {
            self.pot += bet;
            self.highest_bet = bet;
        };
    }

    fn get_action(&mut self) -> Action {
        let options = "Fold(0) Call/Check(1) Raise(2)"; // TODO: Implement all-in
        println!("Options: {}", options);

        fn get_input() -> Action {
            let mut user_input = String::new();
            match io::stdin().read_line(&mut user_input) {
                Ok(_) => {
                    let trimmed_input = user_input.trim();

                    match trimmed_input {
                        "0" => Action::Fold,
                        "1" => Action::CallOrCheck,
                        "2" => Action::Raise(50), // TODO: Implement raise amount
                        _ => {
                            println!("Invalid input");
                            get_input()
                        }
                    }
                }
                Err(error) => {
                    println!("error: {}", error);
                    get_input()
                }
            }
        }

        get_input()
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
