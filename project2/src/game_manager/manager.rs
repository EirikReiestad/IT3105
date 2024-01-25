//! This module contains the game manager which handles the game logic
use super::players::{Player, Players};
use crate::card::Card;
use crate::poker_oracle::deck::Deck;
use rand::{thread_rng, Rng};
use std::io;

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

struct Board {
    flop: (Card, Card, Card),
    turn: Card,
    river: Card,
    pot: u32,
    highest_bet: u32,
    dealer: usize,
}

pub struct GameManager {
    players: Players,
    game_stage: GameStage,
    board: Board,
    buy_in: u32,
}

/// The game manager handles the game logic
/// It is responsible for:
/// - Generating the deck
/// - Shuffling the deck
/// - Dealing cards to players
/// - Dealing flop, turn, and river
/// - Determining dealer
/// - Running the game
impl GameManager {
    pub fn new(num_players: u32) -> GameManager {
        // Generate a stack of cards and shuffle them
        let mut deck = Deck::new();
        deck.reset_stack();
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

        let players = Players { players };

        // Deal flop, turn, and river
        let flop = (
            deck.pop().expect(empty_stack_error),
            deck.pop().expect(empty_stack_error),
            deck.pop().expect(empty_stack_error),
        );
        let turn = deck.pop().expect(empty_stack_error);
        let river = deck.pop().expect(empty_stack_error);

        // Determine dealer
        let mut rng = thread_rng();
        let dealer = rng.gen_range(0..num_players as usize);

        let board = Board {
            flop,
            turn,
            river,
            pot: 0,
            highest_bet: 0,
            dealer,
        };

        GameManager {
            players,
            game_stage: GameStage::PreFlop,
            board,
            buy_in: 10,
        }
    }

    /// Runs the game
    /// The game is run in a loop until a player wins
    /// The game is run in stages
    /// - PreFlop
    /// - Flop
    /// - Turn
    /// - River
    pub fn run(&mut self) {
        loop {
            match self.game_stage {
                GameStage::PreFlop => {
                    if let Some(winner) = self.run_game_stage() {
                        println!("Player {} won {} units!", winner, self.board.pot);
                        break;
                    }
                    self.game_stage = GameStage::Flop
                }
                GameStage::Flop => {
                    self.run_game_stage();
                    self.game_stage = GameStage::Turn
                }
                GameStage::Turn => {
                    self.run_game_stage();
                    self.game_stage = GameStage::River
                }
                GameStage::River => {
                    self.run_game_stage();
                    break;
                }
            }
        }
    }

    /// Runs a game stage
    fn run_game_stage(&mut self) -> Option<usize> {
        let mut check_count = 0;

        println!("{}", self);

        while check_count != self.players.get_number_of_active_players() {
            self.roate_dealer();
            for i in 0..self.players.len() {
                let turn = (self.board.dealer + i + 1) % self.players.len();

                if self.players.has_folded(turn) {
                    continue;
                }

                if self.players.get_number_of_active_players() == 1 {
                    return Some(turn);
                }

                println!(" ");
                println!("--- Player {}'s turn ---", turn);
                println!("{}", self.players.players[turn]);

                if self.game_stage == GameStage::PreFlop {
                    if let Some(x) = self.preflop_bets(turn) {
                        check_count += x;
                        continue;
                    }
                }

                let action = self.get_action();

                match action {
                    Action::Fold => {
                        self.players.fold(turn);
                        continue;
                    }
                    Action::CallOrCheck => {
                        let player_bet = self.players.get_bet(turn);
                        let bet = self.board.highest_bet - player_bet;
                        self.make_bet(turn, bet);
                        if bet == 0 {
                            println!("Checked");
                        } else {
                            println!("Called {}", bet);
                        }
                        check_count += 1;
                    }
                    Action::Raise(x) => {
                        let player_bet = self.players.get_bet(turn);
                        let raise = self.board.highest_bet - player_bet + x;
                        self.make_bet(turn, raise);
                        println!("Raised {}", raise);
                        check_count = 1;
                    }
                    _ => panic!("Invalid action"),
                }

                // If everyone has checked, break
                if check_count == self.players.get_number_of_active_players() {
                    break;
                }
            }
        }
        None
    }

    /// Handles the small and big blind
    /// Handles automatic betting for the small and big blind
    /// Assuming they can not fold.
    /// Returns 1 if the player is the big blind, 0 otherwise
    fn preflop_bets(&mut self, turn: usize) -> Option<usize> {
        let player_bet = self.players.get_bet(turn);
        let small_blind = (self.board.dealer + 1) % self.players.len();
        let big_blind = (self.board.dealer + 2) % self.players.len();

        println!("turn {} player_bet {}", turn, player_bet);

        if turn == small_blind && self.board.highest_bet == 0 {
            // Small blind
            println!(
                "Player {} is the small blind and must bet {}",
                turn,
                self.buy_in / 2
            );
            self.make_bet(turn, self.buy_in / 2);
            Some(0)
        } else if turn == big_blind && self.board.highest_bet == self.buy_in / 2 {
            // Big blind
            println!(
                "Player {} is the big blind and must bet {}",
                turn, self.buy_in
            );
            self.make_bet(turn, self.buy_in);
            Some(1)
        } else {
            None
        }
    }

    /// Handles making a bet
    /// If the player does not have enough money, they are forced to fold
    /// If the player has enough money, the bet is made
    /// The bet is added to the pot
    /// The bet is added to the player_bets HashMap
    /// The highest_bet is updated
    fn make_bet(&mut self, player: usize, bet: u32) {
        if self.players.place_bet(player, bet).is_err() {
            println!("Not enough money. Folded."); // TODO: Do not need to fold. Can go all-in or
                                                   // check if possible
            self.players.fold(player);
        } else {
            let player_bet = self.players.get_bet(player);
            self.board.pot += bet;
            println!("Player bet: {}, bet {}", player_bet, bet);
            self.board.highest_bet = player_bet;
        };
    }

    fn roate_dealer(&mut self) {
        self.board.dealer = (self.board.dealer + 1) % self.players.len();
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
                        "2" => Action::Raise(10), // TODO: Implement raise amount
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
            writeln!(
                f,
                "Flop: {} {} {}",
                self.board.flop.0, self.board.flop.1, self.board.flop.2
            )?;
        }
        if [GameStage::Turn, GameStage::River].contains(&self.game_stage) {
            writeln!(f, "Turn: {}", self.board.turn)?;
        }
        if [GameStage::River].contains(&self.game_stage) {
            writeln!(f, "River: {}", self.board.river)?;
        }
        writeln!(f, " ")?;
        writeln!(f, "===== PLAYERS =====")?;
        for (i, player) in self.players.iter().enumerate() {
            if i == self.board.dealer {
                writeln!(f, "Player {} (Dealer): {}", i, player)?;
            } else if i == (self.board.dealer + 1) % self.players.len() {
                writeln!(f, "Player {} (Small Blind): {}", i, player)?;
            } else if i == (self.board.dealer + 2) % self.players.len() {
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
    fn test_new() {
        let num_players = 2;
        let game_manager = GameManager::new(num_players);
        assert_eq!(game_manager.players.len(), num_players as usize);
    }
}
