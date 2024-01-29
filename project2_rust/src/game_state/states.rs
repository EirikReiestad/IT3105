use crate::card::Card;
use crate::game_manager::GameStage;
pub struct PlayerState {
    pub chips: u32,
    pub folded: bool,
    pub bet: u32,
}
impl PlayerState {
    pub fn new(chips: u32, folded: bool, bet: u32) -> Self {
        Self { chips, folded, bet }
    }
}

pub struct BoardState {
    pub cards: Vec<Card>,
    pub pot: u32,
    pub highest_bet: u32,
    pub dealer: usize,
}

impl BoardState {
    pub fn new(cards: Vec<Card>, pot: u32, highest_bet: u32, dealer: usize) -> Self {
        Self {
            cards,
            pot,
            highest_bet,
            dealer,
        }
    }
}

pub struct GameState {
    player_states: Vec<PlayerState>,
    board_state: BoardState,
    game_stage: GameStage,
}

impl GameState {
    // TODO: Implement player bid history
    pub fn new(
        player_states: Vec<PlayerState>,
        board_state: BoardState,
        game_stage: GameStage,
    ) -> Self {
        Self {
            player_states,
            board_state,
            game_stage,
        }
    }
}
