use crate::card::Card;

/// Keep track of the cards a player has
/// Keep track of the chips (money)
/// Keep track of if the player has folded
/// Keep track of how much the player has bet in the current round
pub struct Player {
    cards: (Card, Card),
    chips: u32,
    folded: bool,
    bet: u32,
}

impl Player {
    pub fn new(cards: (Card, Card)) -> Player {
        let chips = 100; // TODO: Implement hand value
        Player {
            cards,
            chips,
            folded: false,
            bet: 0,
        }
    }

    pub fn bet(&mut self, amount: u32) -> Result<(), String> {
        if self.chips < amount {
            return Err("Not enough chips".to_string());
        }
        self.chips -= amount;
        Ok(())
    }
}

impl std::fmt::Display for Player {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Cards: {}, {} ", self.cards.0, self.cards.1)?;
        write!(f, "Chips: {}", self.chips)
    }
}

/// The players struct contains a vector of players
/// Keep track of the dealer
/// Keep track of the highest bet
pub struct Players {
    pub players: Vec<Player>,
    // TODO: Impleemnt if go bust, remember that impacts the choice of the dealer
}

impl Players {
    pub fn len(&self) -> usize {
        self.players.len()
    }

    pub fn iter(&self) -> std::slice::Iter<Player> {
        self.players.iter()
    }
}

impl Players {
    fn get_number_of_folded(&self) -> usize {
        // Note: This is more expensive than a simple counter. However, the number of players is
        // small so this is not an issue, and it is more readable.
        self.players.iter().filter(|player| player.folded).count()
    }

    pub fn get_number_of_active_players(&self) -> usize {
        // Number of players - number of folded players
        self.players.len() - self.get_number_of_folded()
    }

    pub fn has_folded(&self, player: usize) -> bool {
        self.players[player].folded
    }

    pub fn fold(&mut self, player: usize) {
        self.players[player].folded = true;
    }

    pub fn get_bet(&self, player: usize) -> u32 {
        self.players[player].bet
    }

    pub fn place_bet(&mut self, player: usize, bet: u32) -> Result<(), String> {
        self.players[player].bet += bet;
        self.players[player].bet(bet)
    }
}
