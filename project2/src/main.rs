mod card;
mod game_manager;
// mod hands;

use game_manager::GameManager;

fn main() {
    let num_players = 2;
    let game_manager = GameManager::new(num_players);

    game_manager.run();
    println!("{}", game_manager);
}
