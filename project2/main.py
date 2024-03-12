from src.game_manager.manager import GameManager

if __name__ == "__main__":
    num_players = 1
    num_ai = 5
    game_manager = GameManager(
        num_players=num_players, num_ai=num_ai, graphics=True)

    game_manager.run_game()
