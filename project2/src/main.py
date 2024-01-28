from game_manager.manager import GameManager

if __name__ == '__main__':
    num_players = 6
    game_manager = GameManager(num_players)

    game_manager.run()