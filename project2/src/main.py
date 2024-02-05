from src.game_manager.manager import GameManager
from src.poker_oracle.deck import Deck

if __name__ == "__main__":
    num_players = 0
    num_ai = 2
    deck = Deck()
    deck.reset_stack(simplify=True)
    game_manager = GameManager(num_players=num_players, num_ai=num_ai, deck=deck)

    game_manager.run_game()
