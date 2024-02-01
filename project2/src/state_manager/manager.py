from src.game_state.game_state import PublicGameState
from src.game_state.board_state import PublicGameBoard
from src.game_manager.game_stage import GameStage
from src.game_manager.players import Players


class StateManager:
    def __init__(self, public_game_state: PublicGameState):
        self.players: Players = public_game_state.players
        self.board: PublicGameBoard = public_game_state.board
        self.game_stage: GameStage = public_game_state.game_stage
