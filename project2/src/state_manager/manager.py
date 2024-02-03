from typing import List
from src.game_state.game_state import PublicGameState
from src.game_state.board_state import PublicGameBoard
from src.game_manager.game_stage import GameStage
from src.game_manager.players import Players
from src.game_manager.game_action import Action


class StateManager:
    def __init__(self, public_game_state: PublicGameState):
        self.players: Players = public_game_state.players
        self.board: PublicGameBoard = public_game_state.board
        self.game_stage: GameStage = public_game_state.game_stage
        self.current_player_index: int = public_game_state.current_player_index
        self.buy_in: int = public_game_state.buy_in

    def get_legal_actions(self) -> List[Action]:
        """
        There are some restrictions to reduce the number of states that need to be generated.
        That includes:
        One can only raise
            - 2x big blind
            - 1/2 pot
        """
        if self.players[self.current_player_index].round_bet == self.board.highest_bet:
            # You can fold, but there is no point as you can check for free
            # NOTE: We use Action.CheckOrCall in the game_manager
            # TODO: Need to distinguish between the different types of raises as it can afford one type but not the other
            if self.players[self.current_player_index].chips < self.self.buy_in * 2:
                return [Action.Check]
            if self.players[self.current_player_index].chips < self.board.pot / 2:
                return [Action.Check]
            return [Action.Check, Action.Raise]
        if self.players[self.current_player_index].round_bet < self.board.highest_bet:
            check_sum = self.board.highest_bet - \
                self.players[self.current_player_index].round_bet
            if self.players[self.current_player_index].chips < check_sum:
                return [Action.Fold]
            if self.players[self.current_player_index].chips < self.self.buy_in * 2 + check_sum:
                return [Action.Fold, Action.Call]
            if self.players[self.current_player_index].chips < self.board.pot / 2 + check_sum:
                return [Action.Fold, Action.Call]
            return [Action.Fold, Action.Call, Action.Raise]
        return []

    def generate_sub_states(self, player_index: int, action: Action) -> [PublicGameState]:
        """
        The generate_sub_state method is used to generate a new state from the current state based on the action
        """

        return PublicGameState(
            players=self.players,
            board=self.board,
            game_stage=self.game_stage,
            current_player_index=player_index)

    def generate_possible_states(self) -> List[PublicGameState]:
        possible_states = list()
        for action in self.get_legal_actions():
            possible_states.append(self.generate_sub_state(
                self.current_player_index, action))
        return possible_states
