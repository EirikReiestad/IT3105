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
        actions = list()
        if self._can_fold():
            actions.append(Action.Fold())
        if self._can_check():
            actions.append(Action.Check())
        if self._can_call():
            _, call_sum = self._can_call()
            actions.append(Action.Call(call_sum))
        if self._can_raise(2 * self.buy_in):
            _, raise_sum = self._can_raise(2 * self.buy_in)
            actions.append(Action.Raise(raise_sum))
        if self._can_raise(self.board.pot / 2):
            _, raise_sum = self._can_raise(self.board.pot / 2)
            actions.append(Action.Raise(raise_sum))
        # TODO: Add AllIn

    def _can_fold(self) -> bool:
        """
        Can only fold if the player has not folded and the player has not matched the highest bet
        If it has matched the highest bet, there is no point in folding
        """
        if self.players[self.current_player_index].round_bet == self.board.highest_bet:
            return False
        return True

    def _can_check(self) -> bool:
        return self.players[self.current_player_index].round_bet == self.board.highest_bet

    def _can_call(self) -> (bool, float):
        call_sum = self.board.highest_bet - \
            self.players[self.current_player_index].round_bet
        if self.players[self.current_player_index].chips < call_sum:
            return False, 0
        return True, call_sum

    def _can_raise(self, amount: float) -> (bool, float):
        """
        The amount assume the amount is the amount to raise with and not the total amount to raise to (i.e. the total bet)
        """
        call_sum = self.board.highest_bet - \
            self.players[self.current_player_index].round_bet
        raise_sum = amount + call_sum
        if self.players[self.current_player_index].chips < raise_sum:
            return False, 0
        return True, raise_sum

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
