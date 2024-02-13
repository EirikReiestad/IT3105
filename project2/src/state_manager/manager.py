from typing import List
from src.game_state.game_state import PublicGameState
from src.game_state.board_state import PublicBoardState
from src.game_manager.game_stage import GameStage
from src.game_manager.players import Players
from src.game_manager.game_action import Action


class StateManager:
    def __init__(self, public_game_state: PublicGameState):
        self.players: Players = public_game_state.player_states
        self.board: PublicBoardState = public_game_state.board_state
        self.game_stage: GameStage = public_game_state.game_stage
        self.current_player_index: int = public_game_state.current_player_index
        self.buy_in: int = public_game_state.buy_in
        self.check_count: int = 0

    def get_legal_actions(self) -> List[Action]:
        """
        There are some restrictions to reduce the number of states that need to be generated.
        That includes:
        One can only raise
            - 2x big blind
            - 1/2 pot (wip)
        """
        actions = list()
        if self._can_fold():
            actions.append(Action.Fold())
        if self._can_check():
            actions.append(Action.Check())
        if self._can_call():
            can_call, call_sum = self._can_call()
            if can_call:
                actions.append(Action.Call(call_sum))
        if self._can_raise(2 * self.buy_in):
            can_raise, raise_sum = self._can_raise(2 * self.buy_in)
            if can_raise:
                actions.append(Action.Raise(raise_sum))
        if self._can_raise(self.board.pot / 2):
            can_raise, raise_sum = self._can_raise(self.board.pot / 2)
            if can_raise:
                actions.append(Action.Raise(raise_sum))
        # TODO: Add AllIn
        return actions

    def get_num_legal_actions(self) -> int:
        return len(self.get_legal_actions())

    def _can_fold(self) -> bool:
        """
        Can only fold if the player has not folded and the player has not matched the highest bet
        If it has matched the highest bet, there is no point in folding
        """
        if self.players[self.current_player_index].round_bet == self.board.highest_bet:
            return False
        return True

    def _can_check(self) -> bool:
        return (
            self.players[self.current_player_index].round_bet == self.board.highest_bet
        )

    def _can_call(self) -> (bool, float):
        if self._can_check():
            return False, 0
        call_sum = (
            self.board.highest_bet -
            self.players[self.current_player_index].round_bet
        )
        if self.players[self.current_player_index].chips < call_sum:
            return False, 0
        return True, call_sum

    def _can_raise(self, amount: float) -> (bool, float):
        """
        The amount assume the amount is the amount to raise with and not the total amount to raise to (i.e. the total bet)
        """
        call_sum = (
            self.board.highest_bet -
            self.players[self.current_player_index].round_bet
        )
        raise_sum = amount + call_sum
        if raise_sum <= 0:
            return False, 0
        if self.players[self.current_player_index].chips < raise_sum:
            return False, 0
        return True, raise_sum

    def generate_sub_state(self, action: Action) -> [PublicGameState]:
        """
        The generate_sub_state method is used to generate a new state from the current state based on the action
        """
        public_game_state = PublicGameState(
            self.players,
            self.board,
            self.game_stage,
            self.current_player_index,
            self.buy_in,
            self.check_count,
        )
        state = StateManager(public_game_state)
        return state.generate_state(action)

    def generate_state(self, action: Action) -> PublicGameState:
        """
        Generate a new state based on the action
        """
        # Increment the current player index
        self.current_player_index = (
            self.current_player_index + 1) % len(self.players)

        # TODO: This is a bit of a mess. Need to clean this up.
        # The logic is not very clear as it already exists in the Player class
        # Consider passing in the Player class instead of PublicPlayerState
        #
        active_players = [
            player for player in self.players if not player.folded and not player.bust
        ]

        if self.check_count == active_players:
            self.game_stage = self.game_stage.next_stage()
            self.check_count = 0

        # TODO: Check if the game is over

        if action == Action.Fold():
            self.players[self.current_player_index].folded = True
        elif action == Action.Check():
            self.check_count += 1
        elif action == Action.Call(0):
            _, call_sum = self._can_call()
            self.players[self.current_player_index].chips -= call_sum
            self.players[self.current_player_index].round_bet += call_sum
            self.board.pot += call_sum
        elif action == Action.Raise(0):
            _, raise_sum = self._can_raise(action.amount)
            self.players[self.current_player_index].chips -= raise_sum
            self.players[self.current_player_index].round_bet += raise_sum
            self.board.pot += raise_sum
            self.board.highest_bet += raise_sum
        elif action == Action.AllIn(0):
            raise NotImplementedError
        else:
            raise ValueError("Invalid action")

        return PublicGameState(
            self.players,
            self.board,
            self.game_stage,
            self.current_player_index,
            self.buy_in,
            self.check_count,
        )

    def generate_possible_states(self) -> List[PublicGameState]:
        possible_states = list()
        for action in self.get_legal_actions():
            possible_states.append(self.generate_sub_state(action))
        return possible_states
