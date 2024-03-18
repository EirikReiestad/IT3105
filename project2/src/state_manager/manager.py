import copy
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
        # NOTE: Chanced this from 0 to public_game_state.check_count because it seemed correct, but not sure
        self.check_count: int = public_game_state.check_count
        self.raise_count: int = public_game_state.raise_count
        self.chance_event: bool = public_game_state.chance_event

    def get_legal_actions(self) -> List[Action]:
        """
        There are some restrictions to reduce the number of states that need to be generated.
        That includes:
        One can only raise
            - 2x big blind
            - 1/2 pot (wip)
        """
        actions = list()
        can_fold = self._can_fold()
        can_check = self._can_check()
        can_call, call_sum = self._can_call()
        can_raise_1, raise_sum_1 = self._can_raise(1.0)
        # can_raise2x, raise_sum2x = self._can_raise(2 * self.buy_in)
        # can_raise_half_pot, raise_sum_half_pot = self._can_raise(
        #     self.board.pot / 2)

        if can_fold:
            actions.append(Action.Fold())
        if can_check:
            actions.append(Action.Check())
        if can_call:
            actions.append(Action.Call(call_sum))
        if can_raise_1:
            actions.append(Action.Raise(raise_sum_1))

        # if can_raise2x:
        # actions.append(Action.Raise(raise_sum2x))
        # if can_raise_half_pot:
        # actions.append(Action.Raise(raise_sum_half_pot))
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
        # print("HIGHEST BET", self.board.highest_bet, "ROUND_BET", self.players[self.current_player_index].round_bet)
        call_sum = (
            self.board.highest_bet -
            self.players[self.current_player_index].round_bet
        )
        if call_sum < 0:
            raise ValueError("Call sum should never be less than 0. Current highest bet: {}. Current player bet: {}".format(
                self.board.highest_bet, self.players[self.current_player_index].round_bet))
        if self.players[self.current_player_index].chips < call_sum:
            return False, 0
        return True, call_sum

    def _can_raise(self, amount: float) -> (bool, float):
        """
        The amount assume the amount is the amount to raise with and not the total amount to raise to (i.e. the total bet)
        """

        if self.raise_count == 4:
            return False, 0

        call_sum = (
            self.board.highest_bet -
            self.players[self.current_player_index].round_bet
        )

        if call_sum < 0:
            raise ValueError("Call sum should never be less than 0. Current highest bet: {}. Current player bet: {}".format(
                self.board.highest_bet, self.players[self.current_player_index].round_bet))

        if amount <= 0:
            raise ValueError(
                "Amount to raise with should be greater than 0. Amount: {}".format(amount))

        raise_sum = float(call_sum + amount)

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
            self.raise_count,
            self.chance_event,
        )
        state = StateManager(copy.deepcopy(public_game_state))
        public_game_state = state.generate_state(action)
        return public_game_state

    def generate_state(self, action: Action) -> PublicGameState:
        """
        Generate a new state based on the action.
        It also rotates the player index to the next player
        """
        # TODO: This is a bit of a mess. Need to clean this up.
        # The logic is not very clear as it already exists in the Player class
        # Consider passing in the Player class instead of PublicPlayerState
        #
        active_players = [
            player for player in self.players if not player.folded and not player.bust
        ]

        if self.check_count == active_players:
            self.chance_event = True
            self.game_stage = self.game_stage.next_stage()
            self.check_count = 0

        # TODO: Check if the game is over

        if action == Action.Fold():
            self.players[self.current_player_index].folded = True
        elif action == Action.Check():
            self.check_count += 1
        elif action == Action.Call(0):
            call_sum = action.amount
            self.check_count += 1

            self.players[self.current_player_index].chips -= call_sum
            self.players[self.current_player_index].round_bet += call_sum
            self.board.pot += call_sum
        elif action == Action.Raise(0):
            # action.amount should be the amount to raise with + the amount to call
            self.check_count = 1
            self.raise_count += 1
            self.players[self.current_player_index].chips -= action.amount
            self.players[self.current_player_index].round_bet += action.amount
            self.board.pot += action.amount
            self.board.highest_bet = self.players[self.current_player_index].round_bet

        elif action == Action.AllIn(0):
            raise NotImplementedError
        else:
            raise ValueError("Invalid action")

        # Increment the current player index
        # self.current_player_index = (
        #     self.current_player_index + 1) % len(self.players)

        return PublicGameState(
            self.players,
            self.board,
            self.game_stage,
            (self.current_player_index + 1) % len(self.players),
            self.buy_in,
            self.check_count,
            self.raise_count,
            self.chance_event,
        )

    def generate_possible_states(self, actions: [Action]) -> List[PublicGameState]:
        possible_states = list()
        # print("START")
        for action in actions:

            possible_states.append(self.generate_sub_state(action))
        return possible_states

    def __repr__(self):
        return f"""
        ===================================
        Players: {self.players},
        -----------------------------------
        Board: {self.board},
        -----------------------------------
        Game Stage: {self.game_stage},
        -----------------------------------
        Current Player Index: {self.current_player_index},
        -----------------------------------
        Buy In: {self.buy_in},
        -----------------------------------
        Check Count: {self.check_count}
        ==================================="""
