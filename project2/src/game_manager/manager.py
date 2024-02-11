from typing import List, Tuple, Optional
import random
from src.poker_oracle.deck import Deck
from src.game_state.player_state import PublicPlayerState, PrivatePlayerState
from src.game_state.board_state import PublicBoardState, PrivateBoardState
from src.game_state.game_state import PublicGameState, PrivateGameState
from src.resolver.resolver import Resolver
from .players import Players
from .game_stage import GameStage
from .game_action import Action
from src.gui.display import Display


class GameManager:
    def __init__(
        self, num_players: int, deck: Deck, num_ai: int = 1, graphics: bool = True
    ):
        # Initialization
        self.buy_in: int = 10

        total_players = num_players + num_ai

        dealer: int = random.randint(0, total_players - 1)
        self.board: PrivateBoardState = PrivateBoardState(0, 0)
        deck = self.board.reset_round(deck, dealer)
        self.players: Players = Players(num_players, num_ai)
        deck = self.players.reset_round(deck)

        self.game_stage: GameStage = GameStage.PreFlop
        self.check_count: int = 0
        self.resolver = Resolver()
        self.graphics: bool = graphics
        if graphics:
            self._init_graphics()

    def _init_graphics(self):
        self.display = Display()

    def get_player_action(self) -> Action:
        """
        Returns
        -------
        Action: The action and the amount to bet if the action is raise
        """

        def get_input() -> Action:
            if self.graphics:
                user_input = self.display.get_input()
            else:
                user_input = input()

            if user_input == "0":
                return Action.Fold()
            elif user_input == "1":
                return Action.CallOrCheck()
            elif user_input == "2":
                return Action.Raise(10)  # TODO: Implement raise amount
            else:
                print("Invalid input")
                return get_input()

        return get_input()

    def get_ai_action(self) -> Action:
        # TODO: Environment file?
        end_stage = self.game_stage.next_stage()
        end_depth = 3
        num_rollouts = 10
        return self.resolver.resolve(
            self.get_current_public_state(),
            end_stage,
            end_depth,
            num_rollouts,
        )

        # Handles generating the game state

    def get_current_public_state(self) -> PublicGameState:
        player_states: List[PublicPlayerState] = self.players.get_public_player_states()
        self.board.update_board_state(self.game_stage)
        board_state: PublicBoardState = self.board.to_public()
        game_stage: GameStage = self.game_stage

        return PublicGameState(
            player_states,
            board_state,
            game_stage,
            self.current_player_index,
            self.buy_in,
            self.check_count,
        )

    def get_current_private_state(self) -> PrivateGameState:
        player_states: PrivatePlayerState = self.players.get_private_player_states()
        self.board.update_board_state(self.game_stage)
        board_state: PrivateBoardState = self.board
        game_stage: GameStage = self.game_stage
        return PublicGameState(
            player_states,
            board_state,
            game_stage,
            self.current_player_index,
            self.buy_in,
            self.check_count,
        )

    # Implements the rules

    # Handles making a bet
    # If the player does not have enough money, they are forced to fold
    # If the player has enough money, the bet is made
    # The bet is added to the pot
    # The bet is added to the player_bets HashMap
    # The highest_bet is updated
    def make_bet(self, player: int, action: Action):
        if not self.players.action(player, action):
            self.players.action(player, Action.Fold())
        else:
            player_bet = self.players.get_bet(player)
            self.board.pot += action.amount
            self.board.highest_bet = player_bet
            if not self.graphics:
                print(f"Player bet: {player_bet}, bet {action.amount}")

    def reset_round(self, deck: Deck):
        """
        Resets the round data
        """
        self.game_stage = GameStage.PreFlop
        deck = self.players.reset_round(deck)
        dealer = self.get_new_dealer(self.board.dealer)
        deck = self.board.reset_round(deck, dealer)

    # Implements the game logic

    # Runs the game
    # The game is run in a loop until a player wins
    # The game is run in stages
    # - PreFlop
    # - Flop
    # - Turn
    # - River

    def run_game(self):
        """
        Runs the game, multiple rounds
        """
        print(self._rules())
        # TODO: Remove the generation of deck, it should ideally be passed in
        while True:
            deck = Deck()
            deck.reset_stack()
            self.run_round()
            self.reset_round(deck)

    def run_round(self):
        """
        Runs a single round, meaning from the pre-flop to the river
        """
        while True:
            if not self.graphics:
                print(self)
            match self.game_stage:
                case GameStage.PreFlop:
                    winner = self.run_game_stage()
                    if self.round_winner(winner):
                        return
                    self.game_stage.next_stage()
                case GameStage.Flop:
                    winner = self.run_game_stage()
                    if self.round_winner(winner):
                        return
                    self.game_stage.next_stage()
                case GameStage.Turn:
                    winner = self.run_game_stage()
                    if self.round_winner(winner):
                        return
                    self.game_stage.next_stage()
                case GameStage.River:
                    winner = self.run_game_stage()
                    if self.round_winner(winner):
                        return
                    self.game_stage.next_stage()
                case GameStage.Showdown:
                    self.round_winner(winner)
                    break

    def round_winner(self, winner: int) -> bool:
        """
        Handles the winner of the round
        """
        if winner is None:
            return False
        self.players.winner(winner, self.board.pot)
        print(f"Player {winner} won {self.board.pot} units!")
        return True

    # Runs a game stage
    def run_game_stage(self) -> int:
        self.check_count = 0

        while self.check_count != self.players.get_number_of_active_players():
            game_over, value = self.game_stage_next()
            if game_over:
                return value
            self.check_count = value

    def game_stage_next(self) -> (bool, int):
        """
        Returns
        -------
        bool: True if the round is over, False otherwise
        int: The winner of the round, if the round is over, else the check count
        """
        for i in range(len(self.players)):
            turn = (self.board.dealer + i + 1) % len(self.players)

            if self.players.has_folded(turn):
                continue

            if self.players.get_number_of_active_players() == 1:
                return True, turn

            self.current_player_index = turn

            if self.graphics:
                self.display.update(self.get_current_private_state())
            else:
                print(" ")
                print(f"--- Player {turn}'s turn ---")
                print(self.players.players[turn])

            if self.game_stage == GameStage.PreFlop:
                self.check_count += self.preflop_bets(turn)

            # Check if the player is an AI
            if self.players.is_ai(turn):
                action: Action = self.get_ai_action()
            else:
                action: Action = self.get_player_action()

            if action == Action.Fold():
                self.players.fold(turn)
                continue
            elif action == Action.CallOrCheck():
                player_bet = self.players.get_bet(turn)
                bet = self.board.highest_bet - player_bet
                if bet == 0:
                    if not self.graphics:
                        print("Checked")
                    self.make_bet(turn, Action.Check())
                else:
                    if not self.graphics:
                        print(f"Called {bet}")
                    self.make_bet(turn, Action.Call(bet))
                self.check_count += 1
            elif action == Action.Raise():
                player_bet = self.players.get_bet(turn)
                raise_amount = self.board.highest_bet - player_bet + action.amount
                self.make_bet(turn, Action.Raise(raise_amount))
                self.check_count = 1
            else:
                raise ValueError("Invalid action")

            if self.check_count == self.players.get_number_of_active_players():
                break
        return False, self.check_count

    def run_player(self):
        """
        Run the game with players turn
        """
        pass

    def run_ai(self):
        """
        Run the game with AIs turn
        """
        # Add resolver
        pass

        # Handles the small and big blind
        # Handles automatic betting for the small and big blind
        # Assuming they can not fold.
        # Returns 1 if the player is the big blind, 0 otherwise

    def preflop_bets(self, turn: int) -> int:
        player_bet: int = self.players.get_bet(turn)
        small_blind = (self.board.dealer + 1) % len(self.players)
        big_blind = (self.board.dealer + 2) % len(self.players)

        if not self.graphics:
            print(f"turn {turn} player_bet {player_bet}")

        if turn == small_blind and self.board.highest_bet == 0:
            # Small blind
            if not self.graphics:
                print(
                    f"Player {turn} is the small blind and must bet {self.buy_in / 2}"
                )
            self.make_bet(turn, Action.Raise(self.buy_in / 2))
            return 0
        elif turn == big_blind and self.board.highest_bet == self.buy_in / 2:
            # Big blind
            if not self.graphics:
                print(f"Player {turn} is the big blind and must bet {self.buy_in}")
            self.make_bet(turn, Action.Raise(self.buy_in))
            return 1
        else:
            return 0

    def get_new_dealer(self, dealer: int):
        dealer = (dealer + 1) % len(self.players)

        if self.players.is_bust(dealer):
            self.rotate_dealer()
        return dealer

    def _rules(self) -> str:
        s = "Welcome to Poker!\n"
        s += "Press 0 to fold, 1 to call/check, and 2 to raise\n"
        return s

    def __str__(self):
        result = (
            "\n" + "=" * 75 + "\n" + " " * 25 + "GAME MANAGER" + "\n" + "=" * 75 + "\n"
        )
        result += "===== {} =====\n".format(self.game_stage)
        result += "Pot: {}\n".format(self.board.pot)

        if self.game_stage in ["Flop", "Turn", "River"]:
            result += "Flop: {} {} {}\n".format(
                self.board.flop[0], self.board.flop[1], self.board.flop[2]
            )
        if self.game_stage in ["Turn", "River"]:
            result += "Turn: {}\n".format(self.board.turn)
        if self.game_stage == "River":
            result += "River: {}\n".format(self.board.river)

        result += "\n===== PLAYERS =====\n"
        for i, player in enumerate(self.players):
            if i == self.board.dealer:
                result += "Player {} (Dealer): {}\n".format(i, player)
            elif i == (self.board.dealer + 1) % len(self.players):
                result += "Player {} (Small Blind): {}\n".format(i, player)
            elif i == (self.board.dealer + 2) % len(self.players):
                result += "Player {} (Big Blind): {}\n".format(i, player)
            else:
                result += "Player {}: {}\n".format(i, player)

        return result
