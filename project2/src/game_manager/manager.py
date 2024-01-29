from src.poker_oracle.deck import Deck, Card
from src.game_state.states import GameState, PlayerState, BoardState
from .players import Players, Player
from typing import List, Tuple, Optional
from dataclasses import dataclass
from .game_stage import GameStage
from .game_action import Action
import random


@dataclass
class Board:
    def __init__(
        self,
        flop: Tuple[Card, Card, Card],
        turn: Card,
        river: Card,
        pot: int,
        highest_bet: int,
        dealer: int,
    ):
        if not isinstance(flop, tuple):
            raise TypeError("flop must be a tuple")
        if len(flop) != 3:
            raise ValueError("flop must be a tuple of length 3")
        if not isinstance(turn, Card):
            raise TypeError("turn must be a Card")
        if not isinstance(river, Card):
            raise TypeError("river must be a Card")
        self.flop = flop  # A tuple of three Card instances
        self.turn = turn  # A Card instance
        self.river = river  # A Card instance
        self.pot = pot  # An integer
        self.highest_bet = highest_bet  # An integer
        self.dealer = dealer  # An integer


class GameManager:
    def __init__(self, num_players: int):
        # Generate a stack of cards and shuffle them
        deck = Deck()
        deck.reset_stack()

        # Error if there are not enough cards for the game

        # Deal cards to players
        players = []

        for _ in range(num_players):
            first_card = deck.pop()
            second_card = deck.pop()

            if not isinstance(first_card, Card):
                raise TypeError(
                    "first_card must be a Card, not {}".format(type(first_card))
                )
            if not isinstance(second_card, Card):
                raise TypeError(
                    "second_card must be a Card, not {}".format(type(second_card))
                )

            players.append(Player((first_card, second_card)))

        # Deal flop, turn, and river
        flop = (deck.pop(), deck.pop(), deck.pop())
        turn = deck.pop()
        river = deck.pop()

        for card in flop:
            if not isinstance(card, Card):
                raise TypeError("card must be a Card")
        if not isinstance(turn, Card):
            raise TypeError("turn must be a Card")
        if not isinstance(river, Card):
            raise TypeError("river must be a Card")

        # Determine dealer
        dealer: int = random.randint(0, num_players - 1)

        self.players: Players = Players(players)
        self.game_stage: GameStage = GameStage.PreFlop
        self.board: Board = Board(flop, turn, river, 0, 0, dealer)
        self.buy_in: int = 10

    def get_action(self) -> Tuple[Action, Optional[int]]:
        """
        Returns
        -------
        Tuple[Action, Optional[int]]: The action and the amount to bet if the action is raise
        """
        options = "Fold(0) Call/Check(1) Raise(2)"  # TODO: Implement all-in
        print(f"Options: {options}")

        def get_input():
            user_input = input()

            if user_input == "0":
                return Action.Fold, None
            elif user_input == "1":
                return Action.CallOrCheck, None
            elif user_input == "2":
                return Action.Raise, 10  # TODO: Implement raise amount
            else:
                print("Invalid input")
                return get_input()

        return get_input()

    # Handles generating the game state
    def get_current_state(self) -> GameState:
        player_states: List[PlayerState] = self.get_player_states()
        board_state: BoardState = self.get_board_state()
        game_stage: GameStage = self.game_stage

        return GameState(player_states, board_state, game_stage)

    def get_player_states(self) -> List[PlayerState]:
        player_states = []
        for player in self.players:
            player_state = PlayerState(player.chips, player.folded, player.bet)
            player_states.append(player_state)
        return player_states

    def get_board_state(self):
        if self.game_stage == GameStage.PreFlop:
            cards = []
        elif self.game_stage == GameStage.Flop:
            flop = self.board.flop
            cards = [flop[0], flop[1], flop[2]]
        elif self.game_stage == GameStage.Turn:
            flop = self.board.flop
            turn = self.board.turn
            cards = [flop[0], flop[1], flop[2], turn]
        elif self.game_stage == GameStage.River:
            flop = self.board.flop
            turn = self.board.turn
            river = self.board.river
            cards = [flop[0], flop[1], flop[2], turn, river]

        return BoardState(
            cards, self.board.pot, self.board.highest_bet, self.board.dealer
        )

    # Implements the rules

    # Handles making a bet
    # If the player does not have enough money, they are forced to fold
    # If the player has enough money, the bet is made
    # The bet is added to the pot
    # The bet is added to the player_bets HashMap
    # The highest_bet is updated
    def make_bet(self, player, bet):
        if not self.players.place_bet(player, bet):
            print(
                "Not enough money. Folded."
            )  # TODO: Do not need to fold. Can go all-in or check if possible
            self.players.fold(player)
        else:
            player_bet = self.players.get_bet(player)
            self.board.pot += bet
            print(f"Player bet: {player_bet}, bet {bet}")
            self.board.highest_bet = player_bet

    def rotate_dealer(self):
        self.board.dealer = (self.board.dealer + 1) % len(self.players)

    # Implements the game logic

    # Runs the game
    # The game is run in a loop until a player wins
    # The game is run in stages
    # - PreFlop
    # - Flop
    # - Turn
    # - River
    def run(self):
        # TODO: Add rotation of dealer
        while True:
            if self.game_stage == GameStage.PreFlop:
                winner = self.run_game_stage()
                if winner is not None:
                    print(f"Player {winner} won {self.board.pot} units!")
                    break
                self.game_stage = GameStage.Flop
            elif self.game_stage == GameStage.Flop:
                self.run_game_stage()
                self.game_stage = GameStage.Turn
            elif self.game_stage == GameStage.Turn:
                self.run_game_stage()
                self.game_stage = GameStage.River
            elif self.game_stage == GameStage.River:
                self.run_game_stage()
                break

    # Runs a game stage
    def run_game_stage(self):
        check_count = 0

        print(self)

        while check_count != self.players.get_number_of_active_players():
            for i in range(len(self.players)):
                turn = (self.board.dealer + i + 1) % len(self.players)

                if self.players.has_folded(turn):
                    continue

                if self.players.get_number_of_active_players() == 1:
                    return turn

                print(" ")
                print(f"--- Player {turn}'s turn ---")
                print(self.players.players[turn])

                if self.game_stage == GameStage.PreFlop:
                    x = self.preflop_bets(turn)
                    if x is not None:
                        check_count += x
                        continue

                action, amount = self.get_action()

                if action == Action.Fold:
                    self.players.fold(turn)
                    continue
                elif action == Action.CallOrCheck:
                    player_bet = self.players.get_bet(turn)
                    bet = self.board.highest_bet - player_bet
                    self.make_bet(turn, bet)
                    if bet == 0:
                        print("Checked")
                    else:
                        print(f"Called {bet}")
                    check_count += 1
                elif action == Action.Raise:
                    player_bet = self.players.get_bet(turn)
                    raise_amount = self.board.highest_bet - player_bet + amount
                    self.make_bet(turn, raise_amount)
                    check_count = 1
                else:
                    raise ValueError("Invalid action")

                if check_count == self.players.get_number_of_active_players():
                    break

    # Handles the small and big blind
    # Handles automatic betting for the small and big blind
    # Assuming they can not fold.
    # Returns 1 if the player is the big blind, 0 otherwise
    def preflop_bets(self, turn: int):
        player_bet: int = self.players.get_bet(turn)
        small_blind = (self.board.dealer + 1) % len(self.players)
        big_blind = (self.board.dealer + 2) % len(self.players)

        print(f"turn {turn} player_bet {player_bet}")

        if turn == small_blind and self.board.highest_bet == 0:
            # Small blind
            print(f"Player {turn} is the small blind and must bet {self.buy_in / 2}")
            self.make_bet(turn, self.buy_in / 2)
            return 0
        elif turn == big_blind and self.board.highest_bet == self.buy_in / 2:
            # Big blind
            print(f"Player {turn} is the big blind and must bet {self.buy_in}")
            self.make_bet(turn, self.buy_in)
            return 1
        else:
            return None

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
