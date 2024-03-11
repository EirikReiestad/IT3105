"""
This module contains the strategy when not using the AI model. This is purely strategy.
It is based on the website: https://www.pokerprofessor.com/university/how-to-win-at-poker

The strategy that is implemented is THIGHT - AGGRESSIVE.
Thight - this means that we are going to limit the starting hadns that we play to the hands that give us the best chance of winning.
    All the hand where the odds are agains us, we are simply going to fold.
Aggressive - when we do get those hands that we are looking for, we are going to play very aggressively.
    This gives us a number of advantages.
    - Limites the number of players: forces weaker hands to fold
    - Increases the pot size: the more money in the pot, the more you can win
    - Two ways to win the pot: by having the best hand, or by making the other players fold.

This introduction only gives a brief overview of the strategy. For a more detailed explanation, please visit the website.

The starting hand groups indicates the value of the type of hands. The s on the right side indicates that the two cards are suited.
    E.g. AK is one ace and one king. AKs is one ace and one king of the same suit.
To make this simpler, we will use a dataclass to represent the hands in the groups.

We will then use the groups to determine what action to do, based on the current state and position
For example if everyone before us has folded or called the big blind, and we are in the early position (defined on the website and further down the code)
we will only raise if we have a hand in group A, B, C or D.

We will use the charts with these steps:
1. What group is your starting hand in? If it isn't in any group then you Fold.
2. What situation are you in? Choose one of the three action charts relevant to teh situation you are in.
3. What position are you in? Look at the column in the chart for the position you are in.
4. Starting hand group now shown? If your starting hand group is not shown in that columns, then you Fold.
5. Starting Hand Group Shown? If your starting hand group letter is shown then take the action the chart is showing you.

Different actions in each of the charts are:
Opening Raise - Make the first raise
Call - Just call when a person has raised
Re-Raise - Re-raise a person who has raised
Call a Re-Raise - Call a person who has re-raised
Raise a Re-Raise - Re-raise a person who has re-raised your original raise
Call the Big Blind - Call the big blind

We will disregard the Call a Re-Raise and Raise a Re-Raise in the Unraised chart, and just use another chart for simplicity.
There is a blinds charts that we should follow if we are in the small or big blind positions.
We will disregard this for simplicity.

To decide what action to take in the other stages, we will use statistics.
We consider two elements:
1. How many "Outs" we have (Card that will make us a winning hand) and how likely it is that an Out will be dealt
2. What are our "Pot Odds" - How much money will we win in return for us taking the gamble that our Out will be dealt

We will use pot odds to decide.
Pot Odds = (Total Pot Size) / (Amount of money you need to call)
With the pot odds, we will calculate the break even percentage.
Break Even Percentage = 100% / sum(Pot Odds)

For example, if the pot are 30 and we need to call 10, the pot odds are 30/10 = 3
The sum of the pot odds are 3 + 1 = 4 (1 is because 10 respect to 30, where 30 is 3 units then 10 is 1 unit)
Note it will only workd if we express the pot odds against a factor of 1 eg. 3:1, 4:1, 5:1 etc.

So based on this, should you call?
Call if - Probability of hitting an Out > Break Even Percentage
Fold if - Probability of hitting an Out < Break Even Percentage
"""

import random
from src.poker_oracle.deck import Deck, Card, Suit
from src.game_state.game_state import PublicGameState
from src.game_state.player_state import PrivatePlayerState
from src.game_manager.game_stage import GameStage
from src.game_manager.game_action import Action
from src.poker_oracle.oracle import Oracle
from src.poker_oracle.hands import Hands


# Starting Hand Cards, SHC for short because it will be used a lot


class SHC:
    def __init__(self, card_one: int, card_two: int, suited: bool = False):
        # Starting Hand Cards
        self.card_one = Card(Suit.None_, card_one)
        self.card_two = Card(Suit.None_, card_two)
        self.suited = suited


STARTING_HAND_GROUPS = {
    'group_a': [SHC(1, 1), SHC(13, 13), SHC(1, 13, True)],
    'group_b': [SHC(1, 13), SHC(12, 12)],
    'group_c': [SHC(11, 11), SHC(10, 10)],
    'group_d': [SHC(1, 12, True), SHC(1, 12), SHC(1, 11, True),
                SHC(9, 9), SHC(8, 8)],
    'group_e': [SHC(1, 11), SHC(1, 10, True), SHC(13, 12, True),
                SHC(7, 7), SHC(6, 6), SHC(5, 5)],
    'group_f': [SHC(1, 10), SHC(13, 12), SHC(13, 11, True),
                SHC(12, 11, True), SHC(4, 4), SHC(3, 3), SHC(2, 2)],
    'group_g': [SHC(1, 9, True), SHC(1, 8, True), SHC(1, 7, True),
                SHC(1, 6, True), SHC(1, 5, True), SHC(1, 4, True),
                SHC(1, 3, True), SHC(1, 2, True), SHC(13, 10, True),
                SHC(12, 10, True), SHC(11, 10, True), SHC(11, 9, True),
                SHC(10, 9, True), SHC(9, 8, True)],
    'group_h': [SHC(13, 11), SHC(13, 10), SHC(12, 11), SHC(11, 8, True),
                SHC(10, 8, True), SHC(8, 7, True), SHC(7, 6, True)]
}

# In the preflop, if everyone before us has folded or called the big blind
PREFLOP_UNRAISED = {
    'raise': {
        'early': ['group_a', 'group_b', 'group_c', 'group_d'],
        'middle': ['group_a', 'group_b', 'group_c', 'group_d', 'group_e'],
        'late': ['group_a', 'group_b', 'group_c', 'group_d', 'group_e', 'group_f'],
    },
    'call': {
        'early': [],
        'middle': ['group_f', 'group_g'],
        'late': ['group_g', 'group_h']
    }
}

# If someone has raised before us
PREFLOP_RAISED = {
    'raise': {
        'early': ['group_a', 'group_b'],
        'middle': ['group_a', 'group_b'],
        'late': ['group_a', 'group_b'],
    },
    'call': {
        'early': ['group_c'],
        'middle': ['group_c'],
        'late': ['group_c', 'group_d']
    }
}


class Strategy:
    def resolve(
        self,
        player: PrivatePlayerState,
        state: PublicGameState,
        end_stage: GameStage,
        end_depth: int,
        verbose: bool = False
    ) -> Action:
        self.player = player
        self.state = state
        self.end_stage = end_stage
        self.end_depth = end_depth
        self.verbose = verbose

        if state.game_stage == GameStage.PreFlop:
            action = self.preflop_bet()
        elif state.game_stage == GameStage.Flop or state.game_stage == GameStage.Turn:
            action = self.flop_turn_bet()
        elif state.game_stage == GameStage.River:
            action = self.river_bet()

        return action

    def river_bet(self) -> Action:
        call_sum = self.state.board_state.highest_bet - self.player.round_bet

        cards = [i for i in self.player.cards]
        cards.extend(self.state.board_state.cards)

        if call_sum == 0:
            if self.should_raise(cards):
                return Action.Raise(self.state.buy_in)
            return Action.Check()
        if self.should_fold(cards):
            return Action.Fold()
        return Action.Call(call_sum)

    def flop_turn_bet(self) -> Action:
        """
        Including the flop and turn stages
        """
        # Decide if we should call
        # Chance of hitting an out > break even percentage
        if self.state.game_stage == GameStage.Flop:
            cards_left = 2
        elif self.state.game_stage == GameStage.Turn:
            cards_left = 1
        else:
            raise ValueError(
                "GameStage must be Flop or Turn, not: {}".format(self.state.game_stage))

        call_sum = self.state.board_state.highest_bet - self.player.round_bet

        # Find number of outs
        # This will be simple, so we will just go through every card that is not visible to us
        # Then if that hand beats, lets say two pairs for now, then we increase the number of outs
        cards = [i for i in self.player.cards]
        cards.extend(self.state.board_state.cards)

        if call_sum == 0:
            if self.should_raise(cards):
                return Action.Raise(self.state.buy_in)
            return Action.Check()

        deck = Deck()

        for card in cards:
            deck.remove(card)

        outs = 0
        for card in deck.stack:
            hand = cards.copy()
            hand.append(card)
            if Oracle.hand_evaluator(cards, hand) == -1:
                outs += 1

        # Calculate pot odds
        pot = self.state.board_state.pot
        pot_odds = pot / call_sum

        # Calculate break even percentage
        print(pot, pot_odds, call_sum)
        break_even_percentage = 100 / (pot_odds + 1)

        probability = 1 - (1 - (outs / len(deck))) ** cards_left

        if probability > break_even_percentage or not self.should_fold(cards):
            # This could also potentially be a raise, but for now we will just call
            return Action.Call(call_sum)
        else:
            return Action.Fold()

    def preflop_bet(self) -> Action:
        fold_threshold = 0.2
        raise_threshold = 0.8
        # Find the position of the player (early, middle, late)
        relative_position = self.get_relative_position()

        group = self.get_preflop_bet_group()
        call_sum = self.state.board_state.highest_bet - self.player.round_bet

        if call_sum < 0:
            raise ValueError("Call sum can not be < 0")

        print("Call sum", call_sum)

        if group is None:
            if call_sum == 0 and random.random() < fold_threshold:
                return Action.Check()
            return Action.Fold()

        if self.player.round_bet == 0 and self.state.board_state.highest_bet == self.state.buy_in:
            if group in PREFLOP_UNRAISED['raise'][relative_position]:
                # TODO: Maybe vary some values but ok for now:)
                raise_amount = call_sum + self.state.buy_in
                if random.random() < raise_threshold:
                    return Action.Raise(raise_amount)
                return Action.Call(call_sum)
            if group in PREFLOP_UNRAISED['call'][relative_position]:
                if call_sum == 0:
                    return Action.Check()
                return Action.Call(call_sum)
        if group in PREFLOP_RAISED['raise'][relative_position]:
            raise_amount = call_sum + self.state.buy_in
            if random.random() < raise_threshold:
                return Action.Raise(raise_amount)
            return Action.Call(call_sum)
        if group in PREFLOP_RAISED['call'][relative_position]:
            if call_sum == 0:
                return Action.Check()
            return Action.Call(call_sum)
        if random.random() < fold_threshold:
            return Action.Fold()
        if call_sum == 0:
            return Action.Check()
        return Action.Call(call_sum)

    def get_preflop_bet_group(self) -> str:
        cards = self.state.player_states[self.state.current_player_index].cards
        cards = sorted(cards, key=lambda x: x.rank, reverse=True)
        if cards[1].rank == 1:
            cards = [cards[1], cards[0]]

        for group in STARTING_HAND_GROUPS:
            for i in range(len(STARTING_HAND_GROUPS[group])):
                if cards[0].rank == STARTING_HAND_GROUPS[group][i].card_one.rank and cards[1].rank == STARTING_HAND_GROUPS[group][i].card_two.rank:
                    if STARTING_HAND_GROUPS[group][i].suited and cards[0].suit == cards[1].suit:
                        return group
                    elif not STARTING_HAND_GROUPS[group][i].suited:
                        return group
        return None

    def get_relative_position(self) -> str:
        """
        return the position of the player (early, middle, late)
        """

        number_of_players = len(self.state.player_states)
        player_index = self.state.current_player_index
        dealer = self.state.board_state.dealer

        player_relative_index = (player_index - dealer) % number_of_players

        if number_of_players == 2:
            if player_index == dealer:
                return "late"
            else:
                return "early"

        # If there are only two players, it is always late if the player_index is the dealer
        # Else, we say that the first 25% of the players are early, 50% are middle, and 25% are late
        # This is just arbitrary, but it is a good starting point
        early = number_of_players // 4
        middle = early * 2

        if player_relative_index < early:
            return "early"
        elif player_relative_index < middle:
            return "middle"
        else:
            return "late"

    @staticmethod
    def should_raise(cards) -> bool:
        hand, _ = Oracle.hand_classifier(cards)
        max_hand_value = Hands.get_max_hand_value()
        value = max_hand_value.value / hand.value
        if random.random() > value:
            return True
        return False

    @staticmethod
    def should_fold(cards) -> bool:
        hand, _ = Oracle.hand_classifier(cards)
        min_hand_value = Hands.get_min_hand_value()
        value = hand.value / min_hand_value.value
        if random.random() > value:
            return True
        return False
