from typing import List
from src.game_state.player_state import PublicPlayerState, PrivatePlayerState
from src.poker_oracle.deck import Deck, Card


class Players:
    def __init__(self, num_players: int):
        self.players: List(PrivatePlayerState) = [
            PrivatePlayerState() for _ in range(num_players)]

    def reset_round(self, deck: Deck) -> Deck:
        # Deal cards to players
        for player in self.players:
            first_card = deck.pop()
            second_card = deck.pop()

            if not isinstance(first_card, Card):
                raise TypeError(
                    "first_card must be a Card, not {}".format(
                        type(first_card))
                )
            if not isinstance(second_card, Card):
                raise TypeError(
                    "second_card must be a Card, not {}".format(
                        type(second_card))
                )

            player.reset_round((first_card, second_card))
        return deck

    def winner(self, player: int, amount: int):
        self.players[player].chips += amount

    def get_private_player_states(self) -> List[PrivatePlayerState]:
        return self.players

    def get_public_player_states(self) -> List[PublicPlayerState]:
        return [player.to_public() for player in self.players]

    def __len__(self):
        return len(self.players)

    def __iter__(self):
        return iter(self.players)

    def get_number_of_folded(self) -> int:
        # Note: This is more expensive than a simple counter.
        # However, the number of players is small so this is not an issue,
        # and it is more readable.
        # TODO: Consider using a counter instead.
        return sum(player.folded for player in self.players)

    def get_number_of_bust(self) -> int:
        return sum(player.bust for player in self.players)

    def get_number_of_active_players(self) -> int:
        return len(self.players) - self.get_number_of_folded() - self.get_number_of_bust()

    def has_folded(self, player: int) -> bool:
        return self.players[player].folded

    def is_bust(self, player: int) -> bool:
        return self.players[player].bust

    def fold(self, player: int):
        self.players[player].folded = True

    def get_bet(self, player: int) -> int:
        return self.players[player].round_bet

    def action(self, player, action) -> bool:
        return self.players[player].action(action, action.amount)
