import pygame as pg
import os
from src.game_state.game_state import PublicGameState
from src.game_state.board_state import PrivateBoardState
from src.game_state.player_state import PrivatePlayerState
from src.game_manager.game_stage import GameStage
from src.poker_oracle.card import Card


class Display:
    def __init__(self, width=1000, height=800):
        self.width = width
        self.height = height
        self.display = pg.display.set_mode((width, height))
        self.clock = pg.time.Clock()
        self.fps = 60
        self.background = (50, 100, 50)
        pg.font.init()
        self.font = pg.font.Font(None, 36)

    def update(self, game_state: PublicGameState):
        self.player_states: list[PrivatePlayerState] = game_state.player_states
        self.board_state: PrivateBoardState = game_state.board_state
        self.game_stage: GameStage = game_state.game_stage
        self._update_player_states()
        pg.display.update()
        self.clock.tick(self.fps)

    def draw(self, obj):
        self.display.blit(obj.image, obj.rect)

    def draw_text(self, text, x, y, color=(255, 255, 255)):
        text = self.font.render(text, True, color)
        self.display.blit(text, (x, y))

    def _draw_card(self, card: Card, x: int, y: int):
        pass

    def clear(self):
        self.display.fill(self.background)

    def get_width(self):
        return self.width

    def get_height(self):
        return self.height

    def _update_player_states(self):
        player_width = self.width // len(self.player_states)
        for (i, player) in enumerate(self.player_states):
            self.draw_text(f"Player {i + 1}", player_width * i, 20)
            for card in player.hand:
                self.draw_card(card, player_width * i, 40)


if __name__ == "__main__":
    from src.game_manager.manager import GameManager
    from src.poker_oracle.deck import Deck
    display = Display()
    deck = Deck()
    deck.reset_stack()
    num_players = 4
    game_manager = GameManager(num_players, deck)
    private_state = game_manager.get_current_private_state()
    display.clear()
    while True:
        display.update(private_state)
        display.clear()
        display.clock.tick(display.fps)
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
    pg.quit()
