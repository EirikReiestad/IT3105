import sys
import pygame as pg
from src.game_state.game_state import PublicGameState
from src.game_state.board_state import PrivateBoardState
from src.game_state.player_state import PrivatePlayerState
from src.game_manager.game_stage import GameStage
from .card import CardSprite
from src.config import Config

config = Config()


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
        self.occupied_y = 0

    def update(self, game_state: PublicGameState):
        self.player_states: list[PrivatePlayerState] = game_state.player_states
        self.board_state: PrivateBoardState = game_state.board_state
        self.game_stage: GameStage = game_state.game_stage
        self.current_player_index = game_state.current_player_index
        self.occupied_y = 0
        self.clear()
        self._update_title()
        self._update_player_states()
        self._update_board_state()
        pg.display.update()
        self.clock.tick(self.fps)

        for event in pg.event.get():
            if event.type == pg.QUIT:
                sys.exit()

    def draw(self, obj: pg.sprite.Sprite):
        self.display.blit(obj.image, obj.rect)

    def draw_text(self, text, x, y, color=(255, 255, 255)):
        text = self.font.render(text, True, color)
        self.display.blit(text, (x, y))

    def clear(self):
        self.display.fill(self.background)

    def get_width(self):
        return self.width

    def get_height(self):
        return self.height

    def get_input(self):
        while True:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
                if event.type == pg.KEYDOWN:
                    return pg.key.name(event.key)

    def _update_title(self):
        width = self.width // 2 - 20
        self.draw_text(self.game_stage.value, width, 20, (255, 255, 255))
        self.occupied_y += 60

    def _update_player_states(self):
        player_width = self.width // len(self.player_states)
        width = player_width // 3.0
        height = width * 1.5
        margin = player_width // 15

        for i, player in enumerate(self.player_states):
            player_text = f"Player {i + 1}"
            color = (255, 255, 255)
            if self.board_state.dealer == i:
                player_text = f"Player (D) {i + 1}"
            if player.folded:
                color = (100, 100, 100)
            elif i == self.current_player_index:
                color = (255, 0, 0)

            self.draw_text(player_text, player_width * i,
                           self.occupied_y + 30, color)
            self.draw_text(
                f"Chips: {player.chips}", player_width *
                i, self.occupied_y, color
            )
            card_height = 0
            for j, card in enumerate(player.cards):
                x = player_width * i + (width + margin) * j
                y = self.occupied_y + 70
                card_path = "black_joker.png"
                if not config.data['show_cards'] and self.current_player_index == i and not player.ai:
                    card_path = card.to_png_str()
                card_sprite = CardSprite(card_path, x, y, width, height)
                card_height = y
                self.draw(card_sprite)
        self.occupied_y += card_height + 30

    def _update_board_state(self):
        player_width = self.width // len(self.player_states)
        width = player_width // 3.0
        height = width * 1.5
        margin = player_width // 15

        self.occupied_y += 225
        self.draw_text(f"Pot: {self.board_state.pot}", 20, self.occupied_y)
        self.occupied_y += 30
        for i, card in enumerate(self.board_state.cards):
            x = i * (width + margin)
            y = self.occupied_y
            card_path = card.to_png_str()
            card_sprite = CardSprite(card_path, x, y, width, height)
            self.draw(card_sprite)


if __name__ == "__main__":
    from src.game_manager.manager import GameManager
    from src.poker_oracle.deck import Deck

    display = Display()
    deck = Deck()
    num_players = 4
    game_manager = GameManager(num_players, deck)
    game_manager.reset_round(deck)
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
