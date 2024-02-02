import pygame
import os
from src.game_state.game_state import PublicGameState
from src.game_state.board_state import PrivateBoardState
from src.game_state.player_state import PrivatePlayerState
from src.game_manager.game_stage import GameStage


class Display:
    def __init__(self, width=600, height=600):
        self.width = width
        self.height = height
        self.display = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        self.fps = 60
        self.background = (0, 0, 0)
        self.font = pygame.font.Font(os.path.join(
            "src", "gui", "fonts", "font.ttf"), 20)

    def update(self, game_state: PublicGameState):
        self.player_states: list[PrivatePlayerState] = game_state.player_states
        self.board_state: PrivateBoardState = game_state.board_state
        self.game_stage: GameStage = game_state.game_stage
        pygame.display.update()
        self.clock.tick(self.fps)

    def draw(self, obj):
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

    def _update_player_states(self, player_states: list[PrivatePlayerState]):
        for player_state in player_states:
            self.draw_text(
                f"Player {player_state.player_id}: {player_state.score}", 10, 10)
