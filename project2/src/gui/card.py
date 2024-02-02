import pygame as pg
from src.poker_oracle.deck import Card


class CardSprite(pg.sprite.Sprite):
    def __init__(self, card: Card, x: int, y: int, width: int = 100, height: int = 150):
        super().__init__()
        src = "static/images/" + card.to_png_str()
        self.image = pg.image.load(src)
        self.image = pg.transform.scale(self.image, (width, height))
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
