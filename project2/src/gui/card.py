import pygame as pg


class CardSprite(pg.sprite.Sprite):
    def __init__(self, card_path: str, x: int, y: int, width: int = 100, height: int = 150):
        super().__init__()
        src = "static/images/" + card_path
        self.image = pg.image.load(src)
        self.image = pg.transform.scale(self.image, (width, height))
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
