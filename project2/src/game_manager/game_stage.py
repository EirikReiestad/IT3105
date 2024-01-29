from enum import Enum


class GameStage(Enum):
    PreFlop = "PreFlop"
    Flop = "Flop"
    Turn = "Turn"
    River = "River"

    def __str__(self):
        return self.value
