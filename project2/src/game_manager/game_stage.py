from enum import Enum


class GameStage(Enum):
    PreFlop = "PreFlop"
    Flop = "Flop"
    Turn = "Turn"
    River = "River"
    Showdown = "Showdown"

    def next_stage(self):
        if self == GameStage.PreFlop:
            return GameStage.Flop
        elif self == GameStage.Flop:
            return GameStage.Turn
        elif self == GameStage.Turn:
            return GameStage.River
        elif self == GameStage.River:
            return GameStage.Showdown
        else:
            return None

    def __str__(self):
        return self.value
