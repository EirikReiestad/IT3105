from enum import Enum


class Action(Enum):
    Fold = "Fold"
    CallOrCheck = "CallOrCheck"
    Call = "Call"
    Check = "Check"
    Raise = "Raise"
    AllIn = "AllIn"
