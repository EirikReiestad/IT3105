from dataclasses import dataclass


class _Action:
    def __init__(self, action: str, index: int, amount: float = 0):
        self.action: str = action
        self.amount: float = amount
        self.index: int = index  # Used to index the action in the action space

    def __repr__(self):
        return f"{self.action}({self.amount})"

    def __eq__(self, other):
        if isinstance(other, _Action):
            return self.action == other.action
        return False

    def __index__(self):
        return self.index


class Action:
    @staticmethod
    def Fold(amount=0):
        return _Action('Fold', 0, amount)

    @staticmethod
    def Call(amount):
        return _Action('Call', 1, amount)

    @staticmethod
    def Check(amount=0):
        return _Action('Check', 2, amount)

    @staticmethod
    def Raise(amount):
        return _Action('Raise', 3,  amount)

    @staticmethod
    def AllIn(amount):
        return _Action('AllIn', 4, amount)

    @staticmethod
    def CallOrCheck(amount=0):
        return _Action('CallOrCheck', 5, amount)
