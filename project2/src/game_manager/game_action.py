from dataclasses import dataclass


class _Action:
    def __init__(self, action: str, amount: float = 0):
        self.action: str = action
        self.amount: float = amount

    def __repr__(self):
        return f"{self.action}({self.amount})"

    def __eq__(self, other):
        if isinstance(other, _Action):
            return self.action == other.action
        return False


class Action:
    @staticmethod
    def Fold(amount=0):
        return _Action('Fold', amount)

    @staticmethod
    def CallOrCheck(amount=0):
        return _Action('CallOrCheck', amount)

    @staticmethod
    def Call(amount=0):
        return _Action('Call', amount)

    @staticmethod
    def Check(amount=0):
        return _Action('Check', amount)

    @staticmethod
    def Raise(amount=0):
        return _Action('Raise', amount)

    @staticmethod
    def AllIn(amount=0):
        return _Action('AllIn', amount)
