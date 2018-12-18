class EWMA:
    def __init__(self, gamma):
        self._value = None
        self.gamma = gamma

    def update(self, x):

        if self._value is None:
            self._value = x
        else:
            self._value *= self.gamma
            self._value += (1.0 - self.gamma)* x

    def value(self):
        return self._value
