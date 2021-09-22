class Termico:

    def __init__(self, pot_max, costo):
        self._pot_max = pot_max
        self._costo = costo

    @property
    def pot_max(self):
        return self._pot_max

    @pot_max.setter
    def pot_max(self, value):
        self._pot_max = value

    @property
    def costo(self):
        return self._costo

    @costo.setter
    def costo(self, value):
        self._costo = value
