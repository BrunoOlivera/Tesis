class Termico:

    def __init__(self, nombre, pot_max, costo):
        self._nombre = nombre
        self._pot_max = pot_max
        self._costo = costo

    @property
    def nombre(self):
        return self._nombre

    @nombre.setter
    def nombre(self, value):
        self._nombre = value

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
