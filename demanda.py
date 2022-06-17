class Demanda:

    def __init__(self, valor, valores_especificos={}):
        self._valor = valor
        self._valores_especificos = valores_especificos

    @property
    def valor(self):
        return self._valor

    @valor.setter
    def valor(self, value):
        self._valor = value

    def getDemanda(self, paso):
        if paso in self._valores_especificos:
            return self._valores_especificos[paso]
        return self._valor
