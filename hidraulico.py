import numpy as np


class Hidraulico:

    def __init__(self, name, v_actual, v_max, tur_max, coef_energ, v_inicial=0):
        self._name = name
        self._v_inicial = v_inicial
        self._v_actual = v_actual
        self._v_max = v_max
        self._tur_max = tur_max
        self._coef_energ = coef_energ

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def v_actual(self):
        return self._v_actual

    @v_actual.setter
    def v_actual(self, value):
        self._v_actual = value

    @property
    def v_max(self):
        return self._v_max

    @v_max.setter
    def v_max(self, value):
        self._v_max = value

    @property
    def tur_max(self):
        return self._tur_max

    @tur_max.setter
    def tur_max(self, value):
        self._tur_max = value

    @property
    def coef_energ(self):
        return self._coef_energ

    @coef_energ.setter
    def coef_energ(self, value):
        self._coef_energ = value

    def reset(self):
        self._v_actual = self.v_inicial

    def actuar(self, tur, horas_paso):
        self.v_actual += tur * horas_paso * 0.0036
        if self.v_actual > self._v_max:
            vertimiento = self. v_actual - self._v_max
            self.v_actual = self.v_max
        potencia = self._coef_energ * tur
        return potencia, vertimiento
