# import decimal
import random


class Hidraulico:

    # def __init__(self, nombre, v_min, v_max, tur_max, coef_energ, v_inicial=decimal.Decimal(0)):
    def __init__(self, nombre, v_min, v_max, tur_max, coef_energ, v_inicial=0):
        self._nombre = nombre
        self._v_inicial = v_inicial
        self._v_actual = v_inicial
        self._v_min = v_min
        self._v_max = v_max
        self._tur_max = tur_max
        self._coef_energ = coef_energ
        self._aporte = 0  # Aporte en Volumen

    @property
    def nombre(self):
        return self._nombre

    @nombre.setter
    def nombre(self, value):
        self._nombre = value

    @property
    def v_inicial(self):
        return self._v_inicial

    @v_inicial.setter
    def v_inicial(self, value):
        self._v_inicial = value

    @property
    def v_actual(self):
        return self._v_actual

    @v_actual.setter
    def v_actual(self, value):
        self._v_actual = value

    @property
    def v_min(self):
        return self._v_min

    @v_min.setter
    def v_min(self, value):
        self._v_min = value

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

    @property
    def aporte(self):
        return self._aporte

    @aporte.setter
    def aporte(self, value):
        self._aporte = value

    def reset(self, random_start=False):
        if random_start:
            self._v_actual = random.uniform(self._v_min, self._v_max)
        else:
            self._v_actual = self.v_inicial

    def actuar(self, tur, horas_paso):
        self._v_actual += self._aporte
        # self._v_actual -= decimal.Decimal(tur) * decimal.Decimal(horas_paso) * decimal.Decimal(0.0036)
        self._v_actual -= tur * horas_paso * 0.0036
        vertimiento = 0
        if self._v_actual > self._v_max:
            vertimiento = self._v_actual - self._v_max
            self._v_actual = self.v_max
        potencia = self._coef_energ * tur
        return potencia, vertimiento
