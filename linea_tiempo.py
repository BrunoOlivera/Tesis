import constantes
import random


class LineaTiempo:

    def __init__(self, tipo_paso, anio_ini, anio_fin, paso_actual=0):
        self._tipo_paso = tipo_paso
        self._paso_actual = paso_actual
        self._anio_ini = anio_ini
        self._anio_fin = anio_fin
        self._pasos_por_naio = self.pasosPorAnio()
        self._total_pasos = self.totalPasos()
        self._horas_paso = self.horasPaso()

    @property
    def tipo_paso(self):
        return self._tipo_paso

    @tipo_paso.setter
    def tipo_paso(self, value):
        self._tipo_paso = value

    @property
    def paso_actual(self):
        return self._paso_actual

    @paso_actual.setter
    def paso_actual(self, value):
        self._paso_actual = value

    @property
    def anio_ini(self):
        return self._anio_ini

    @anio_ini.setter
    def anio_ini(self, value):
        self._anio_ini = value

    @property
    def anio_fin(self):
        return self._anio_fin

    @anio_fin.setter
    def anio_fin(self, value):
        self._anio_fin = value

    @property
    def pasos_por_anio(self):
        return self._pasos_por_naio

    @property
    def total_pasos(self):
        return self._total_pasos

    @property
    def horas_paso(self):
        return self._horas_paso

    def reset(self, random_start=False):
        if random_start:
            self._paso_actual = random.randint(0, 103)  # TODO: generalizar
        else:
            self._paso_actual = 0

    def pasosPorAnio(self):
        if self._tipo_paso == constantes.PASO_SEMANAL:
            return 52
        if self._tipo_paso == constantes.PASO_DIARIO:
            return 365
        if self._tipo_paso == constantes.PASO_HORARIO:
            return 365 * 24

    def totalPasos(self):
        cantAnios = (self._anio_fin - self._anio_ini) + 1
        if self._tipo_paso == constantes.PASO_SEMANAL:
            return 52 * cantAnios
        if self._tipo_paso == constantes.PASO_DIARIO:
            return 365 * cantAnios  # anios bisiestos
        if self._tipo_paso == constantes.PASO_HORARIO:
            return 365 * 24 * cantAnios

    def horasPaso(self):
        if self._tipo_paso == constantes.PASO_SEMANAL:
            return 168
        if self._tipo_paso == constantes.PASO_DIARIO:
            return 24
        if self._tipo_paso == constantes.PASO_HORARIO:
            return 1
