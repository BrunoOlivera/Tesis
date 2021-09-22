import constantes


class LineaTiempo:

    def __init__(self, tipo_paso, anio_ini, anio_fin, paso_actual=1):
        self._tipo_paso = tipo_paso
        self._paso_actual = paso_actual
        self._anio_ini = anio_ini
        self._anio_fin = anio_fin

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
    def total_pasos(self):
        cantAnios = (self._anio_fin - self._anio_ini) + 1
        if self._tipo_paso == constantes.PASO_SEMANAL:
            return 52 * cantAnios
        if self._tipo_paso == constantes.PASO_DIARIO:
            return 365 * cantAnios  # anios bisiestos
        if self._tipo_paso == constantes.PASO_HORARIO:
            return 365 * 24 * cantAnios

    @property
    def hora_paso(self):
        if self._tipo_paso == constantes.PASO_SEMANAL:
            return 168
        if self._tipo_paso == constantes.PASO_DIARIO:
            return 24
        if self._tipo_paso == constantes.PASO_HORARIO:
            return 1

    def reset(self):
        self._paso_actual = 1
