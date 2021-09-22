import pymongo
import random
import numpy as np


class Corrida:

    def __init__(self, linea_tiempo, hidraulicos, termicos, demanda):
        self._linea_tiempo = linea_tiempo
        self._hidraulicos = hidraulicos
        self._termicos = termicos
        self._demanda = demanda

    def reset(self):
        for hidro in self.hidraulicos.values():
            hidro.reset()
        self._linea_tiempo.reset()

    def step(self, action):
        demanda_a_cubrir = self._demanda
        for hidro, tur in action:
            potencia, _ = self._hidraulicos[hidro].actuar(np.clip(tur, 0, self.tur_max), self._linea_tiempo.horas_paso)
            demanda_a_cubrir -= potencia
        costo = 0
        for ter in self._termicos:
            costo += ter.costo * demanda_a_cubrir * self._linea_tiempo.horas_paso
        self._linea_tiempo += 1
        if self._linea_tiempo.paso_actual > self._linea_tiempo.total_pasos():
            return -costo, True
        self.sortearAportes()
        return -costo, False

    def sortearAportes(self):
        for nombre, hidraulico in self.hidraulicos:
            aporte = sortearAporte(self.linea_tiempo.paso_actual, nombre)
            hidraulico.v_actual += aporte

    sortearAportes()


def sortearAporte(semana, lago):
    mongo = pymongo.MongoClient("mongodb://localhost:27017/")

    db = mongo["prototipo"]
    coll = db["aportes"]

    listaAportes = list(coll.find({"semana": semana}, {"_id": 0, "semana": 0}))

    return random.choice(listaAportes)[lago]


# print(sortearAporte(42))
