import pymongo
import random
import math
import numpy as np
from mpi4py import MPI


class Environment:

    def __init__(self, linea_tiempo, hidraulicos, termicos, demanda, state_discretization, aportes, aportes_fijos={}, historico=False, anio_simulacion=None, generador=False, gen_aportes_fijos=-1, h_fijo=None):
        self._linea_tiempo = linea_tiempo
        self._hidraulicos = hidraulicos
        self._termicos = termicos
        self._termicos.sort(key=lambda t: t.costo)
        self._demanda = demanda
        self._state_discretization = state_discretization
        # self.get_lista_aportes()
        # self._lista_aportes = {}
        # self.sortearAportes()
        self._historico = historico
        self._generador = generador
        self._anio_actual = -1
        self._max_anios = 2000
        self._first_year = 0
        self._estado_hidrologico_actual = -1
        # self._lista_aportes = self.get_lista_aportes(hidraulicos.keys())
        self._lista_aportes = aportes
        self._aportes_fijos = aportes_fijos
        self._h_fijo = h_fijo
        self._train_dataset = False
        # print(f'{self._lista_aportes["bonete"][1909, 1]=}')
        # print(f'{len(self._lista_aportes)=}')
        self._stats_anios = {}
        self.anio_simulacion = anio_simulacion
        # self.DEBUG_volumen = np.zeros((110, 104))
        self.DEBUG_volumen = np.zeros((2000, 104))

        ############################################################################
        ################             GENERADOR APORTES              ################
        ############################################################################
        self.Q = np.array([[0.2, 0.8, 0, 0],
                           [0.2, 0.3, 0.5, 0],
                           [0, 0.5, 0.3, 0.2],
                           [0, 0, 0.8, 0.2]])

        self.M = np.zeros(self.Q.shape)

        for i in range(0, 4):
            self.M[i, 0] = self.Q[i, 0]
            for j in range(1, 4):
                self.M[i, j] = self.M[i, j - 1] + self.Q[i, j]

        self.Ehe = np.array([[23.1, 60.4, 80.5, 47.9],
                             [154.3, 403.2, 537.4, 320.2],
                             [308.6, 806.3, 1074.9, 640.4],
                             [780.5, 2039.7, 2719.0, 1620.0]])

        self._gen_aportes_fijos = {}
        if gen_aportes_fijos != -1:
            self._gen_aportes_fijos_cant = gen_aportes_fijos
            ######################## MPI ########################
            if MPI.COMM_WORLD.Get_rank() == 0:
                self._gen_aportes_fijos = self.generar_aportes_fijos(gen_aportes_fijos, self._h_fijo)
            self._gen_aportes_fijos = MPI.COMM_WORLD.bcast(self._gen_aportes_fijos)
            # print(f'PID: {MPI.COMM_WORLD.Get_rank()} Aportes: {self._gen_aportes_fijos[(0,1)]}')
            ######################## MPI ########################
        self._gen_fijos_anio_actual = -1
        ############################################################################
        ################             GENERADOR APORTES              ################
        ############################################################################

    def reset(self, random_start=False):
        # print(f'{self.anio_simulacion=}')
        for hidro in self._hidraulicos.values():
            hidro.reset(random_start)
        self._linea_tiempo.reset(random_start)
        # print('===================================')
        # print(f'{self._linea_tiempo.paso_actual=}')
        # print('===================================')
        self._anio_actual = -1
        self._gen_fijos_anio_actual = -1
        self._estado_hidrologico_actual = -1
        # self.state(True)
        self.sortearAportes()
        return self.state(), self.current_max_tur()

    def step(self, action):
        demanda_a_cubrir = self._demanda.getDemanda(self._linea_tiempo.paso_actual)
        # if demanda_a_cubrir == 0:
        #     print(f'{self._linea_tiempo.paso_actual=}')
        # print(f'{self._linea_tiempo.paso_actual=}')
        # demanda_a_cubrir = self._demanda.valor
        # for hidro, tur in [action]:
        for idx, tur in np.ndenumerate(action):
            # potencia, _ = self._hidraulicos[hidro].actuar(np.clip(tur, 0, self._hidraulicos[hidro].tur_max), self._linea_tiempo.horas_paso)
            if idx == ():
                idx = 0
            else:
                idx = idx[0]
            # potencia, _ = list(self._hidraulicos.values())[idx].actuar(np.clip(tur, 0, list(self._hidraulicos.values())[idx].tur_max), self._linea_tiempo.horas_paso)
            potencia, _ = list(self._hidraulicos.values())[idx].actuar(tur, self._linea_tiempo.horas_paso)
            demanda_a_cubrir -= potencia

        # print(f'asd')
        # print(f'{self=}')

        costo = 0
        # generalizar termicos
        for ter in self._termicos:
            dem = min(demanda_a_cubrir, ter.pot_max)
            # print(f'{ter.nombre=} {dem=:.1f}', end=' ')
            costo += ter.costo * dem * self._linea_tiempo.horas_paso
            demanda_a_cubrir -= dem
        self._linea_tiempo.paso_actual += 1
        # print(f'*')

        reward = -costo
        # reward = 1/costo
        # reward = costo
        # reward = 1000000000 - costo
        # print(type(costo))
        if self._linea_tiempo.paso_actual >= self._linea_tiempo.total_pasos:
            # print(f'==========EPISODE_END==========')
            # return self.state(), reward, True
            return self.state(), reward, True, self.current_max_tur()
        self.sortearAportes()
        # print(f'asd2')
        # print(f'{self=}')
        # return self.state(), reward, False
        return self.state(), reward, False, self.current_max_tur()
        # state, volume_state = self.state()
        # return state, volume_state, reward, False, self.current_max_tur()

    # def sortearAportes(self):
    #     semana = self._linea_tiempo.paso_actual % self._linea_tiempo.pasos_por_anio + 1
    #     if semana not in self._lista_aportes:
    #         self._lista_aportes[semana] = self.get_lista_aportes(semana, self._hidraulicos.keys())
    #     for nombre, hidraulico in self._hidraulicos.items():
    #         with decimal.localcontext(create_decimal128_context()):
    #             aporte = random.choice(self._lista_aportes[semana][nombre])
    #             # hidraulico.v_actual += aporte.to_decimal() * decimal.Decimal(self._linea_tiempo.horas_paso) * decimal.Decimal(0.0036)
    #             hidraulico.aporte = aporte.to_decimal() * decimal.Decimal(self._linea_tiempo.horas_paso) * decimal.Decimal(0.0036)
    #             # hidraulico.aporte = decimal.Decimal(681) * decimal.Decimal(self._linea_tiempo.horas_paso) * decimal.Decimal(0.0036)

    def sortearAportes(self):
        semana = self._linea_tiempo.paso_actual % self._linea_tiempo.pasos_por_anio + 1
        # print(f'{self._linea_tiempo.paso_actual=}')
        # print(f'{semana=}')
        # print(f'{self.anio_simulacion=}')
        for nombre, hidraulico in self._hidraulicos.items():
            if self._linea_tiempo.paso_actual in self._aportes_fijos:
                aporte = self._aportes_fijos[self._linea_tiempo.paso_actual]
            elif self._historico or self.anio_simulacion is not None:
                aportes_dataset = self._gen_aportes_fijos if self._train_dataset else self._lista_aportes[nombre]
                if semana == 1 and self._anio_actual != -1:
                    self._anio_actual += 1
                    # if self._anio_actual == 2019:
                    #     self._anio_actual = 1909
                    if self._anio_actual == self._max_anios:
                        self._anio_actual = self._first_year
                if self._anio_actual == -1:
                    if self.anio_simulacion is not None:
                        self._anio_actual = self.anio_simulacion
                    else:
                        # self._anio_actual = random.choice(list(self._lista_aportes[nombre].keys()))[0]
                        # self._anio_actual = random.choice(filter(lambda anio: anio % 2 == 0, list(self._lista_aportes[nombre].keys())))[0] # sin repetir anios
                        # modo sin repetir anios
                        # anio par si es el primero, impar si es el segundo
                        if self._linea_tiempo.paso_actual <= 51:
                            # self._anio_actual = random.choice(list(filter(lambda anio: anio[0] % 2 == 0, list(self._lista_aportes[nombre].keys()))))[0]
                            self._anio_actual = random.choice(list(filter(lambda anio: anio[0] % 2 == 0, list(aportes_dataset.keys()))))[0]
                        else:
                            # self._anio_actual = random.choice(list(filter(lambda anio: anio[0] % 2 != 0, list(self._lista_aportes[nombre].keys()))))[0]
                            self._anio_actual = random.choice(list(filter(lambda anio: anio[0] % 2 != 0, list(aportes_dataset.keys()))))[0]
                    # stats anios #
                    # if self._anio_actual in self._stats_anios:
                    #     self._stats_anios[self._anio_actual] += 1
                    # else:
                    #     self._stats_anios[self._anio_actual] = 1
                    # stats anios #
                # print(f'{self._anio_actual=}')
                # aporte = self._lista_aportes[nombre][self._anio_actual, semana]
                aporte = aportes_dataset[self._anio_actual, semana]
            elif self._generador:
                # if self._anio_actual == -1:
                #     if self.anio_simulacion is not None:
                #         self._anio_actual = self.anio_simulacion
                estacion = math.floor((semana - 1) / 13)
                if self._estado_hidrologico_actual == -1:
                    # self._estado_hidrologico_actual = random.randint(0, 3)
                    self._estado_hidrologico_actual = self._h_fijo  # MS
                elif semana == 1:
                    self._estado_hidrologico_actual = self.newH(self._estado_hidrologico_actual)
                # else:
                #     self._estado_hidrologico_actual = self.newH(self._estado_hidrologico_actual)
                # aporte = self.generar_aporte(self._estado_hidrologico_actual, estacion)
                aporte = [154.3, 403.2, 537.4, 320.2][estacion]  # PRUEBA APORTE MEDIO
            elif len(self._gen_aportes_fijos) > 0:
                # if self._anio_actual == -1:
                #     if self.anio_simulacion is not None:
                #         self._anio_actual = self.anio_simulacion
                if self._gen_fijos_anio_actual == -1:
                    # self._gen_fijos_anio_actual = random.randint(0, self._gen_aportes_fijos_cant - 1)
                    # modo sin repetir anios
                    # anio par si es el primero, impar si es el segundo
                    if self._linea_tiempo.paso_actual <= 51:
                        self._gen_fijos_anio_actual = random.choice(list(filter(lambda anio: anio % 2 == 0, list(range(0, self._gen_aportes_fijos_cant)))))
                    else:
                        self._gen_fijos_anio_actual = random.choice(list(filter(lambda anio: anio % 2 != 0, list(range(0, self._gen_aportes_fijos_cant)))))

                # else:
                #     self._gen_fijos_anio_actual += 1
                #     self._gen_fijos_anio_actual %= self._gen_aportes_fijos_cant
                elif semana == 1:
                    self._gen_fijos_anio_actual += 1
                    self._gen_fijos_anio_actual %= self._gen_aportes_fijos_cant
                # print(f'{self._gen_fijos_anio_actual=}')
                aporte = self._gen_aportes_fijos[(self._gen_fijos_anio_actual, semana)]
            else:
                aporte = random.choice([self._lista_aportes[nombre][anio, sem] for anio, sem in self._lista_aportes[nombre] if sem == semana])
            # hidraulico.aporte = aporte.to_decimal() * decimal.Decimal(self._linea_tiempo.horas_paso) * decimal.Decimal(0.0036)
            hidraulico.aporte = aporte * self._linea_tiempo.horas_paso * 0.0036
            # hidraulico.aporte = .5 * aporte * self._linea_tiempo.horas_paso * 0.0036  # %50 aportes
            # hidraulico.aporte = 0 * aporte * self._linea_tiempo.horas_paso * 0.0036  # %0 aportes
            # if hidraulico.aporte == 0:
            #     print(f'{self._linea_tiempo.paso_actual=}')
            # print(f'======================================================')

    # def state(self, inicial=False):
    #     if inicial:
    #         l = list(map(lambda x: x.v_inicial, self._hidraulicos.values()))
    #     else:
    #         l = list(map(lambda x: x.v_actual, self._hidraulicos.values()))
    #     l.append(decimal.Decimal(self._linea_tiempo.paso_actual))
    #     return self._state_discretization.state(l)

    def state(self):
        if self._linea_tiempo.paso_actual == self._linea_tiempo.total_pasos:
            return None
        volume_state = list(map(lambda x: x.v_actual, self._hidraulicos.values()))
        # volume_state.append(decimal.Decimal(self._linea_tiempo.paso_actual))
        volume_state.append(self._linea_tiempo.paso_actual)
        # print(f't: {self._linea_tiempo.paso_actual}')
        # print(f'{volume_state[0]:.1f}', end='\t')
        if self.anio_simulacion is not None:  # test mode
            # self.DEBUG_volumen[self._anio_actual - 1909, volume_state[1]] = volume_state[0]
            self.DEBUG_volumen[self._anio_actual, volume_state[1]] = volume_state[0]
        return self._state_discretization.state(volume_state)
        # return self._state_discretization.state(volume_state), volume_state

    def current_max_tur(self):
        # return list(map(lambda x: min(decimal.Decimal(x._tur_max), (x.v_actual + x.aporte) / (decimal.Decimal(0.0036) * self._linea_tiempo.horas_paso)), self._hidraulicos.values()))
        return list(map(lambda x: min(x._tur_max, (x.v_actual + x.aporte) / (0.0036 * self._linea_tiempo.horas_paso)), self._hidraulicos.values()))

    def __repr__(self):
        res = ""
        for nombre, hidro in self._hidraulicos.items():
            # res += 'Hidro: ' + nombre + ' Volumen: ' + str(hidro.v_actual) + ',' + ' Aporte: ' + str(hidro.aporte)
            res += f'Hidro: {nombre} Volumen: {hidro.v_actual:.2f}, Aporte: {hidro.aporte:.2f}'
        return res

# def sortearAporte(semana, lago):
#     mongo = pymongo.MongoClient("mongodb://localhost:27017/")

#     db = mongo["prototipo"]
#     coll = db["aportes"]

#     listaAportes = list(coll.find({"semana": semana}, {"_id": 0, "semana": 0}))
#     return random.choice(listaAportes)[lago]

    # def get_lista_aportes(self, semana, lagos):

    #     mongo = pymongo.MongoClient("mongodb://localhost:27017/", connect=False)

    #     db = mongo["prototipo"]
    #     coll = db["aportes"]

    #     listaAportes = list(coll.find({"semana": semana}, {"_id": 0, "semana": 0}))
    #     mongo.close()

    #     res = {}
    #     for lago in lagos:
    #         res[lago] = [elem[lago] for elem in listaAportes]
    #     return res

    def get_lista_aportes(self, lagos):

        mongo = pymongo.MongoClient("mongodb://localhost:27017/", connect=False)

        db = mongo["prototipo"]
        coll = db["aportesConAnio"]

        listaAportes = list(coll.find({}, {"_id": 0}))
        mongo.close()

        # print(f'{listaAportes=}')

        res = {}
        for lago in lagos:
            res[lago] = {}
            for elem in listaAportes:
                res[lago][elem["anio"], elem["semana"]] = elem[lago]
        return res

    ############################################################################
    ################             GENERADOR APORTES              ################
    ############################################################################
    def newH(self, h):
        r = random.random()
        return np.where(r < self.M[h, :])[0][0]

    # def generar_aporte(self, h, e):
    def generar_aporte(self, esp):
        r = random.random()
        y = 1
        for i in range(10):
            y = y + ((1 + y) * math.exp(-y) + r - 1) / (y * math.exp(-y))
        return esp * y / 2
        # return self.Ehe[h, e] * y / 2

    def generar_aportes_fijos(self, N, h_fijo=None):
        res = {}
        if h_fijo is not None:
            h = h_fijo
        else:
            h = random.randint(0, 3)
        # e = 0
        # mean_sem = {}
        for anio in range(N):
            for semana in range(52):
                # res[(anio, semana + 1)] = self.generar_aporte(h, e)
                # res[(anio, semana + 1)] = self.generar_aporte(h, semana // 13)
                res[(anio, semana + 1)] = self.generar_aporte(self.Ehe[h, semana // 13])
                # print(f'{semana=} {semana // 13=}')
                # if semana in mean_sem:
                #     mean_sem[semana] += res[(anio, semana + 1)]
                # else:
                #     mean_sem[semana] = res[(anio, semana + 1)]
            if h_fijo is None:
                h = self.newH(h)
        # print(f'{mean_sem=}')
        return res

    ############################################################################
    ################             GENERADOR APORTES              ################
    ############################################################################

# print(get_lista_aportes(10, ['bonete']))
