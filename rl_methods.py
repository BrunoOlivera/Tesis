import numpy as np
import time
import copy
from mpi4py import MPI


class ActorCritic:

    def __init__(self, policy, state_space, policy_step_size, value_step_size, gamma=0.99):
        self.policy = policy
        self.state_space = state_space
        self.policy_step_size = policy_step_size
        self.value_step_size = value_step_size
        self.gamma = gamma

    def act(self, state_disc, state_cont):
        return self.policy.act(state_disc, state_cont)

    # GRID
    # def gradient_step(self, state, action, reward, next_state, time):
    #     # delta = reward + decimal.Decimal(self.gamma) * decimal.Decimal(self.state_space.value[np.array(next_state, dtype=bool)][0]) - decimal.Decimal(self.state_space.value[np.array(state, dtype=bool)][0])
    #     if next_state is None:
    #         delta = reward - self.state_space.value[np.array(state, dtype=bool)][0]
    #         # if abs(delta) > 4_200_000:
    #         #     print(f'{np.where(state==1)=} {reward=:.1f} state_value={self.state_space.value[np.array(state, dtype=bool)][0]:.1f} next_sate_value={next_state} {delta=:.1f}')
    #     else:
    #         delta = reward + self.gamma * self.state_space.value[np.array(next_state, dtype=bool)][0] - self.state_space.value[np.array(state, dtype=bool)][0]
    #         # if abs(delta) > 4_200_000:
    #         #     print(f'{np.where(state==1)=} {reward=:.1f} state_value={self.state_space.value[np.array(state, dtype=bool)][0]:.1f} next_sate_value={self.state_space.value[np.array(next_state, dtype=bool)][0]:.1f} {delta=:.1f}')
    #     # delta = reward - decimal.Decimal(self.gamma) * decimal.Decimal(self.state_space.value[np.array(next_state, dtype=bool)][0]) + decimal.Decimal(self.state_space.value[np.array(state, dtype=bool)][0])
    #     changed = self.policy.gradient_step(state, action, delta, self.gamma, self.policy_step_size, time)
    #     if changed:
    #         self.state_space.gradient_step(state, delta, self.value_step_size)
    #     # return self.policy.gradient_step(state, action, delta, self.gamma, self.policy_step_size, time)
    #     return delta

    # RBF
    def gradient_step(self, state, action, reward, next_state, time):
        # delta = reward + decimal.Decimal(self.gamma) * decimal.Decimal(self.state_space.value[np.array(next_state, dtype=bool)][0]) - decimal.Decimal(self.state_space.value[np.array(state, dtype=bool)][0])
        if next_state is None:
            delta = reward - self.state_space.value(state)
            # if abs(delta) > 4_200_000:
            #     print(f'{np.where(state==1)=} {reward=:.1f} state_value={self.state_space.value[np.array(state, dtype=bool)][0]:.1f} next_sate_value={next_state} {delta=:.1f}')
        else:
            delta = reward + self.gamma * self.state_space.value(next_state) - self.state_space.value(state)
            # if abs(delta) > 4_200_000:
            #     print(f'{np.where(state==1)=} {reward=:.1f} state_value={self.state_space.value[np.array(state, dtype=bool)][0]:.1f} next_sate_value={self.state_space.value[np.array(next_state, dtype=bool)][0]:.1f} {delta=:.1f}')
        # delta = reward - decimal.Decimal(self.gamma) * decimal.Decimal(self.state_space.value[np.array(next_state, dtype=bool)][0]) + decimal.Decimal(self.state_space.value[np.array(state, dtype=bool)][0])
        # print(f'{delta=}')
        changed = self.policy.gradient_step(state, action, delta, self.gamma, self.policy_step_size, time)
        if changed:
            self.state_space.gradient_step(state, delta, self.value_step_size)
        # return self.policy.gradient_step(state, action, delta, self.gamma, self.policy_step_size, time)
        return delta



# class MonteCarlo:

#     def __init__(self, gamma):
#         self.gamma = gamma


# GRID
# def train(env, model, num_episodes=1000, verbose=0, exploring_start=False):
#     train_start_time = time.time()
#     rewards = []
#     print_interval = 10
#     rewardsARetornar = []
#     # expected_value = [-model.state_space.value[0][0]]
#     # expected_value = [model.policy._mean_approximation._mean_discretization[936]]
#     expected_value = []
#     # best_result = 436_000_000
#     # best_result = 1_310_400_000  # dos termicos
#     best_result = 611_520_000  # dos termicos
#     best_model = np.zeros_like(model.policy._mean_approximation._mean_discretization)
#     # best_unclipped_model = np.zeros_like(model.policy._mean_approximation._mean_discretization)
#     best_episode = 0
#     best_value = np.zeros_like(model.state_space.value)
#     exploring_start = True
#     # exploring_start = False

#     print(f'{model.policy_step_size=}')
#     print(f'{model.value_step_size=}')
#     print(f'{model.gamma=}')
#     print(f'{model.policy.sigma=}')
#     init_mean = model.policy._mean_approximation._mean_discretization[936]
#     print(f'init_lago={env._hidraulicos["bonete"].v_inicial}')
#     print(f'{init_mean=:.2f}')
#     # print(f'{model.policy._mean_approximation._mean_discretization.shape}')
#     print(f'{exploring_start=}')
#     print(f'======================================================')

#     # -- DESHARDCODEAR -- #
#     # qq = np.zeros((10, 104))
#     # init_cost = [(qq.shape[-1] - v) * -5_880_000 for v in range(qq.shape[-1])]
#     # # init_cost = [(qq.shape[-1] - v) * -12_600_000 for v in range(qq.shape[-1])]
#     # # init_cost = [(qq.shape[-1] - v) * -4_200_000 for v in range(qq.shape[-1])]
#     # # init_cost = [(qq.shape[-1] - v) * -4.2 for v in range(qq.shape[-1])]
#     # for x in range(len(init_cost)):
#     #     # qq[:, x] = decimal.Decimal(init_cost[x])
#     #     qq[:, x] = init_cost[x]
#     # -- DESHARDCODEAR -- #
#     # rr = np.zeros((10, 104))
#     for n in range(num_episodes + 1):
#         # state, current_max_turs = env.reset(True)  # EXPLORING STARTS
#         # state, current_max_turs = env.reset()
#         state, current_max_turs = env.reset(exploring_start)
#         done = False
#         t = -1
#         if verbose == 1:
#             print(f'Episodio {n}:')
#             print(f'-------------')
#         while not done:
#             t += 1  # TIME EN EXPLORING STARTS
#             # print(f'{np.where(state != 0)}')
#             # rr[state != 0] += 1
#             action = model.act(state, current_max_turs)

#             #################################################
#             #################################################
#             # test_env = copy.deepcopy(env)
#             # test_model = copy.deepcopy(model)
#             # # print(f'Env Before:{env._linea_tiempo.paso_actual}')
#             # # print(f'Test Env Before:{test_env._linea_tiempo.paso_actual}')
#             # # print(f'Test Model Before: {test_model.policy._mean_approximation.mean[test_model.policy._mean_approximation.getIndex(state)]}')
#             # # print(f'Model Before: {model.policy._mean_approximation.mean[model.policy._mean_approximation.getIndex(state)]}')
#             # # print(f'{action=}')
#             # test_model.policy._mean_approximation.mean[test_model.policy._mean_approximation.getIndex(state)] = action
#             # # reward = -611_520_000 + test(test_env, test_model)  # 0 init means
#             # costo = test(test_env, test_model)
#             # if costo < best_result:
#             #     best_result = costo
#             #     best_model = np.copy(model.policy._mean_approximation._mean_discretization)
#             #     best_unclipped_model = np.copy(model.policy._mean_approximation._unclipped_mean)
#             #     best_episode = n
#             #     best_value = np.copy(model.state_space.value)
#             # expected_value.append(costo)
#             # reward = -costo / 104  # 0 init means
#             # # reward = -test(test_env, test_model) * ((104 - t) / 104)  # 0 init means
#             # # reward = -269_464_557.5 + test(test_env, test_model)  # 680 init means
#             # # print(f'{reward=:.2f}')
#             # # print(f'Env After:{env._linea_tiempo.paso_actual}')
#             # # print(f'Test Env After:{test_env._linea_tiempo.paso_actual}')
#             # # print(f'Model After: {model.policy._mean_approximation.mean[model.policy._mean_approximation.getIndex(state)]}')
#             # # print(f'Test Model After: {test_model.policy._mean_approximation.mean[test_model.policy._mean_approximation.getIndex(state)]}')
#             # # print('===========================================')
#             # next_state, _, done, current_max_turs = env.step(action)
#             #################################################
#             #################################################

#             # print(f't0: {np.where(state==1)}')
#             next_state, reward, done, current_max_turs = env.step(action)
#             # print(f't1: {np.where(state==1)}')
#             # if next_state is None:
#             #     print(f'None')
#             # else:
#             #     print(f't1: {np.where(next_state==1)}')
#             if verbose == 1:
#                 print(f'Semana: {t}')
#                 print(f'Action: {action[0]:.1f} Reward: {reward:.1f}')
#             step = model.gradient_step(state, action, reward, next_state, t)
#             # expected_value.append(step)
#             rewards.append(reward)
#             state = next_state
#         # rewardsARetornar.append(np.sum(rewards) / print_interval)
#         # rewardsARetornar.append(np.sum(rewards))
#         # expected_value.append(-model.state_space.value[0][0])  # vacio
#         # expected_value.append(-model.state_space.value[9][0])  # lleno
#         # expected_value.append(model.policy._mean_approximation._mean_discretization[936])

#         ###################################################
#         # if n == 1_000_000:
#         #     model.policy.epsilon /= 2
#         # if n == 2_000_000:
#         #     model.policy.epsilon /= 2
#         # if n == 3_000_000:
#         #     model.policy.epsilon /= 2
#         # if n == 4_000_000:
#         #     model.policy.epsilon /= 2
#         # if n == 4_500_000:
#         #     model.policy.epsilon = 0
#         ###################################################

#         if (n % print_interval == 0 and n > 0) or (n == 0) or (n == num_episodes - 1):
#             # if n >= 0:
#             # if n >= 950_000 or n % 100_000 == 0:
#             # if n >= 9_750_000 or n % 25_000 == 0:
#             # if n > 500_000 and n % 50 == 0:
#             if (n < 2_500_000 and n % 5000 == 0) or (n > 2_500_000 and n < 4_500_000 and n % 250 == 0) or (n > 4_500_000 and n % 10 == 0):
#                 costo = test(env, model)
#                 model.policy.train = True
#                 env.anio_simulacion = None
#                 if costo < best_result:
#                     best_result = costo
#                     best_model = np.copy(model.policy._mean_approximation._mean_discretization)
#                     # best_unclipped_model = np.copy(model.policy._mean_approximation._unclipped_mean)
#                     best_episode = n
#                     best_value = np.copy(model.state_space.value)
#                 expected_value.append(costo)  # lleno

#             # print("Episode %d: %.1f Time: %.2fs VS: %.2f%%" % (n, -model.state_space.value[0][0], time.time() - train_start_time, 100 * np.sum(model.state_space.value != qq) / model.state_space.value.size))
#             # print(f'{-model.state_space.value[0][0]:.1f}')
#             # print(f'Episode {n}: Value: {-model.state_space.value[0][0]:.1f} Mean: {model.policy._mean_approximation._mean_discretization[0]:.2f}')
#             # print(f'Episode {n}: Value: {-model.state_space.value[9][0]:.1f} Mean: {model.policy._mean_approximation._mean_discretization[9]:.2f}')
#             # print(f'Episode {n}: Value: {-model.state_space.value[9][0]:.1f} Mean: {model.policy._mean_approximation._mean_discretization[np.where(model.policy._mean_approximation._mean_discretization!=680)]}')
#             # print(f'Episode {n}: Value: {-model.state_space.value[9][0]:.1f} Time: {time.time() - train_start_time:.2f}')
#             # print(f'Episode {n}: Value: {-model.state_space.value[9][0]:.1f} VS: {100 * np.sum(model.state_space.value != qq) / model.state_space.value.size:.2f}%')
#             print(f'Episode {n}: Value: {-model.state_space.value[9][0]:.1f} Mean: {model.policy._mean_approximation._mean_discretization[939]:.1f} Mean: {model.policy._mean_approximation._mean_discretization[940]:.1f} Mean: {model.policy._mean_approximation._mean_discretization[944]:.1f} Mean: {model.policy._mean_approximation._mean_discretization[945]:.1f} Costo: {best_result:.1f} Time: {time.time() - train_start_time:.1f}')
#             # print(f'Episode {n}: Value: {-model.state_space.value[99][0]:.1f} Mean: {model.policy._mean_approximation._mean_discretization[939]:.1f} Mean: {model.policy._mean_approximation._mean_discretization[940]:.1f} Mean: {model.policy._mean_approximation._mean_discretization[944]:.1f} Mean: {model.policy._mean_approximation._mean_discretization[945]:.1f} Costo: {best_result:.1f} Time: {time.time() - train_start_time:.1f}')
#             pass
#         rewards = []
#     # print("Total Train Time: %.2fs" % (time.time() - train_start_time))
#     print(f'Total Train Time: {time.time() - train_start_time:.2f}s')
#     # print(f'Stats Anios: {env._stats_anios}')
#     print(f'======================================================')
#     print(f'{model.policy_step_size=}')
#     print(f'{model.value_step_size=}')
#     print(f'{model.gamma=}')
#     print(f'{model.policy.sigma=}')
#     print(f'init_lago={env._hidraulicos["bonete"].v_inicial}')
#     print(f'{init_mean=:.2f}')
#     print(f'{exploring_start=}')
#     print(f'======================================================')
#     # print(f'means_9: {model.policy._mean_approximation._mean_discretization[936:]}')
#     # print(f'means_8: {model.policy._mean_approximation._mean_discretization[832:936]}')
#     # print(f'means_7: {model.policy._mean_approximation._mean_discretization[728:832]}')
#     # print(f'eps: {(model.policy.eps / model.policy.total) * 100:.2f}%')
#     np.set_printoptions(threshold=np.inf)
#     # print(f'{rr}')
#     # print(f'{-model.state_space.value}')
#     # print(f'{np.where(model.state_space.value < qq)}')
#     # return rewardsARetornar
#     # print(f'{best_model=}')
#     # print(f'{best_unclipped_model=}')
#     # print(f'{best_value=}')
#     print(f'{best_episode=}')
#     print(f'{best_result=:.1f}')
#     return expected_value
#     # return best_model, best_value


# RBF
def train(env, model, num_episodes=1000, verbose=0, exploring_start=False):
    train_start_time = time.time()
    rewards = []
    # print_interval = 10
    print_interval = 100
    rewardsARetornar = []
    # expected_value = [-model.state_space.value[0][0]]
    # expected_value = [model.policy._mean_approximation._mean_discretization[936]]
    expected_value = []
    # best_result = 611_520_000  # dos termicos
    # best_result = 98_560_000  # dos termicos, nuevo modelo, RBF
    # best_result = 350_084_468  # dos termicos, nuevo modelo, RBF
    # best_result = 1_135_680_000  # Termico('t_barato', 250, 100),Termico('t1_caro', 100, 400)
    best_result = 7_425_600_000  # Termico('t_barato', 250, 100),Termico('t1_caro', 100, 4000) #FALLA
    best_model = np.zeros_like(model.policy._mean_approximation.mean_weights)
    # best_unclipped_model = np.zeros_like(model.policy._mean_approximation.mean_weights)
    best_episode = 0
    best_value = np.zeros_like(model.state_space.value)
    exploring_start = True
    # exploring_start = False

    # init_state_tensor = model.state_space.state([8200, 0])
    init_state_tensor = model.state_space.state([4100, 0])  # TODO: gen

    print(f'{model.policy_step_size=}')
    print(f'{model.value_step_size=}')
    print(f'{model.gamma=}')
    print(f'{model.policy.sigma=}')
    init_mean = model.policy._mean_approximation.mean_weights[0][0]
    print(f'init_lago={env._hidraulicos["bonete"].v_inicial}')
    print(f'{init_mean=:.2f}')
    # print(f'{model.policy._mean_approximation.mean_weights.shape}')
    print(f'{exploring_start=}')
    print(f'{model.state_space.sigma=}')
    print(f'======================================================')

    # rr = np.zeros((10, 104))
    for n in range(num_episodes + 1):
        state, current_max_turs = env.reset(exploring_start)
        init_state = np.copy(state)
        done = False
        t = -1
        if verbose == 1:
            print(f'Episodio {n}:')
            print(f'-------------')
        while not done:
            t += 1  # TIME EN EXPLORING STARTS
            # print(f'{np.where(state != 0)}')
            # rr[state != 0] += 1
            action = model.act(state, current_max_turs)

            #################################################
            #################################################
            # test_env = copy.deepcopy(env)
            # test_model = copy.deepcopy(model)
            # # print(f'Env Before:{env._linea_tiempo.paso_actual}')
            # # print(f'Test Env Before:{test_env._linea_tiempo.paso_actual}')
            # # print(f'Test Model Before: {test_model.policy._mean_approximation.mean[test_model.policy._mean_approximation.getIndex(state)]}')
            # # print(f'Model Before: {model.policy._mean_approximation.mean[model.policy._mean_approximation.getIndex(state)]}')
            # # print(f'{action=}')
            # test_model.policy._mean_approximation.mean[test_model.policy._mean_approximation.getIndex(state)] = action
            # # reward = -611_520_000 + test(test_env, test_model)  # 0 init means
            # costo = test(test_env, test_model)
            # if costo < best_result:
            #     best_result = costo
            #     best_model = np.copy(model.policy._mean_approximation.mean_weights)
            #     best_unclipped_model = np.copy(model.policy._mean_approximation._unclipped_mean)
            #     best_episode = n
            #     best_value = np.copy(model.state_space.value)
            # expected_value.append(costo)
            # reward = -costo / 104  # 0 init means
            # # reward = -test(test_env, test_model) * ((104 - t) / 104)  # 0 init means
            # # reward = -269_464_557.5 + test(test_env, test_model)  # 680 init means
            # # print(f'{reward=:.2f}')
            # # print(f'Env After:{env._linea_tiempo.paso_actual}')
            # # print(f'Test Env After:{test_env._linea_tiempo.paso_actual}')
            # # print(f'Model After: {model.policy._mean_approximation.mean[model.policy._mean_approximation.getIndex(state)]}')
            # # print(f'Test Model After: {test_model.policy._mean_approximation.mean[test_model.policy._mean_approximation.getIndex(state)]}')
            # # print('===========================================')
            # next_state, _, done, current_max_turs = env.step(action)
            #################################################
            #################################################

            # print(f't0: {np.where(state==1)}')
            next_state, reward, done, current_max_turs = env.step(action)
            # print(f't1: {np.where(state==1)}')
            # if next_state is None:
            #     print(f'None')
            # else:
            #     print(f't1: {np.where(next_state==1)}')
            if verbose == 1:
                print(f'Semana: {t}')
                print(f'Action: {action[0]:.1f} Reward: {reward:.1f}')
            step = model.gradient_step(state, action, reward, next_state, t)
            # expected_value.append(step)
            rewards.append(reward)
            state = next_state
        # rewardsARetornar.append(np.sum(rewards) / print_interval)
        # rewardsARetornar.append(np.sum(rewards))
        # expected_value.append(-model.state_space.value[0][0])  # vacio
        # expected_value.append(-model.state_space.value[9][0])  # lleno
        # expected_value.append(model.policy._mean_approximation.mean_weights[936])

        ###################################################
        # if n == 1_000_000:
        #     model.policy.epsilon /= 2
        # if n == 2_000_000:
        #     model.policy.epsilon /= 2
        # if n == 3_000_000:
        #     model.policy.epsilon /= 2
        # if n == 4_000_000:
        #     model.policy.epsilon /= 2
        # if n == 4_500_000:
        #     model.policy.epsilon = 0
        ###################################################

        if (n % print_interval == 0 and n > 0) or (n == 0) or (n == num_episodes - 1):
            # if n >= 0:
            # if n >= 950_000 or n % 100_000 == 0:
            # if n >= 9_750_000 or n % 25_000 == 0:
            # if n > 500_000 and n % 50 == 0:
            # if (n < 2_500_000 and n % 5000 == 0) or (n > 2_500_000 and n < 4_500_000 and n % 250 == 0) or (n > 4_500_000 and n % 10 == 0):
            if (n < 2_500_000 and n % 5000 == 0) or (n > 2_500_000 and n < 4_500_000 and n % 250 == 0) or (n > 4_500_000 and n % 50 == 0):
                costo, _ = test(env, model)
                model.policy.train = True
                env.anio_simulacion = None
                if costo < best_result:
                    best_result = costo
                    best_model = np.copy(model.policy._mean_approximation.mean_weights)
                    # best_unclipped_model = np.copy(model.policy._mean_approximation._unclipped_mean)
                    best_episode = n
                    best_value = np.copy(model.state_space.value)
                expected_value.append(costo)  # lleno

            # print("Episode %d: %.1f Time: %.2fs VS: %.2f%%" % (n, -model.state_space.value[0][0], time.time() - train_start_time, 100 * np.sum(model.state_space.value != qq) / model.state_space.value.size))
            # print(f'{-model.state_space.value[0][0]:.1f}')
            # print(f'Episode {n}: Value: {-model.state_space.value[0][0]:.1f} Mean: {model.policy._mean_approximation.mean_weights[0]:.2f}')
            # print(f'Episode {n}: Value: {-model.state_space.value[9][0]:.1f} Mean: {model.policy._mean_approximation.mean_weights[9]:.2f}')
            # print(f'Episode {n}: Value: {-model.state_space.value[9][0]:.1f} Mean: {model.policy._mean_approximation.mean_weights[np.where(model.policy._mean_approximation.mean_weights!=680)]}')
            # print(f'Episode {n}: Value: {-model.state_space.value[9][0]:.1f} Time: {time.time() - train_start_time:.2f}')
            # print(f'Episode {n}: Value: {-model.state_space.value[9][0]:.1f} VS: {100 * np.sum(model.state_space.value != qq) / model.state_space.value.size:.2f}%')
            # print(f'Episode {n}: Value: {-model.state_space.value[9][0]:.1f} Mean: {model.policy._mean_approximation.mean_weights[939]:.1f} Mean: {model.policy._mean_approximation.mean_weights[940]:.1f} Mean: {model.policy._mean_approximation.mean_weights[944]:.1f} Mean: {model.policy._mean_approximation.mean_weights[945]:.1f} Costo: {best_result:.1f} Time: {time.time() - train_start_time:.1f}')
            print(f'Episode {n}: Value: {-model.state_space.value(init_state_tensor):.1f} Mean: {model.policy._mean_approximation.mean_weights[10][0]:.1f} Mean: {model.policy._mean_approximation.mean_weights[10][1]:.1f} Mean: {model.policy._mean_approximation.mean_weights[10][2]:.1f} Mean: {model.policy._mean_approximation.mean_weights[10][3]:.1f} Costo: {best_result:.1f} Time: {(time.time() - train_start_time) / 60:.1f}m')
            # print(f'Episode {n}: Value: {-model.state_space.value[99][0]:.1f} Mean: {model.policy._mean_approximation.mean_weights[939]:.1f} Mean: {model.policy._mean_approximation.mean_weights[940]:.1f} Mean: {model.policy._mean_approximation.mean_weights[944]:.1f} Mean: {model.policy._mean_approximation.mean_weights[945]:.1f} Costo: {best_result:.1f} Time: {time.time() - train_start_time:.1f}')
            pass
        rewards = []
    # print("Total Train Time: %.2fs" % (time.time() - train_start_time))
    print(f'Total Train Time: {time.time() - train_start_time:.2f}s')
    # print(f'Stats Anios: {env._stats_anios}')
    print(f'======================================================')
    print(f'{model.policy_step_size=}')
    print(f'{model.value_step_size=}')
    print(f'{model.gamma=}')
    print(f'{model.policy.sigma=}')
    print(f'init_lago={env._hidraulicos["bonete"].v_inicial}')
    print(f'{init_mean=:.2f}')
    print(f'{exploring_start=}')
    print(f'{model.state_space.sigma=}')
    print(f'======================================================')
    np.set_printoptions(threshold=np.inf)
    # print(f'{rr}')
    # print(f'{-model.state_space.value}')
    # print(f'{model.policy._mean_approximation.mean_weights=}')
    # print(f'{np.where(model.state_space.value < qq)}')
    # return rewardsARetornar
    # print(f'{best_unclipped_model=}')
    # print(f'{best_value=}')
    print(f'{best_episode=}')
    print(f'{best_result=:.1f}')
    print(f'{best_model=}')
    return expected_value
    # return best_model, best_value


def train_parallel(env, model, num_episodes=1000, verbose=0, exploring_start=False):
    MPI.COMM_WORLD.Barrier()
    train_start_time = time.time()
    rewards = []
    print_interval = 100
    # rewardsARetornar = []
    expected_value = []
    # best_result = 611_520_000  # dos termicos
    # best_result = 98_560_000  # dos termicos, nuevo modelo, RBF
    # best_result = 350_084_468  # dos termicos, nuevo modelo, RBF
    # best_result = 1_135_680_000  # Termico('t_barato', 250, 100),Termico('t1_caro', 100, 400)
    best_result = 7_425_600_000  # Termico('t_barato', 250, 100),Termico('t1_caro', 100, 4000) #FALLA
    best_model = np.zeros_like(model.policy._mean_approximation.mean_weights)
    # best_unclipped_model = np.zeros_like(model.policy._mean_approximation.mean_weights)
    best_episode = 0
    best_value = np.zeros_like(model.state_space.value)
    exploring_start = True

    init_state_tensor = model.state_space.state([8200, 0])
    init_mean = model.policy._mean_approximation.mean_weights[0][0]

    if MPI.COMM_WORLD.Get_rank() == 0:
        print(f'{model.policy_step_size=}')
        print(f'{model.value_step_size=}')
        print(f'{model.gamma=}')
        print(f'{model.policy.sigma=}')
        print(f'init_lago={env._hidraulicos["bonete"].v_inicial}')
        print(f'{init_mean=:.2f}')
        # print(f'{model.policy._mean_approximation.mean_weights.shape}')
        print(f'{exploring_start=}')
        print(f'{model.state_space.sigma=}')
        print(f'{env._gen_aportes_fijos_cant=}')
        print(f'======================================================')
        print(f'Start Time: {time.strftime("%H:%M:%S", time.localtime())}')
        print(f'======================================================')

    num_episodes_before_reduce = 1000
    # for n in range(num_episodes + 1):
    # for n in range(num_episodes // (MPI.COMM_WORLD.Get_size() / 2) + 1):
    for n in range(num_episodes // MPI.COMM_WORLD.Get_size() + 1):
        for i in range(num_episodes_before_reduce):
            state, current_max_turs = env.reset(exploring_start)
            init_state = np.copy(state)
            done = False
            t = -1
            # if verbose == 1:
            #     print(f'Episodio {n}:')
            #     print(f'-------------')
            while not done:
                t += 1  # TIME EN EXPLORING STARTS
                action = model.act(state, current_max_turs)
                next_state, reward, done, current_max_turs = env.step(action)
                # if verbose == 1:
                #     print(f'Semana: {t}')
                #     print(f'Action: {action[0]:.1f} Reward: {reward:.1f}')
                step = model.gradient_step(state, action, reward, next_state, t)
                # expected_value.append(step)
                # rewards.append(reward)
                state = next_state

        # pid_cost, _ = test(env, model)
        # print(f'=================================================================== PID: {MPI.COMM_WORLD.Get_rank()} Costo: {pid_cost:.1f}')

        ######################## MPI ########################
        if MPI.COMM_WORLD.Get_size() > 1:
            model.policy._mean_approximation.mean_weights /= MPI.COMM_WORLD.Get_size()
            model.state_space.weights /= MPI.COMM_WORLD.Get_size()
            MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, model.policy._mean_approximation.mean_weights)
            MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, model.state_space.weights)
        ######################## MPI ########################

        # if (n % print_interval == 0 and n > 0) or (n == 0) or (n == num_episodes - 1):
        #     if (n < 2_500_000 and n % 5000 == 0) or (n > 2_500_000 and n < 4_500_000 and n % 250 == 0) or (n > 4_500_000 and n % 50 == 0):
        # if MPI.COMM_WORLD.Get_rank() == 0:
        #     costo, _ = test(env, model)
        #     model.policy.train = True
        #     env.anio_simulacion = None
        #     if costo < best_result:
        #         best_result = costo
        #         best_model = np.copy(model.policy._mean_approximation.mean_weights)
        #         # best_unclipped_model = np.copy(model.policy._mean_approximation._unclipped_mean)
        #         best_episode = n
        #         best_value = np.copy(model.state_space.value)
        #     expected_value.append(costo)  # lleno

        #     print(f'Episode {n}: Value: {-model.state_space.value(init_state_tensor):.1f} Mean: {model.policy._mean_approximation.mean_weights[10][0]:.1f} Mean: {model.policy._mean_approximation.mean_weights[10][1]:.1f} Mean: {model.policy._mean_approximation.mean_weights[10][2]:.1f} Mean: {model.policy._mean_approximation.mean_weights[10][3]:.1f} Costo: {best_result:.1f} Time: {(time.time() - train_start_time) / 60:.1f}m')

        ######################## TEST_PARALLEL ########################
        costo, _ = test_parallel(env, model)
        costo_np = np.array([0.0])
        MPI.COMM_WORLD.Allreduce(np.array([costo]), costo_np)
        costo = costo_np[0]
        model.policy.train = True
        env.anio_simulacion = None
        if MPI.COMM_WORLD.Get_rank() == 0:
            if costo < best_result:
                best_result = costo
                best_model = np.copy(model.policy._mean_approximation.mean_weights)
                # best_unclipped_model = np.copy(model.policy._mean_approximation._unclipped_mean)
                best_episode = n
                best_value = np.copy(model.state_space.value)
            expected_value.append(costo)  # lleno

            print(f'Episode {n}: Value: {-model.state_space.value(init_state_tensor):.1f} Mean: {model.policy._mean_approximation.mean_weights[10][0]:.1f} Mean: {model.policy._mean_approximation.mean_weights[10][1]:.1f} Mean: {model.policy._mean_approximation.mean_weights[10][2]:.1f} Mean: {model.policy._mean_approximation.mean_weights[10][3]:.1f} Costo: {best_result:.1f} Time: {(time.time() - train_start_time) / 60:.1f}m')
        ######################## TEST_PARALLEL ########################


        ###################################################
        # # if n == 1_000_000:
        # if n == 1_000 // MPI.COMM_WORLD.Get_size():
        #     model.policy.epsilon /= 2
        #     model.policy_step_size /= 2
        #     model.value_step_size /= 2
        # # if n == 2_000_000:
        # if n == 2_000 // MPI.COMM_WORLD.Get_size():
        #     model.policy.epsilon /= 2
        #     model.policy_step_size /= 2
        #     model.value_step_size /= 2
        # # if n == 3_000_000:
        # if n == 3_000 // MPI.COMM_WORLD.Get_size():
        #     model.policy.epsilon /= 2
        #     model.policy_step_size /= 2
        #     model.value_step_size /= 2
        # # if n == 4_000_000:
        # if n == 4_000 // MPI.COMM_WORLD.Get_size():
        #     model.policy.epsilon /= 2
        #     model.policy_step_size /= 2
        #     model.value_step_size /= 2
        # # if n == 4_500_000:
        # if n == 4_500 // MPI.COMM_WORLD.Get_size():
        #     model.policy.epsilon /= 2
        #     model.policy_step_size /= 2
        #     model.value_step_size /= 2

        # # if n == 2_500_000:
        # if n == 2_500 // MPI.COMM_WORLD.Get_size():
        # if n == 2_500:
        #     num_episodes_before_reduce = 250
        # # if n == 4_500_000:
        # if n == 4_500 // MPI.COMM_WORLD.Get_size():
        # if n == 4_500:
        #     num_episodes_before_reduce = 50
        ###################################################

        # rewards = []
    MPI.COMM_WORLD.Barrier()
    if MPI.COMM_WORLD.Get_rank() == 0:
        print(f'Total Train Time: {time.time() - train_start_time:.2f}s')
        print(f'======================================================')
        print(f'{model.policy_step_size=}')
        print(f'{model.value_step_size=}')
        print(f'{model.gamma=}')
        print(f'{model.policy.sigma=}')
        print(f'init_lago={env._hidraulicos["bonete"].v_inicial}')
        print(f'{init_mean=:.2f}')
        print(f'{exploring_start=}')
        print(f'{model.state_space.sigma=}')
        print(f'======================================================')
        np.set_printoptions(threshold=np.inf)
        print(f'{model.policy._mean_approximation.mean_weights=}')
        print(f'{best_episode=}')
        print(f'{best_result=:.1f}')
        print(f'{best_model=}')
    return expected_value
    # return best_model, best_value


def train_parallel_2(env, model, num_episodes=1000, verbose=0, exploring_start=False):
    MPI.COMM_WORLD.Barrier()
    train_start_time = time.time()
    rewards = []
    print_interval = 100
    # rewardsARetornar = []
    expected_value = []
    # best_result = 611_520_000  # dos termicos
    # best_result = 98_560_000  # dos termicos, nuevo modelo, RBF
    # best_result = 350_084_468  # dos termicos, nuevo modelo, RBF
    # best_result = 1_135_680_000  # Termico('t_barato', 250, 100),Termico('t1_caro', 100, 400)
    best_result = 7_425_600_000  # Termico('t_barato', 250, 100),Termico('t1_caro', 100, 4000) #FALLA
    best_model = np.zeros_like(model.policy._mean_approximation.mean_weights)
    # best_unclipped_model = np.zeros_like(model.policy._mean_approximation.mean_weights)
    best_episode = 0
    best_pid_cost = best_result
    best_pid_episode = -1
    pid_costs = []
    pid_costs_train_dataset = []
    best_costos_por_escenarios = []
    best_values = np.zeros_like(model.state_space.value)
    exploring_start = True
    best_expected_val = best_result
    best_value = None

    init_state_tensor = model.state_space.state([4100, 0])
    if MPI.COMM_WORLD.Get_rank() == 1:
        states_RBF = []
        # for i in range(0, 8200, 820):
        # for i in np.linspace(0, 8200, 101):
        for i in np.linspace(0, 8200, model.state_space.num_centers[0]):
            states_RBF.append(model.state_space.state([i, 0])[:, 0])
            print(f'INIT_VAL: {-model.state_space.value(model.state_space.state([i, 0]))}')
        print(f'{states_RBF=}')
        print(f'{init_state_tensor[:,0]=}')
        print(f'{model.state_space.weights[:,0]=}')
        # print(f'{model.state_space.weights=}')
        print(f'E_VAL: {-model.state_space.value(init_state_tensor)}')
        print(f'======================================================')
    init_mean = model.policy._mean_approximation.mean_weights[0][0]

    if MPI.COMM_WORLD.Get_rank() == 0:
        print(f'{model.policy_step_size=}')
        print(f'{model.value_step_size=}')
        print(f'{model.gamma=}')
        print(f'{model.policy.sigma=}')
        print(f'init_lago={env._hidraulicos["bonete"].v_inicial}')
        print(f'{init_mean=:.2f}')
        # print(f'{model.policy._mean_approximation.mean_weights.shape}')
        print(f'{exploring_start=}')
        print(f'{model.state_space.sigma=}')
        print(f'{env._h_fijo=}')
        # print(f'{env._gen_aportes_fijos_cant=}')
        print(f'{model.policy.epsilon=}')
        print(f'======================================================')
        print(f'Start Time: {time.strftime("%H:%M:%S", time.localtime())}')
        print(f'======================================================')

    # num_episodes_before_reduce = 1000
    num_episodes_before_reduce = 500
    # num_episodes_before_reduce = 1
    for n in range(num_episodes // (MPI.COMM_WORLD.Get_size() - 1 // 2) + 1):
    # for n in range(num_episodes // MPI.COMM_WORLD.Get_size() + 1):
        if MPI.COMM_WORLD.Get_rank() > 0:
            for i in range(num_episodes_before_reduce):
                state, current_max_turs = env.reset(exploring_start)
                init_state = np.copy(state)
                done = False
                t = -1
                # if verbose == 1:
                #     print(f'Episodio {n}:')
                #     print(f'-------------')
                while not done:
                    t += 1  # TIME EN EXPLORING STARTS
                    action = model.act(state, current_max_turs)
                    next_state, reward, done, current_max_turs = env.step(action)
                    # if verbose == 1:
                    #     print(f'Semana: {t}')
                    #     print(f'Action: {action[0]:.1f} Reward: {reward:.1f}')
                    step = model.gradient_step(state, action, reward, next_state, t)
                    # expected_value.append(step)
                    # rewards.append(reward)
                    state = next_state

            pid_cost, _, costos_por_escenario = test(env, model)
            # pid_cost_train_dataset, _ = test(env, model, train_dataset=True)
            # print(f'{model.policy.train=}')
            # print(f'{env.anio_simulacion=}')
            # print(f'{env._train_dataset=}')
            env.anio_simulacion = None
            env._train_dataset = False
            model.policy.train = True
            # print(f'{model.policy.train=}')
            # print(f'{env.anio_simulacion=}')
            # print(f'{env._train_dataset=}')
            pid_costs.append(pid_cost)
            # pid_costs_train_dataset.append(pid_cost_train_dataset)
            if pid_cost < best_pid_cost:
                best_pid_cost = pid_cost
                best_pid_episode = n
            # print(f'=================================================================== PID: {MPI.COMM_WORLD.Get_rank()} Costo: {pid_cost:.1f}')

        ######################## MPI ########################
            # send pid_cost
            MPI.COMM_WORLD.Send(np.array([pid_cost]), 0)
            # send policy
            MPI.COMM_WORLD.Send(model.policy._mean_approximation.mean_weights, 0)
            MPI.COMM_WORLD.send(-model.state_space.value(init_state_tensor), dest=0)
            MPI.COMM_WORLD.Send(model.state_space.weights, 0)
            MPI.COMM_WORLD.send(costos_por_escenario, dest=0)
        else:
            for i in range(1, MPI.COMM_WORLD.Get_size()):
                # recv pid_cost
                costo_np = np.array([0.0])
                MPI.COMM_WORLD.Recv(costo_np, i)
                costo = costo_np[0]
                # recv policy
                MPI.COMM_WORLD.Recv(model.policy._mean_approximation.mean_weights, i)
                expected_val = MPI.COMM_WORLD.recv(source=i)
                MPI.COMM_WORLD.Recv(model.state_space.weights, i)
                costos_por_escenario = MPI.COMM_WORLD.recv(source=i)
                if costo < best_result:
                    best_result = costo
                    best_model = np.copy(model.policy._mean_approximation.mean_weights)
                    best_episode = n
                    best_expected_val = expected_val
                    best_value = np.copy(model.state_space.weights)
                    best_costos_por_escenarios = costos_por_escenario
            # print(f'Episode {n}: Value: {-model.state_space.value(init_state_tensor):.1f} Mean: {model.policy._mean_approximation.mean_weights[10][0]:.1f} Mean: {model.policy._mean_approximation.mean_weights[10][1]:.1f} Mean: {model.policy._mean_approximation.mean_weights[10][2]:.1f} Mean: {model.policy._mean_approximation.mean_weights[10][3]:.1f} Costo: {best_result:.1f} Time: {(time.time() - train_start_time) / 60:.1f}m')
            print(f'Episode {n}: Value: {best_expected_val:.1f} Mean: {model.policy._mean_approximation.mean_weights[10][0]:.1f} Mean: {model.policy._mean_approximation.mean_weights[10][1]:.1f} Mean: {model.policy._mean_approximation.mean_weights[10][2]:.1f} Mean: {model.policy._mean_approximation.mean_weights[10][3]:.1f} Costo: {best_result:.1f} Time: {(time.time() - train_start_time) / 60:.1f}m')
        # if MPI.COMM_WORLD.Get_size() > 1:
        #     model.policy._mean_approximation.mean_weights /= MPI.COMM_WORLD.Get_size()
        #     model.state_space.weights /= MPI.COMM_WORLD.Get_size()
        #     MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, model.policy._mean_approximation.mean_weights)
        #     MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, model.state_space.weights)
        ######################## MPI ########################

        # if (n % print_interval == 0 and n > 0) or (n == 0) or (n == num_episodes - 1):
        #     if (n < 2_500_000 and n % 5000 == 0) or (n > 2_500_000 and n < 4_500_000 and n % 250 == 0) or (n > 4_500_000 and n % 50 == 0):
        # if MPI.COMM_WORLD.Get_rank() == 0:
        #     costo, _ = test(env, model)
        #     model.policy.train = True
        #     env.anio_simulacion = None
        #     if costo < best_result:
        #         best_result = costo
        #         best_model = np.copy(model.policy._mean_approximation.mean_weights)
        #         # best_unclipped_model = np.copy(model.policy._mean_approximation._unclipped_mean)
        #         best_episode = n
        #         best_value = np.copy(model.state_space.value)
        #     expected_value.append(costo)  # lleno

        #     print(f'Episode {n}: Value: {-model.state_space.value(init_state_tensor):.1f} Mean: {model.policy._mean_approximation.mean_weights[10][0]:.1f} Mean: {model.policy._mean_approximation.mean_weights[10][1]:.1f} Mean: {model.policy._mean_approximation.mean_weights[10][2]:.1f} Mean: {model.policy._mean_approximation.mean_weights[10][3]:.1f} Costo: {best_result:.1f} Time: {(time.time() - train_start_time) / 60:.1f}m')

        ######################## TEST_PARALLEL ########################
        # costo, _ = test_parallel(env, model)
        # costo_np = np.array([0.0])
        # MPI.COMM_WORLD.Allreduce(np.array([costo]), costo_np)
        # costo = costo_np[0]
        # model.policy.train = True
        # env.anio_simulacion = None
        # if MPI.COMM_WORLD.Get_rank() == 0:
        #     if costo < best_result:
        #         best_result = costo
        #         best_model = np.copy(model.policy._mean_approximation.mean_weights)
        #         # best_unclipped_model = np.copy(model.policy._mean_approximation._unclipped_mean)
        #         best_episode = n
        #         best_value = np.copy(model.state_space.value)
        #     expected_value.append(costo)  # lleno

        #     print(f'Episode {n}: Value: {-model.state_space.value(init_state_tensor):.1f} Mean: {model.policy._mean_approximation.mean_weights[10][0]:.1f} Mean: {model.policy._mean_approximation.mean_weights[10][1]:.1f} Mean: {model.policy._mean_approximation.mean_weights[10][2]:.1f} Mean: {model.policy._mean_approximation.mean_weights[10][3]:.1f} Costo: {best_result:.1f} Time: {(time.time() - train_start_time) / 60:.1f}m')
        ######################## TEST_PARALLEL ########################


        ###################################################
        # # if n == 1_000_000:
        # if n == 1_000 // MPI.COMM_WORLD.Get_size():
        #     model.policy.epsilon /= 2
        #     model.policy_step_size /= 2
        #     model.value_step_size /= 2
        # # if n == 2_000_000:
        # if n == 2_000 // MPI.COMM_WORLD.Get_size():
        #     model.policy.epsilon /= 2
        #     model.policy_step_size /= 2
        #     model.value_step_size /= 2
        # # if n == 3_000_000:
        # if n == 3_000 // MPI.COMM_WORLD.Get_size():
        #     model.policy.epsilon /= 2
        #     model.policy_step_size /= 2
        #     model.value_step_size /= 2
        # # if n == 4_000_000:
        # if n == 4_000 // MPI.COMM_WORLD.Get_size():
        #     model.policy.epsilon /= 2
        #     model.policy_step_size /= 2
        #     model.value_step_size /= 2
        # # if n == 4_500_000:
        # if n == 4_500 // MPI.COMM_WORLD.Get_size():
        #     model.policy.epsilon /= 2
        #     model.policy_step_size /= 2
        #     model.value_step_size /= 2

        # if n == 2_500_000:
        if n == 2_500 // (MPI.COMM_WORLD.Get_size() - 1 // 2):
            num_episodes_before_reduce = 250
        # if n == 4_500_000:
        if n == 4_500 // (MPI.COMM_WORLD.Get_size() - 1 // 2):
            num_episodes_before_reduce = 50
        ###################################################

        # rewards = []
    MPI.COMM_WORLD.Barrier()
    if MPI.COMM_WORLD.Get_rank() == 1:
        print(f'{init_state_tensor[:,0]=}')
        print(f'{model.state_space.weights[:,0]=}')
        print(f'E_VAL: {-model.state_space.value(init_state_tensor)}')
        print(f'===========================================')
    if MPI.COMM_WORLD.Get_rank() == 0:
        print(f'Total Train Time: {time.time() - train_start_time:.2f}s')
        print(f'======================================================')
        print(f'{model.policy_step_size=}')
        print(f'{model.value_step_size=}')
        print(f'{model.gamma=}')
        print(f'{model.policy.sigma=}')
        print(f'init_lago={env._hidraulicos["bonete"].v_inicial}')
        print(f'{init_mean=:.2f}')
        print(f'{exploring_start=}')
        print(f'{model.state_space.sigma=}')
        print(f'{model.policy.epsilon=}')
        print(f'======================================================')
        np.set_printoptions(threshold=np.inf)
        print(f'{model.policy._mean_approximation.mean_weights=}')
        print(f'{best_episode=}')
        print(f'{best_result=:.1f}')
        print(f'{best_model=}')
        print(f'{best_value=}')
        print(f'{best_costos_por_escenarios=}')
        #######################################################
        for i in range(1, MPI.COMM_WORLD.Get_size()):
            best_pid_cost = MPI.COMM_WORLD.recv(source=i)
            best_pid_episode = MPI.COMM_WORLD.recv(source=i)
            pid_costs = MPI.COMM_WORLD.recv(source=i)
            pid_costs_train_dataset = MPI.COMM_WORLD.recv(source=i)
            print(f'===============PID: {i} BEST_PID_COST: {best_pid_cost} BEST_PID_EPISODE: {best_pid_episode}')
            print(f'===============PID: {i} COSTS: {pid_costs}')
            print(f'===============PID: {i} COSTS_TRAIN_DATASET: {pid_costs_train_dataset}')
    else:
        MPI.COMM_WORLD.send(best_pid_cost, dest=0)
        MPI.COMM_WORLD.send(best_pid_episode, dest=0)
        MPI.COMM_WORLD.send(pid_costs, dest=0)
        MPI.COMM_WORLD.send(pid_costs_train_dataset, dest=0)
    return expected_value
    # return best_model, best_value


def test(env, model, sorteos=1000, train_dataset=False):
    # train_start_time = time.time()
    env._train_dataset = train_dataset
    model.policy.train = False
    model.policy.eps = 0
    costo_total = []
    # res = [0] * 104
    actions = []
    # cant_anios = 2000
    # print(f'{len(env._gen_aportes_fijos)=}')
    # print(f'{len(env._lista_aportes["bonete"])=}')
    cant_anios = (len(env._gen_aportes_fijos) // 52) if train_dataset else (len(env._lista_aportes['bonete']) // 52)
    # print(f'{cant_anios=}')
    volumen = np.zeros((cant_anios, 104))
    costo_por_paso = np.zeros((cant_anios, 104))
    # for n in range(sorteos):
    # for n in range(cant_anios):
    for n in range(0, cant_anios, 2):  # sin repetir anios
    # for n in range(1909, 2019):
        # print(f'\n{n=}')
        env.anio_simulacion = n
        state, current_max_turs = env.reset()
        done = False
        t = -1
        episode_cost = 0
        while not done:
            t += 1
            action = model.act(state, current_max_turs)
            # print(f'{action=}')
            actions.append(action)
            # print(f'{action[0]=:.2f}', end=' ')
            next_state, costo, done, current_max_turs = env.step(action)
            # next_state, volume_state, costo, done, current_max_turs = env.step(action)
            state = next_state
            episode_cost += -costo
            # costo_por_paso[n - 1909, t] = -costo  # historicos anio 1909
            costo_por_paso[n, t] = -costo  # sintecticos anio 0
            # volumen[n, volume_state[1]] = volume_state[0]
        # print(f'{len(actions)=}')
        # res = [sum(x) for x in zip(res, actions)]
        # print(f'{len(res)=}')
        actions.clear()
        costo_total.append(episode_cost)
    # res = [x / sorteos for x in res]
    # print(res)
    # print(f'Costo Promedio: {np.mean(costo_total):.1f} MUSD')
    # print(f'eps: {(model.policy.eps / model.policy.total) * 100:.2f}%')
    # print(f'Costo Promedio: {np.mean(costo_total):.1f} USD')
    # return res
    # np.set_printoptions(threshold=np.inf)
    # print(f'{volumen=}')
    return np.mean(costo_total), costo_por_paso.mean(axis=0), costo_total


def test_parallel(env, model, sorteos=1000):
    # train_start_time = time.time()
    model.policy.train = False
    model.policy.eps = 0
    costo_total = []
    # res = [0] * 104
    actions = []
    cant_anios = 2000
    volumen = np.zeros((cant_anios, 104))
    costo_por_paso = np.zeros((cant_anios, 104))
    # for n in range(sorteos):
    # for n in range(cant_anios):
    # for n in range(MPI.COMM_WORLD.Get_rank(), cant_anios, MPI.COMM_WORLD.Get_size()):
    for n in range(MPI.COMM_WORLD.Get_rank()*2, cant_anios, MPI.COMM_WORLD.Get_size()*2): # sin repetir anios
    # for n in range(1909, 2019):
        # print(f'\n{n=}')
        env.anio_simulacion = n
        state, current_max_turs = env.reset()
        done = False
        t = -1
        episode_cost = 0
        while not done:
            t += 1
            action = model.act(state, current_max_turs)
            actions.append(action)
            next_state, costo, done, current_max_turs = env.step(action)
            # next_state, volume_state, costo, done, current_max_turs = env.step(action)
            state = next_state
            episode_cost += -costo
            # costo_por_paso[n - 1909, t] = -costo  # historicos anio 1909
            costo_por_paso[n, t] = -costo  # sintecticos anio 0
            # volumen[n, volume_state[1]] = volume_state[0]
        # res = [sum(x) for x in zip(res, actions)]
        actions.clear()
        costo_total.append(episode_cost)
    # res = [x / sorteos for x in res]
    # return res
    # np.set_printoptions(threshold=np.inf)
    # print(f'{volumen=}')
    # return np.mean(costo_total), costo_por_paso.mean(axis=0)
    # return np.sum(costo_total) / cant_anios, costo_por_paso.mean(axis=0)
    return np.sum(costo_total) / (cant_anios / 2), costo_por_paso.mean(axis=0)


# def test(env, model, sorteos=1000):
#     # train_start_time = time.time()
#     # model.policy.train = False
#     model.policy.train = True  # para forzar epsilon greedy
#     # model.policy.eps = 0
#     model.policy.eps = 101  # para forzar epsilon greedy
#     # costo_total = []
#     costo_total = {}  # devolver dict por anio
#     res = [0] * 104
#     actions = []
#     # c = 100 / .19
#     costo_150M = 2_520_000
#     # for n in range(sorteos):
#     for n in range(1909, 2019):
#         env.anio_simulacion = n
#         state, current_max_turs = env.reset()
#         done = False
#         t = -1
#         episode_cost = 0
#         save = 0
#         while not done:
#             t += 1
#             action = model.act(state, current_max_turs)
#             # print(f'{action=}')
#             actions.append(action)
#             # print(f'{action[0]=:.2f}', end=' ')
#             next_state, costo, done, current_max_turs = env.step(action)
#             state = next_state
#             episode_cost += -costo

#             # ------------------------------------- #
#             # ------ OPTIMIZACION ALMACENERO ------ #
#             # ------------------------------------- #
#             # if costo > -costo_150M:
#             #     save += costo + costo_150M
#             #     episode_cost += -costo
#             # if costo < -costo_150M:
#             #     # print(f'{costo=:.2f}')
#             #     extra_cost = -costo - costo_150M
#             #     episode_cost += -costo - extra_cost
#             #     if save >= extra_cost:
#             #         episode_cost += extra_cost
#             #         save -= extra_cost
#             #     else:
#             #         episode_cost += save
#             #         extra_cost -= save
#             #         save = 0
#             #         episode_cost += extra_cost * 2
#             # ------------------------------------- #
#             # ------ OPTIMIZACION ALMACENERO ------ #
#             # ------------------------------------- #

#         # print(f'{len(actions)=}')
#         res = [sum(x) for x in zip(res, actions)]
#         # print(f'{len(res)=}')
#         actions.clear()
#         # costo_total.append(episode_cost)
#         costo_total[n] = episode_cost  # devolver dict por anio
#     res = [x / sorteos for x in res]
#     # print(res)
#     # print(f'Costo Promedio: {np.mean(costo_total):.1f} MUSD')
#     # print(f'eps: {(model.policy.eps / model.policy.total) * 100:.2f}%')
#     # print(f'Costo Promedio: {np.mean(costo_total):.1f} USD')
#     # return res
#     # return np.mean(costo_total)
#     return costo_total  # devolver dict por anio
