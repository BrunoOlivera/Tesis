import numpy as np
import random


# GRID
# class LinearMeanApproximation:

#     def __init__(self, state_discretization, inits):
#         # self._mean_discretization = np.zeros(sum(state_discretization.shape), dtype=decimal.Decimal) + 680 # deshardcodear
#         # self._mean_discretization = np.array(list(map(lambda x: decimal.Decimal(x), np.array(inits)[self.flatten([[0+i]*v for i,v in enumerate(state_discretization.shape)])])))
#         self._mean_discretization = np.concatenate([np.zeros(state_discretization.shape[v] * state_discretization.shape[-1]) + inits[v] for v in range(len(state_discretization.shape) - 1)])
#         # self._mean_discretization = np.array(list(map(lambda x: decimal.Decimal(x), self._mean_discretization)))
#         # self._mean_discretization = np.array(list(map(lambda x: decimal.Decimal(x), self._mean_discretization)))
#         # print(f'{self._mean_discretization.shape=}')
#         # print(f'{state_discretization.shape=}')
#         # self._state_index = np.zeros_like(state_discretization.value, dtype=object)
#         self._state_index = {}
#         self.index_pre_processing(state_discretization.shape)
#         # print(f'{self._state_index.shape=}')
#         # aa = np.zeros((10,104))
#         # aa[9][0] = 1
#         # print(f'{self._state_index[aa.tobytes()]}')
#         self._unclipped_mean = np.zeros_like(self._mean_discretization)

#     def index_pre_processing(self, state_shape):
#         it = np.nditer(np.zeros(state_shape), flags=['multi_index', 'refs_ok'])
#         for _ in it:
#             aux_state = np.zeros(state_shape)
#             aux_state[it.multi_index] = 1
#             key = aux_state.tobytes()
#             self._state_index[key] = [sum(e) for e in zip([it.multi_index[v] * state_shape[-1] + it.multi_index[-1] for v in range(len(it.multi_index) - 1)], [0] + [v * state_shape[-1] for v in state_shape[:-2]])]

#     # def index_pre_processing(self):
#     #     it = np.nditer(self._state_index, flags=['multi_index', 'refs_ok'])
#     #     for _ in it:
#     #         # print(f'{type(it.multi_index)=}')
#     #         # zip([it.multi_index[v] * self._state_index.shape[-1] + it.multi_index[-1] for v in range(len(it.multi_index) - 1)], [0] + [v * self._state_index.shape[-1] for v in self._state_index.shape[:-2]])
#     #         self._state_index[it.multi_index] = [sum(e) for e in zip([it.multi_index[v] * self._state_index.shape[-1] + it.multi_index[-1] for v in range(len(it.multi_index) - 1)], [0] + [v * self._state_index.shape[-1] for v in self._state_index.shape[:-2]])]
#     #         # print([sum(e) for e in zip([it.multi_index[v] * self._state_index.shape[-1] + it.multi_index[-1] for v in range(len(it.multi_index) - 1)], [0] + [v * self._state_index.shape[-1] for v in self._state_index.shape[:-2]])])

#     def getIndex(self, state):
#         return self._state_index[state.tobytes()]

#     # def getIndex(self, state):
#     #     # ones = list(map(lambda x: np.sum(x), np.where(state == 1)))
#     #     # return [val + sum(state.shape[0:idx]) for idx, val in enumerate(ones)]
#     #     ones = np.where(state == 1)
#     #     ones = [f[0] for f in ones]
#     #     return [sum(e) for e in zip([ones[v] * state.shape[-1] + ones[-1] for v in range(len(ones) - 1)], [0] + [v * state.shape[-1] for v in state.shape[:-2]])]

#     @property
#     def mean(self):
#         return self._mean_discretization

#     @mean.setter
#     def mean(self, value):
#         self._mean = value

#     @property
#     def unclipped_mean(self):
#         return self._unclipped_mean

#     @unclipped_mean.setter
#     def unclipped_mean(self, value):
#         self._unclipped_mean = value

#     def flatten(self, t):
#         return [item for sublist in t for item in sublist]

# RBF
class LinearMeanApproximation:

    def __init__(self, state_discretization, inits):
        # self._mean_discretization = np.zeros(sum(state_discretization.shape), dtype=decimal.Decimal) + 680 # deshardcodear
        # self._mean_discretization = np.array(list(map(lambda x: decimal.Decimal(x), np.array(inits)[self.flatten([[0+i]*v for i,v in enumerate(state_discretization.shape)])])))
        # self._mean_discretization = np.concatenate([np.zeros(state_discretization.shape[v] * state_discretization.shape[-1]) + inits[v] for v in range(len(state_discretization.shape) - 1)])
        # self._mean_discretization = np.array(list(map(lambda x: decimal.Decimal(x), self._mean_discretization)))
        # self._mean_discretization = np.array(list(map(lambda x: decimal.Decimal(x), self._mean_discretization)))
        # print(f'{self._mean_discretization.shape=}')
        # print(f'{state_discretization.shape=}')
        # self._state_index = np.zeros_like(state_discretization.value, dtype=object)
        # self._state_index = {}
        # self.index_pre_processing(state_discretization.shape)
        # print(f'{self._state_index.shape=}')
        # aa = np.zeros((10,104))
        # aa[9][0] = 1
        # print(f'{self._state_index[aa.tobytes()]}')
        # self._unclipped_mean = np.zeros_like(self._mean_discretization)
        self.state_discretization = state_discretization
        ####################################################################
        # self.mean_weights = np.zeros_like(state_discretization.shape)
        self.mean_weights = np.zeros(state_discretization.shape) + inits[0]  # TODO: generalizar
        # print(f'{self.mean_weights.shape=}')
        ####################################################################

    # def index_pre_processing(self, state_shape):
    #     it = np.nditer(np.zeros(state_shape), flags=['multi_index', 'refs_ok'])
    #     for _ in it:
    #         aux_state = np.zeros(state_shape)
    #         aux_state[it.multi_index] = 1
    #         key = aux_state.tobytes()
    #         self._state_index[key] = [sum(e) for e in zip([it.multi_index[v] * state_shape[-1] + it.multi_index[-1] for v in range(len(it.multi_index) - 1)], [0] + [v * state_shape[-1] for v in state_shape[:-2]])]

    # def index_pre_processing(self):
    #     it = np.nditer(self._state_index, flags=['multi_index', 'refs_ok'])
    #     for _ in it:
    #         # print(f'{type(it.multi_index)=}')
    #         # zip([it.multi_index[v] * self._state_index.shape[-1] + it.multi_index[-1] for v in range(len(it.multi_index) - 1)], [0] + [v * self._state_index.shape[-1] for v in self._state_index.shape[:-2]])
    #         self._state_index[it.multi_index] = [sum(e) for e in zip([it.multi_index[v] * self._state_index.shape[-1] + it.multi_index[-1] for v in range(len(it.multi_index) - 1)], [0] + [v * self._state_index.shape[-1] for v in self._state_index.shape[:-2]])]
    #         # print([sum(e) for e in zip([it.multi_index[v] * self._state_index.shape[-1] + it.multi_index[-1] for v in range(len(it.multi_index) - 1)], [0] + [v * self._state_index.shape[-1] for v in self._state_index.shape[:-2]])])

    # def getIndex(self, state):
    #     return self._state_index[state.tobytes()]

    # def getIndex(self, state):
    #     # ones = list(map(lambda x: np.sum(x), np.where(state == 1)))
    #     # return [val + sum(state.shape[0:idx]) for idx, val in enumerate(ones)]
    #     ones = np.where(state == 1)
    #     ones = [f[0] for f in ones]
    #     return [sum(e) for e in zip([ones[v] * state.shape[-1] + ones[-1] for v in range(len(ones) - 1)], [0] + [v * state.shape[-1] for v in state.shape[:-2]])]

    # @property
    # def mean(self):
    #     return self.mean_weights

    # @mean.setter
    # def mean(self, value):
    #     self._mean = value

    # @property
    # def unclipped_mean(self):
    #     return self._unclipped_mean

    # @unclipped_mean.setter
    # def unclipped_mean(self, value):
    #     self._unclipped_mean = value

    # def flatten(self, t):
    #     return [item for sublist in t for item in sublist]

    def mean_from_state(self, state):
        return np.sum(self.mean_weights * state)
        # mean = np.sum(np.sum(self._mean_approximation.mean * state, axis=1), axis=1)  # TODO: para multinormal

# GRID
# class GaussianPolicy:

#     def __init__(self, mean_approximation, sigma=0.5, eps=.1, verbose=0):
#         self._mean_approximation = mean_approximation
#         self._sigma = sigma
#         self.train = True
#         self.total = 0
#         self.eps = 0
#         self.epsilon = eps
#         self.verbose = verbose

#     @property
#     def sigma(self):
#         return self._sigma

#     def act(self, state, current_max_turs):
#         # print(f'{state=}')
#         index = self._mean_approximation.getIndex(state)
#         mean = self._mean_approximation.mean[index]
#         if mean.size == 1:
#             # action = np.array(decimal.Decimal(np.random.normal(mean[0], self._sigma))).reshape(1,)
#             self.total += 1
#             if self.train and random.random() < self.epsilon:
#                 # action = np.array(current_max_turs[0])  # max_tur eps-greedy
#                 # action = np.array([680])
#                 action = np.array(current_max_turs[0] * random.random())  # random eps-greedy
#                 self.eps += 1
#             elif self.train:
#                 action = np.array(np.random.normal(mean[0], self._sigma)).reshape(1,)
#             else:
#                 action = np.array(mean[0])
#             # action = np.array([680])
#             action = np.array(np.clip(action, 0, current_max_turs[0]))
#             # action = np.array(np.clip(action, decimal.Decimal(0), current_max_turs[0]))
#         else:
#             if self.train and random.random() < .10:
#                 action = np.array(current_max_turs)
#             else:
#                 cov = np.zeros((mean.size, mean.size))
#                 np.fill_diagonal(cov, self._sigma**2)
#                 action = np.random.multivariate_normal(mean, cov)
#             # np.clip(action, decimal.Decimal(0), current_max_turs, action)
#             np.clip(action, 0, current_max_turs, action)
#         return action

#     def gradient_step(self, state, action, delta, gamma, policy_step_size, time):
#         index = self._mean_approximation.getIndex(state)
#         # print(f'State: {np.where(state == 1)} {index=}')
#         # print(f'{index=}')
#         # print(f'{delta=}')
#         # gradient = ((action - self._mean_approximation.mean[index]) / decimal.Decimal(self._sigma**2))
#         mean_before = self._mean_approximation.mean[index]
#         # ############## #
#         # unclipped mean #
#         # gradient_unclipped = (action - self._mean_approximation.unclipped_mean[index]) / self._sigma**2
#         # self._mean_approximation.unclipped_mean[index] += policy_step_size * gamma**time * delta * gradient_unclipped
#         # unclipped mean
#         # ############## #
#         gradient = (action - self._mean_approximation.mean[index]) / self._sigma**2
#         # print(f'{gradient=}')
#         if self.verbose == 1:
#             print(f'Mean antes:{self._mean_approximation.mean[index][0]:.1f} Delta: {delta:.3f} Gradient: {gradient[0]:.3f}', end=' ')
#         # print(f'{(policy_step_size * gamma**time * delta * gradient)[0]}')
#         # self._mean_approximation.mean[index] += decimal.Decimal(policy_step_size) * decimal.Decimal(gamma**time) * delta * gradient
#         # print(f'{policy_step_size=} {delta=} {gradient=} step: {policy_step_size * gamma**time * delta * gradient}')
#         self._mean_approximation.mean[index] += policy_step_size * gamma**time * delta * gradient
#         # step = np.clip(policy_step_size * gamma**time * delta * gradient, -300, 300)
#         # self._mean_approximation.mean[index] += step
#         # self._mean_approximation.mean[index] += policy_step_size * delta * gradient
#         # self._mean_approximation.mean[index] = np.clip(self._mean_approximation.mean[index], decimal.Decimal(0), decimal.Decimal(680))
#         self._mean_approximation.mean[index] = np.clip(self._mean_approximation.mean[index], 0, 680)
#         # self._mean_approximation.mean[index] -= decimal.Decimal(policy_step_size) * decimal.Decimal(gamma**time) * delta * gradient
#         if self.verbose == 1:
#             print(f'Mean desp:{self._mean_approximation.mean[index][0]:.1f}')
#         mean_after = self._mean_approximation.mean[index]
#         return mean_after != mean_before


# RBF
class GaussianPolicy:

    def __init__(self, mean_approximation, sigma=0.5, eps=.1, verbose=0):
        self._mean_approximation = mean_approximation
        self._sigma = sigma
        self.train = True
        self.total = 0
        self.eps = 0
        self.epsilon = eps
        self.verbose = verbose

    @property
    def sigma(self):
        return self._sigma

    def act(self, state, current_max_turs):
        mean = self._mean_approximation.mean_from_state(state)
        if mean.ndim == 0:
            # self.total += 1
            if self.train and random.random() < self.epsilon:
                action = np.array(current_max_turs[0] * random.random())  # random eps-greedy
                # self.eps += 1
            elif self.train:
                action = np.array(np.random.normal(mean, self._sigma)).reshape(1,)
            else:
                action = np.array(mean)
            action = np.array(np.clip(action, 0, current_max_turs[0]))
        else:
            if self.train and random.random() < .10:
                action = np.array(current_max_turs)
            else:
                cov = np.zeros((mean.size, mean.size))
                np.fill_diagonal(cov, self._sigma**2)
                action = np.random.multivariate_normal(mean, cov)
            np.clip(action, 0, current_max_turs, action)
        return action

    def gradient_step(self, state, action, delta, gamma, policy_step_size, time):
        # index = self._mean_approximation.getIndex(state)
        # print(f'State: {np.where(state == 1)} {index=}')
        # print(f'{index=}')
        # print(f'{delta=}')
        # gradient = ((action - self._mean_approximation.mean[index]) / decimal.Decimal(self._sigma**2))
        mean = self._mean_approximation.mean_from_state(state)
        # mean_before = mean
        # ############## #
        # unclipped mean #
        # gradient_unclipped = (action - self._mean_approximation.unclipped_mean[index]) / self._sigma**2
        # self._mean_approximation.unclipped_mean[index] += policy_step_size * gamma**time * delta * gradient_unclipped
        # unclipped mean
        # ############## #

        # print(f'POLICY->GRADIENT_STEP->{state[:,time]=}')
        # print(f'POLICY->GRADIENT_STEP->{mean=}')
        # print(f'POLICY->GRADIENT_STEP->{action=}')
        # print(f'POLICY->GRADIENT_STEP->{action.shape=}')
        # print(f'POLICY->GRADIENT_STEP->{(mean * state)[:,time]=}')
        # print(f'POLICY->GRADIENT_STEP->{(mean * state).shape=}')

        # gradient = state * (action - mean * state) / self._sigma**2
        gradient = state * (action - mean) / self._sigma**2  # FORMULA CORRECTA PARA RBF  !!!
        # print(f'{time=}')
        # print(f'{gradient=}')
        # print(f'{gradient_test=}')
        # print(f'{(gradient-gradient_test).sum()=}')
        # print(f'{delta.shape=}')
        if self.verbose == 1:
            print(f'Mean antes:{mean:.1f} Delta: {delta:.1f} Gradient: {np.sum(gradient):.1f}', end=' ')
        # print(f'{(policy_step_size * gamma**time * delta * gradient)[0]}')
        # self._mean_approximation.mean[index] += decimal.Decimal(policy_step_size) * decimal.Decimal(gamma**time) * delta * gradient
        # print(f'{policy_step_size=} {delta=} {gradient=} step: {policy_step_size * gamma**time * delta * gradient}')
        self._mean_approximation.mean_weights += policy_step_size * gamma**time * delta * gradient
        # step = np.clip(policy_step_size * gamma**time * delta * gradient, -300, 300)
        # self._mean_approximation.mean[index] += step
        # self._mean_approximation.mean[index] += policy_step_size * delta * gradient
        # self._mean_approximation.mean[index] = np.clip(self._mean_approximation.mean[index], decimal.Decimal(0), decimal.Decimal(680))
        # TODO: clips?????????
        ################
        # self._mean_approximation.mean_weights = np.clip(self._mean_approximation.mean_weights, 0, 680)  # INCORRECTO EN RBF !!!
        self._mean_approximation.mean_weights = np.clip(self._mean_approximation.mean_weights, 0, self._mean_approximation.state_discretization.policy_scale_factor.reshape((self._mean_approximation.mean_weights.shape[0], 1)))
        # mean_after = self._mean_approximation.mean_from_state(state)
        # # min_cap = min(0, mean_after)
        # if mean_after > 680:
        #     # self._mean_approximation.mean_weights *= state * 680 / mean_after
        #     self._mean_approximation.mean_weights *= np.divide(state, state, where=state != 0) * 680 / mean_after
        # if mean_after < 0:
        #     self._mean_approximation.mean_weights *= np.divide(state, state, where=state != 0) * 680 / mean_after
        ################
        # self._mean_approximation.mean[index] -= decimal.Decimal(policy_step_size) * decimal.Decimal(gamma**time) * delta * gradient
        if self.verbose == 1:
            print(f'Mean desp:{self._mean_approximation.mean_from_state(state):.1f}')
        # mean_after = self._mean_approximation.mean_from_state(state)
        # return mean_after != mean_before
        return True
