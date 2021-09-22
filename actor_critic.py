import numpy as np
import abc

class Environment:

    def __init__(self, state):
        self._state = state

    def reset(self):
        pass

    def step(action):
        pass

class StateSpace(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def reset(self):
        pass

    @abc.abstractmethod
    def step(self):
        pass

class DiscreteStateSpace(StateSpace):

    def __init__(self, min_states, max_states, num_buckets):
        self.min_states = min_states
        self.max_states = max_states
        self.num_buckets = num_buckets
        self._value = np.zeros((num_buckets))

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value

    def reset(self):
        self._value = np.zeros((self.num_buckets))

    def step(self, action):
        pass


# class MeanAproximation:


# class GaussianPolicy:
#     # Devuelve una accion siguiendo la politica gaussiana
#     def act(self, state):
#         state = self.flatten_state(state)
#         action = np.random.normal(self.mean_policy[state], self.sigma)
#         return [action]


class ActorCritic:
    """
    Clase que implementa el algoritmo Actor-critic de un paso.

    Atributos:
        -
    """

    def __init__(self, min_state, gamma=0.99, sigma=0.5):
        self.min_state = min_state
        self.max_state = max_state
        self.value = np.zeros((np.prod(num_buckets)))
        self.mean_policy = np.zeros(np.prod(num_buckets))

        self.n_grids = num_buckets
        self.res = (self.max_state - self.min_state).astype(float) / (self.n_grids - 1)

        self.gamma = gamma
        self.sigma = sigma

    # Devuelve una accion siguiendo la politica gaussiana
    def act(self, state):
        state = self.flatten_state(state)
        action = np.random.normal(self.mean_policy[state], self.sigma)
        return [action]

    def gradient_step(self, state, action, reward, next_state, value_step_size, policy_step_size, time):
        state = self.flatten_state(state)
        next_state = self.flatten_state(next_state)

        delta = reward + self.gamma * self.value[next_state] - self.value[state]
        self.value[state] += value_step_size * delta
        self.mean_policy[state] += policy_step_size * (self.gamma**time) * delta * ((action - self.mean_policy[state]) / self.sigma**2)

    # def flatten_state(self, state):
    #     # dado un state de dimension (3,), devuelve su representacion unidimensional
    #     idx = ((np.array(state) - self.min_state) / self.res).astype(int).flatten()
    #     flattened = np.ravel_multi_index(idx, dims=self.n_grids)
    #     return flattened

    def train(env, model, gamma, value_step_size, policy_step_size, num_episodes=1000):
        rewards = []
        print_interval = 100
        rewardsARetornar = []
        for n in range(num_episodes):
            state = env.reset()
            done = False
            t = -1
            while not done:
                t += 1
                action = model.act(state)
                next_state, reward, done, _ = env.step(action)
                model.gradient_step(state, action, reward, next_state, value_step_size, policy_step_size, t)
                rewards.append(reward)
                state = next_state
            rewardsARetornar.append(np.sum(rewards) / print_interval)
            if n % print_interval == 0 and n > 0:
                print("Episode " + str(n) + ": " + str(np.sum(rewards) / print_interval))
                rewards = []
        return rewardsARetornar
