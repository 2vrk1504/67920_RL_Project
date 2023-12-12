import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.distributions import Categorical
import torch.nn.functional as F


class IIDChannel:
    def __init__(self, p):
        self.p = p

    def sample(self):
        return int(np.random.rand() < self.p)

class LinearGaussianSystem:
    def __init__(self, n, A, Q, name, sensor_noises = None, channel_on_probabilities = None):
        self.n = n
        self.A = A # system matrix
        self.Q = Q # covariance matrix
        self.P = Q # initialization
        self.sensor_ages = np.zeros(n)
        self.name = name # name the system
        if sensor_noises is not None:
            self.sensor_noises = sensor_noises
        else:
            self.sensor_noises = np.zeros(n)

        if channel_on_probabilities is not None:
            self.channels = [IIDChannel(channel_on_probabilities[i]) for i in range(n)]
        else:
            self.channels = [IIDChannel(1) for i in range(n)]

    def reset(self, method = 'Random'):
        self.sensor_ages = np.zeros(self.n)
        if method == 'Deterministic':
            self.P = self.Q
        elif method == 'Random':
            self.P = self.Q
            actions = np.arange(self.n)
            np.random.shuffle(actions)
            for t in range(self.n):
                self.step(actions[t])
            for t in range(1):
                action = np.random.randint(self.n)
                self.step(action)
        elif method == 'ControlRandom':
            self.P = self.Q
            actions = np.arange(self.n)
            T = np.random.randint(1,3)
            for t in range(T):
                np.random.shuffle(actions)
                for t in range(self.n):
                    self.step(actions[t])
        return self.P

    def step(self, action):
        A = self.A
        P = self.P
        e = np.zeros((self.n,1))
        e[action] = 1

        # sample channel
        u = self.channels[action].sample()
        self.sensor_ages = self.sensor_ages + 1
        self.sensor_ages[action] = (1-u)*self.sensor_ages[action]

        Pe = P.dot(e)
        P_aposteriori = np.around(P - u*(Pe.dot(Pe.T))/(P[action][action] + self.sensor_noises[action]), decimals=3)
        cost = np.trace(P_aposteriori)

        # state update
        self.P = A.dot(P_aposteriori).dot(A.T) + self.Q

        if np.isnan(self.P).any() or np.isinf(self.P).any():
            print(P_aposteriori)
            raise ValueError

        return self.P, self.sensor_ages, cost, u

def policy_rollout(system, algorithm, T, model=None, first_action=None):
    '''
    This function takes the algorithm (function of state) and evaluates its performance over a time horizon
    system: an instance of System
    algorithm: function which takes state as argument and returns an action, has to return categorical
    T: Time horizon
    gamma: discount factor
    '''
    states = np.zeros((T+1,system.n,system.n))
    states[0] = system.reset(method='Random')
    # print(states[0])
    costs = torch.zeros(T)
    actions = torch.zeros(T, dtype=torch.int)
    log_probs = torch.zeros(T)
    values = torch.zeros(T)
    total_cost = 0

    start = 0
    if first_action is not None:
        actions[0] = first_action
        states[1], _, costs[0], _ = system.step(first_action)
        start = 1

    for t in range(start, T):
        if model is None:
            dist, values[t] = algorithm(system.P, system.sensor_ages, system.A, system.Q, system.n)
        else:
            log_dist, values[t] = algorithm(model, system.P, system.A, system.Q, system.n)
            try:
                dist = Categorical(torch.exp(log_dist))
            except ValueError as e:
                print(system.P)
                # print("Bhai ", F.normalize(torch.nn.Flatten()(torch.tensor(system.P)), p=2))
                print(log_dist)
                print(values[t])
                raise e
            
        actions[t] = dist.sample()
        log_probs[t] = dist.log_prob(actions[t])
        states[t+1], _, costs[t], _ = system.step(actions[t])
        total_cost += costs[t]

    return actions, costs, log_probs, values, total_cost, states



