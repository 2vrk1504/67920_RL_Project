import numpy as np
import matplotlib.pyplot as plt
import torch


class SensorSystem:
	def __init__(self, n, A, Q, c, channel_on_probabilites = None):
		self.n = n
		self.A = A # system dynamics matrix
		self.Q = Q # covariance matrix
		self.Qev, self.U = np.linalg.eig(self.Q)
		self.state = (self.U.T).dot((self.Qev ** 0.5 ) * np.random.randn((n,1))) # instantaneous error at  sensors
		self.ages = np.zeros((n,1)) # time since each sensor was scheduled

		self.c = c # cost of transmission

		if channel_on_probabilities is not None:
			self.channels = [IIDChannel(channel_on_probabilities[i]) for i in range(n)]
		else:
			self.channels = [IIDChannel(1) for i in range(n)]

	def reset(self, method = 'Random'):
		self.state = (self.U.T).dot((self.Qev ** 0.5) * np.random.randn((n,1)))
		return self.state

	def step(self, actions):
		'''
			actions: action of each sensor -> 1 = decide to transmit, 0 = no transmit
		'''
		cost = np.zeros((n,1))
		tx_idx = np.asarray(actions == 1)
		num_tx_sensors = np.sum(tx_idx) 
		
		noise = (self.U.T).dot((self.Qev ** 0.5 ) * np.random.randn((n,1)))
		if num_tx_sensors > 1: # number of transmitting sensors
			cost[tx_idx] = num_tx_sensors * self.c
		else:
			cost[tx_idx] =  c
			self.state[tx_idx] = 0 # sampled 

		self.state = self.A.dot(self.state) + noise
		cost = cost + np.abs(self.state)**2
		return self.state, cost


def evaluate_algorithm(system, algorithm, T, gamma):
	'''
	This function takes the algorithm (function of state) and evaluates its performance over a time horizon
	system: an instance of System
	algorithm: function which takes state as argument and returns an action
	T: Time horizon
	gamma: discount factor
	'''
	curr_state = system.reset(method='Random')
	cost_trajectory = np.zeros(T)
	total_cost = 0
	for t in range(T):
		action = algorithm(curr_state)
		next_state, cost_trajectory[t], _ = system.step(action)
		total_cost += (gamma ** t) * cost_trajectory

	return total_cost, cost_trajectory

n = 10
A = np.eye(n)
rho = 0.8
Q = rho*np.ones((n,n)) + (1-rho) * np.eye(n)

system = LinearGaussianSystem(n, A, Q)

def max_weight(state):
	