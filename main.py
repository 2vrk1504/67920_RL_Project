import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from system_module import *
from neural import *
from learning_algos import *
from utils import *

import time


n = 5
# A = np.diag([1.1, 1.2, 1.3, 1.4, 2])
A = np.eye(n)
# A = 5*np.eye(n)
# rho = 0.8
# U = np.random.randn(n,n)
# Q = np.eye(n) + U.dot(U.T) # random Gaussian matrix
# Q = n*Q/np.trace(Q) # normalize so that tr(Q) scales as n
# Q = rho*np.ones((n,n)) + (1-rho) * np.eye(n)
Q = np.array([[ 1.30821331, -0.08839342,  0.27463018,  1.26843693, -0.00212539],
       [-0.08839342,  0.42500649, -0.25027442, -0.05258601,  0.11878064],
       [ 0.27463018, -0.25027442,  0.85715141,  0.51733964, -0.16188765],
       [ 1.26843693, -0.05258601,  0.51733964,  1.93656689,  0.22707067],
       [-0.00212539,  0.11878064, -0.16188765,  0.22707067,  0.4730619 ]])


system = LinearGaussianSystem(n, A, Q, "discrete_Wiener_n5_rho08" ) 
gamma = 0.999

hidden_dim = 2*n*n + 2*n
layer = 3


def greedy(state):
    n = state.shape[0]
    # vector = np.zeros(n)
    vector = np.diag(state)
    dist = np.zeros(n)
    dist[np.argmax(vector)] = 1
    return dist

def greedy_cat(state, ages, A, Q, n):
    dist = greedy(state)
    return Categorical(torch.tensor(dist)), torch.tensor(0)

def train_Q_network_for_system(system, policy, episode_length, max_episodes, log_interval, step_save, lr):
    n = system.n
    Qmodel = QNetwork(n, hidden_dim=hidden_dim,layer=layer, name="QModel")

    ret = q_model_bootstrapping(Qmodel, policy, system, gamma, episode_length, max_episodes//10)
    if not ret:
        return False, None, None

    eps = 0.8

    episode_costs = np.zeros(max_episodes+1)
    TD_history, episode_costs[0] = Q_learning(episode_length, system, Qmodel, gamma, policy='eps_greedy', eps=eps, device=None)
    loss_estimate = np.mean(TD_history)
    running_error = loss_estimate
    TD_errors = []

    iir = 0.1
    history_costs = [running_error]
    
    for step in range(max_episodes):

        TD_history, episode_costs[step+1] = Q_learning(episode_length, system, Qmodel, gamma, policy='eps_greedy', eps=eps, device=None)
        loss_estimate = np.mean(TD_history)
        TD_errors.append(np.mean(TD_history))

        running_error = iir * loss_estimate + (1 - iir) * running_error

        history_costs.append(running_error)
        if step > 20 and lr > 1e-5:
            if np.abs(np.mean(history_costs[:-1]) - np.mean(history_costs[1:])) < 10*lr:
                lr /= 10
                Qmodel.update_lr(lr)
            history_costs.pop(0)

        if step > 500:
            eps *= step/(step + 1)

        if step % log_interval == 0:
            print('Episode {}\tLast TD variance: {:.2f} \tAverage error: {:.2f}'.format(step, loss_estimate, running_error))

        if step % step_save == 0:
            # Saves model checkpoint
            save_checkpoint(Qmodel, Qmodel.opt, step, save_dir="./"+system.name+"_"+Qmodel.name)

    save_checkpoint(Qmodel, Qmodel.opt, step, save_dir="./"+system.name+"_"+Qmodel.name)
    return True, episode_costs, TD_errors

## Policies
def whittle_index(state, ages, A, Q, n):
    aii2 = np.diag(A) ** 2
    qii = np.diag(Q)

    vector = qii * (ages + 1) * ((aii2 + 1e-10) ** (ages + 1) - 1)/(aii2 + 1e-10 - 1)
    for i in range(n):
        for k in range(int(ages[i])):
            vector[i] += ((aii2[i] + 1e-10)**k - 1)/(aii2[i] + 1e-10 - 1) * qii[i]
    dist = torch.zeros(n)
    dist[torch.argmax(torch.tensor(vector))] = 1
    return Categorical(dist), torch.tensor(0)

def mee(state, ages, A, Q, n):
    pii = np.diag(state)
    qii = np.diag(Q)
    # pr = np.array(channel_on_probabilities)
    vector = pii/np.sqrt(qii ) 
    dist = torch.zeros(n)
    dist[torch.argmax(torch.tensor(vector))] = 1
    return Categorical(dist), torch.tensor(0)

def Q_model_policy(Qmodel, state, A, Q, n):
    Qvals = Qmodel.evaluate_all(state)
    # action = np.random.randint(n)
    dist = -torch.inf * torch.ones(n)
    dist[argmin(Qvals)] = 0
    # dist[action] = 0
    return dist, torch.tensor(0)

is_training = True
load_step = 'max'
if len(sys.argv) > 1:
    if sys.argv[1] == 'no_train':
        is_training = False
    if len(sys.argv) > 2:
        load_step = int(sys.argv[2])

if is_training:
    EPISODE_LENGTH = n * 200
    MAX_EPISODES = 10000
    LOG_INTERVAL = 10
    STEP_SAVE = 20
    supervised_learning_worked = False
    start = time.time()
    
    while not supervised_learning_worked: 
        supervised_learning_worked, episode_costs, TD_errors = train_Q_network_for_system(system, greedy_cat, EPISODE_LENGTH, MAX_EPISODES, LOG_INTERVAL, STEP_SAVE, 1e-4)

    end = time.time()
    print("Time elapsed: {}".format(end - start))

np.savetxt("Q_training_costs.csv", episode_costs, delimiter=",")
np.savetxt("Q_training_TD_error.csv", TD_errors, delimiter=",")

# episode_costs = np.genfromtxt("Q_training_costs.csv", delimiter=",")

model = QNetwork(n, hidden_dim=hidden_dim,layer=layer, name=None)
load_checkpoint(model, step=load_step, save_dir="./discrete_wiener_n5_rho08_PVNet_HiddenDim60_Layer3")

T = 10000
# actions_whittle, costs_whittle, _, _, _, _ = policy_rollout(system, whittle_index, T, model=None)
actions_mee, costs_mee, _, _, _, _ = policy_rollout(system, mee, T, model=None)
actions_greedy, costs_greedy, _, _, _, _ = policy_rollout(system, greedy_cat, T, model=None)
actions_model, costs_model, _, _, _, _ = policy_rollout(system, Q_model_policy, T, model=model)



# total_cost_whittle = torch.mean(costs_whittle[100:]).item()
total_cost_mee = torch.mean(costs_mee[100:]).item()
total_cost_greedy = torch.mean(costs_greedy[100:]).item()
total_cost_model = torch.mean(costs_model[100:]).item()

# print("Whittle: {:.2f}".format(total_cost_whittle))
print("MEE: {:.2f}".format(total_cost_mee))
print("Greedy: {:.2f}".format(total_cost_greedy))
print("Model: {:.2f}".format(total_cost_model))

plt.figure()
# plt.plot(total_cost_whittle * np.ones(len(episode_costs)), label="Whittle Index")
plt.plot(total_cost_mee * np.ones(len(episode_costs)), label="MEE")
plt.plot(total_cost_greedy * np.ones(len(episode_costs)), label="Greedy")
plt.plot(episode_costs, label="Cost as we Learn")
plt.xlabel("Episode")
plt.ylabel("Cost")
plt.title("Q Learning")
plt.grid(True)
plt.legend()

MA = 100

plt.figure()
# plt.plot(np.convolve(costs_whittle, np.ones(MA))[:-MA]/MA, label="Whittle")
plt.plot(np.convolve(costs_mee, np.ones(MA))[:-MA]/MA, label="MEE")
plt.plot(np.convolve(costs_greedy, np.ones(MA))[:-MA]/MA, label="Greedy")
plt.plot(np.convolve(costs_model, np.ones(MA))[:-MA]/MA, label="Learned Policy")
plt.title("Smoothed Cost")
plt.ylabel("Expected Estimation Error")
plt.xlabel("Time")
plt.grid(True)
plt.legend()
plt.show()