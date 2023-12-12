import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from tqdm import tqdm

from system_module import *
from neural import *
from learning_algos import *
from utils import *

TRAIN_POLICY = True
n = 5
A = np.diag([1.1, 1.2, 1.3, 1.4, 2])
# A = np.eye(n)
# A = 5*np.eye(n)
# rho = 0.8
# U = np.random.randn(n,n)
# Q = np.eye(n) + U.dot(U.T) # random Gaussian matrix
# Q = n*Q/np.trace(Q) # normalize so that tr(Q) scales as n
# Q = rho*np.ones((n,n)) + (1-rho) * np.eye(n)

# # fixed Q5
Q = np.array([[ 1.30821331, -0.08839342,  0.27463018,  1.26843693, -0.00212539],
       [-0.08839342,  0.42500649, -0.25027442, -0.05258601,  0.11878064],
       [ 0.27463018, -0.25027442,  0.85715141,  0.51733964, -0.16188765],
       [ 1.26843693, -0.05258601,  0.51733964,  1.93656689,  0.22707067],
       [-0.00212539,  0.11878064, -0.16188765,  0.22707067,  0.4730619 ]])

channel_on_probabilities = [0.9, 0.9, 0.8, 0.7, 0.9]
# channel_on_probabilities = [1, 1, 1, 1, 1]

# system = LinearGaussianSystem(n, A, Q, "diagonalA_111213142_randomQ_greedy_pre_train_with_GAE", None, channel_on_probabilities) 
system = LinearGaussianSystem(n, A, Q, "diagonalA_111213142_randomQ_greedy_pre_train_with_GAE_wp", None, channel_on_probabilities) 
# system = LinearGaussianSystem(n, A, Q, "discrete_Wiener_n10_rho0.8_greedy_pre_train_with_GAE", None, channel_on_probabilities) 
# system = LinearGaussianSystem(n, A, Q, "discrete_Wiener_n10_rho0.8_with_GAE", None, channel_on_probabilities) 

gamma = 0.999


model_params = {
    'n': n,
    'policy_hidden_dim': 2*n*n,
    'policy_layer': 3,
    'value_hidden_dim': 2*n*n,
    'value_layer': 2
}

# greedy policy
def greedy(state):
    n = state.shape[0]
    # vector = np.zeros(n)
    vector = np.diag(state)
    dist = np.zeros(n)
    dist[np.argmax(vector)] = 1
    return dist

def train_pv_network(system, model_params, policy, episode_length, max_episodes, log_interval, step_save):
    lr = 1e-4
    
    ## Model    
    model = PolicyValueNetwork(model_params['n'], policy_hidden_dim = model_params['policy_hidden_dim'],
                                policy_layer = model_params['policy_layer'], value_hidden_dim = model_params['value_hidden_dim'], 
                                value_layer=model_params['value_layer'], lr=lr, name=None)
    if TRAIN_POLICY:
        ret = greedy_policy_pretrain(model.policy_model, model.policy_optimizer, system, greedy, episode_length, max_episodes//10)
        if not ret:
            return False, None
    ret = value_net_bootstrapping(model.value_model, policy, model.value_optimizer, system, gamma, episode_length, max_episodes//10, model.policy_model)
    if not ret:
        return False, None

    # Now we do actor critic
    episode_costs = actor_critic(model, system, gamma, episode_length, max_episodes, gae_lam=0.9)
    return True, episode_costs

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

def whittle_index_wp(state, ages, A, Q, n):
    aii2 = np.diag(A) ** 2
    qii = np.diag(Q)
    pr = np.array(channel_on_probabilities)
    vector1 =  (pr**2) * qii * (ages + 1) / (aii2 - 1)
    vector2 = 1 / (1 - pr)
    vector3 = aii2 ** (ages + 1) / (aii2 * (1-pr))

    vector4 = np.zeros(n)
    for i in range(n):
        for k in range(int(ages[i])):
            vector4[i] += ((aii2[i])**k - 1)/(aii2[i] - 1) * qii[i]

    vector = vector1 * (vector3 - vector2) - pr * vector4
    dist = torch.zeros(n)
    dist[torch.argmax(torch.tensor(vector))] = 1
    return Categorical(dist), torch.tensor(0)



def mee(state, ages, A, Q, n):
    pii = np.diag(state)
    qii = np.diag(Q)
    pr = np.array(channel_on_probabilities)
    vector = pii/np.sqrt(qii * pr) 
    dist = torch.zeros(n)
    dist[torch.argmax(torch.tensor(vector))] = 1
    return Categorical(dist), torch.tensor(0)

def greedy_cat(state, ages, A, Q, n):
    dist = greedy(state)
    return Categorical(torch.tensor(dist)), torch.tensor(0)

def model_policy(model, state, A, Q, n):
    return model(normalize_input(state)), torch.tensor(0)

is_training = True
load_step = 'max'
if len(sys.argv) > 1:
    if sys.argv[1] == 'no_train':
        is_training = False
    if len(sys.argv) > 2:
        load_step = int(sys.argv[2])

EPISODE_LENGTH = n * 500
MAX_EPISODES = 1000
LOG_INTERVAL = 10
STEP_SAVE = 20
LR = 1e-4

if is_training:
    
    mean_episode_costs = np.zeros(MAX_EPISODES)
    var_episode_costs = np.zeros(MAX_EPISODES)
    num_iters = 10
    for iters in range(num_iters):
        supervised_learning_worked = False
        while not supervised_learning_worked: 
            supervised_learning_worked, episode_costs = train_pv_network(system, model_params, model_policy, EPISODE_LENGTH, MAX_EPISODES, LOG_INTERVAL, STEP_SAVE)
            mean_episode_costs += episode_costs/num_iters
            var_episode_costs += episode_costs**2 / (num_iters - 1)


model = PolicyValueNetwork(model_params['n'], policy_hidden_dim = model_params['policy_hidden_dim'],
                                policy_layer = model_params['policy_layer'], value_hidden_dim = model_params['value_hidden_dim'], 
                                value_layer=model_params['value_layer'], lr=LR, name=None)
load_pv_checkpoint(model, step=load_step, save_dir="./"+system.name+"_"+model.name)

T = 10000
actions_whittle, costs_whittle, _, _, _, _ = policy_rollout(system, whittle_index_wp, T, model=None)
# actions_mee, costs_mee, _, _, _, _ = policy_rollout(system, mee, T, model=None)
actions_greedy, costs_greedy, _, _, _, _ = policy_rollout(system, greedy_cat, T, model=None)
actions_model, costs_model, _, _, _, _ = policy_rollout(system, model_policy, T, model=model.policy_model)


total_cost_whittle = torch.mean(costs_whittle[100:]).item()
# total_cost_mee = torch.mean(costs_mee[100:]).item()
total_cost_greedy = torch.mean(costs_greedy[100:]).item()
total_cost_model = torch.mean(costs_model[100:]).item()

print("Whittle: {:.2f}".format(total_cost_whittle))
# print("MEE: {:.2f}".format(total_cost_mee))
print("Greedy: {:.2f}".format(total_cost_greedy))
print("Model: {:.2f}".format(total_cost_model))

st_err = np.sqrt(var_episode_costs - mean_episode_costs**2)

plt.figure()
plt.plot(total_cost_whittle * np.ones(len(episode_costs)), label="Whittle Index")
# plt.plot(total_cost_mee * np.ones(len(mean_episode_costs)), label="MEE")
plt.plot(total_cost_greedy * np.ones(len(mean_episode_costs)), label="Greedy")
plt.plot(mean_episode_costs, 'g-', label="Cost as we Learn")
plt.fill_between(np.arange(len(mean_episode_costs)), mean_episode_costs-st_err, mean_episode_costs + st_err, facecolor='green', alpha=0.2)
plt.xlabel("Episode")
plt.ylabel("Cost")
text = " PT " if TRAIN_POLICY else ""
text2 = " DG " if not np.all(np.diag(A) == 1) else ""
plt.title("Actor Critic Training" + text + text2)
plt.grid(True)
plt.legend()

# MA = 100

# plt.figure()
# # plt.plot(np.convolve(costs_whittle, np.ones(MA))[:-MA]/MA, label="Whittle")
# plt.plot(np.convolve(costs_mee, np.ones(MA))[:-MA]/MA, label="MEE")
# plt.plot(np.convolve(costs_greedy, np.ones(MA))[:-MA]/MA, label="Greedy")
# plt.plot(np.convolve(costs_model, np.ones(MA))[:-MA]/MA, label="Learned Policy")
# plt.title("Smoothed Cost")
# plt.ylabel("Expected Estimation Error")
# plt.xlabel("Time")
# plt.grid(True)
# plt.legend()
plt.show()