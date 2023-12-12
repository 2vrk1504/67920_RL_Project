import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import OrderedDict

import numpy as np


torch.set_default_dtype(torch.double)

def zero_init(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # apply a uniform distribution to the weights and a bias=0
        m.weight.data.uniform_(0.0, 1e-40)
        m.bias.data.fill_(1)

class PolicyNetwork(nn.Module):
    def __init__(self, n, hidden_dim = 64, layer = 2, name="PolicyNetwork"):
        """
        Initialize the parameter for the policy network
        """
        super(PolicyNetwork, self).__init__()
        # Note: here we are given the in_dim and out_dim to the network
        self.n = n
        in_dim = int(n*(n+1)/2) #state is covariance and ages
        out_dim = n

        self.name = name

        if layer == 0:
          hidden_dim = out_dim
        self.layer_list = [ ("lin-in", nn.Linear(in_dim, hidden_dim))]
        for i in range(layer):
          self.layer_list.append(("ReLU-"+str(i), nn.ReLU()))
          if i < layer - 1:
            self.layer_list.append(("lin-"+str(i), nn.Linear(hidden_dim, hidden_dim)))
          else:
            self.layer_list.append(("lin-"+str(i), nn.Linear(hidden_dim, out_dim)))

        # layer_list.append(("Softmax-out", nn.LogSoftmax(dim=0)))
        self.model = nn.Sequential(OrderedDict(self.layer_list))

    def forward(self, state):
        """
        This function takes in a batch of observations and a batch of actions, and
        computes a probability distribution (Categorical) over all (discrete) actions

        observation: shape (batch_size, observation_size) torch Tensor

        return: a categorical distribution over all possible actions. You may find torch.distributions.Categorical useful
        """
        #### Your code here
        if len(state.shape) > 2:
            x = np.zeros((state.shape[0], int(self.n*(self.n+1)/2)))
            for i in range(state.shape[0]):
                x[i, :] = state[i][np.triu_indices(self.n)]
            
        else:
            x = np.zeros(0)
            for i in range(self.n):
                x = np.append(x, state[i, i:])

        x = torch.tensor(x, dtype=torch.double)
        out = self.model(x)
        if len(x.size()) > 1:
            pi = F.log_softmax(out, dim=1)
        else:
            pi = F.log_softmax(out, dim=0)
        return pi

class ValueNetwork(nn.Module):
    def __init__(self, n, hidden_dim = 64, layer=1, name=None):
        """
        Initialize the parameter for the value function
        """
        super(ValueNetwork, self).__init__()
        #### Your code here
        self.n = n
        in_dim = int(n*(n+1)/2)
        layer_list = [ ("lin-in", nn.Linear(in_dim, hidden_dim)), ("ReLU-in", nn.ReLU())]
        for i in range(layer):
          if i < layer - 1:
            layer_list.append(("lin-"+str(i), nn.Linear(hidden_dim, hidden_dim)))
          else:
            layer_list.append(("lin-"+str(i), nn.Linear(hidden_dim, 1)))
          layer_list.append(("ReLU-"+str(i), nn.ReLU()))
        self.model = nn.Sequential(OrderedDict(layer_list))
        # self.model.apply(weights_init_uniform)


    def forward(self, state):
        """
        This function takes in a batch of observations, and
        computes the corresponding batch of values V(s)

        observation: shape (batch_size, observation_size) torch Tensor

        return: shape (batch_size,) values, i.e. V(observation)
        """
        #### Your code here
        if len(state.shape) > 2:
            x = np.zeros((state.shape[0], int(self.n*(self.n+1)/2)))
            for i in range(state.shape[0]):
                x[i, :] = state[i][np.triu_indices(self.n)]
        else:
            x = np.zeros(0)
            for i in range(self.n):
                x = np.append(x, state[i, i:])
        
        x = torch.tensor(x, dtype=torch.double)
        V = self.model(x)
        return V

class QNetwork_withValue():
    def __init__(self, n, hidden_dim = 64, layer=1, lr=1e-4, name="QNetwork"):
        self.n = n
        self.name = name
        self.models = [ValueNetwork(n, hidden_dim, layer) for i in range(n)]
        self.optimizers = [optim.Adam(self.models[i].parameters(), lr=lr) for i in range(n)]
        params = []
        for i in range(n):
            params += list(self.models[i].parameters())
        self.opt = optim.Adam(params, lr=lr)

    def evaluate(self, state, action):
        model = self.models[action]
        V = model(state)
        return V

    def evaluate_all(self, state):
        '''
            Only returns the value, detached from graph
        '''
        Q = []
        for i in range(self.n):
            model = self.models[i]
            Q.append(model(state).detach().item())
        return np.array(Q)
            
class QNetwork(nn.Module):
    def __init__(self, n, hidden_dim = 64, layer=1, name=None):
        """
        Initialize the parameter for the value function
        """
        super(QNetwork, self).__init__()
        #### Your code here
        self.n = n
        self.name = "PVNet"+"_HiddenDim"+str(hidden_dim)+"_Layer"+str(layer)
        in_dim = int(n*(n+1)/2) + 1
        layer_list = [ ("lin-in", nn.Linear(in_dim, hidden_dim)), ("ReLU-in", nn.ReLU())]
        for i in range(layer):
          if i < layer - 1:
            layer_list.append(("lin-"+str(i), nn.Linear(hidden_dim, hidden_dim)))
          else:
            layer_list.append(("lin-"+str(i), nn.Linear(hidden_dim, 1)))
          layer_list.append(("ReLU-"+str(i), nn.ReLU()))
        self.model = nn.Sequential(OrderedDict(layer_list))
        # self.model.apply(weights_init_uniform)
        self.opt = optim.Adam(self.model.parameters(), lr=1e-3)


    def forward(self, state, action):
        """
        This function takes in a batch of observations, and
        computes the corresponding batch of values V(s)

        observation: shape (batch_size, observation_size) torch Tensor

        return: shape (batch_size,) values, i.e. V(observation)
        """
        #### Your code here
        if len(state.shape) > 2:
            x = np.zeros((state.shape[0], int(self.n*(self.n+1)/2 + 1)))
            for i in range(state.shape[0]):
                x[i, :-1] = state[i][np.triu_indices(self.n)]
                x[i, -1] = action[i]
        else:
            x = np.zeros(0)
            for i in range(self.n):
                x = np.append(x, state[i, i:])
            x = np.append(x, action)
        
        x = torch.tensor(x, dtype=torch.double)
        Q = self.model(x)
        return Q

    def update_lr(self, lr):
        self.opt = optim.Adam(self.model.parameters(), lr=lr)        

    def evaluate_all(self, state):
        x = np.zeros(0)
        for i in range(self.n):
            x = np.append(x, state[i, i:])
        
        outs = np.zeros(self.n)
        for action in range(self.n):
            input_state = np.concatenate((x, [action]))
            input_state = torch.tensor(input_state, dtype=torch.double)
            outs[action] = self.model(input_state).detach().item()

        return outs

class PolicyValueNetwork():
    def __init__(self, n, policy_hidden_dim= 64, policy_layer = 2, value_hidden_dim= 64, value_layer = 2, lr=1e-3, name=None):
        if name == None:
            self.name = "PVNet"+"_PolicyHiddenDim"+str(policy_hidden_dim) \
                        +"_Layer"+str(policy_layer)+"_ValueHiddenDim"\
                        +str(value_hidden_dim)+"_Layer"+str(value_layer)
        else:
            self.name = name

        self.n = n
        self.policy_model = PolicyNetwork(n, policy_hidden_dim, policy_layer, name)
        self.value_model = ValueNetwork(n, value_hidden_dim, value_layer, name)
        self.policy_optimizer = optim.Adam(list(self.policy_model.parameters()), lr=lr)
        self.value_optimizer = optim.Adam(list(self.value_model.parameters()), lr=lr)

    def update_value_optimizer(self, lr):
        self.value_optimizer = optim.Adam(list(self.value_model.parameters()), lr=lr)

    def update_policy_optimizer(self, lr):
        self.policy_optimizer = optim.Adam(list(self.policy_model.parameters()), lr=lr)
