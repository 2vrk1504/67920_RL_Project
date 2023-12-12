import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import torch.optim as optim

from system_module import *
from utils import *
from neural import ValueNetwork

def argmin(b):
    return np.random.choice(np.where(b == b.min())[0])

def normalize_input(x):
    return x/np.max(x)

def discounted_returns(costs, gamma):
    """
        Compute the cumulative discounted return with discount factor gamma from the immediate rewards
    """
    returns = torch.zeros_like(costs)
    T = returns.size()[0]
    gammas = gamma ** torch.arange(T)
    for t in range(T):
      returns[t] = torch.sum(costs[t:T] * gammas[:T-t])
    return returns

def update_parameters_policy_gradient(optimizer, costs, log_probs, gamma, values=None):
    # compute policy losses
    returns = discounted_returns(costs, gamma)
    if values != None:
        value_loss = F.smooth_l1_loss(values, returns)

    if values != None:  # use the value function as the baseline
        returns = returns - values.detach()  # this is the advantage

    gammas = gamma ** torch.arange(log_probs.size()[0])
    policy_loss = torch.sum(log_probs * returns * gammas, dim=0)

    if values != None:
        loss = policy_loss + value_loss
    else:
        loss = policy_loss

    # parameter update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def Q_learning(episode_length, system, Qmodel, gamma, policy='eps_greedy', eps=0.3, td_lam_param = 0, device=None):
    # compute policy losses
    n = system.n
    system.reset('Random')
    TD_history = [] 

    costs = torch.zeros(episode_length)
    values = torch.zeros(episode_length+1)
    deltas = torch.zeros(episode_length)
    actions = torch.zeros(episode_length, dtype=torch.int)

    for t in range(episode_length):
        qvals_t = Qmodel.evaluate_all(system.P)  
        if np.random.rand() > eps:
            actions[t] = argmin(qvals_t)
        else:
            actions[t] = np.random.randint(n)
       
        values[t] = Qmodel(system.P, actions[t])
        _, _, costs[t], _ = system.step(actions[t])
        qvals_t1 = np.min(Qmodel.evaluate_all(system.P))
        deltas[t] = costs[t] + gamma * qvals_t1 - values[t]

    episode_cost = torch.mean(costs[100:]).item()

    loss = 0.5 * torch.mean(deltas ** 2) 
    Qmodel.opt.zero_grad()
    loss.backward()
    Qmodel.opt.step()

    return loss.item(), episode_cost

def q_model_bootstrapping(qmodel, policy, system, gamma, episode_length, max_episodes, log_interval=10, batch_size=100, training_steps=10000, lr=1e-4):
    replay_bank_states = []
    replay_bank_actions = []
    replay_bank_returns = []

    for first_action in range(system.n):
        for step in tqdm(range(max_episodes//system.n)):
            actions, costs, log_probs, values, total_cost, states = policy_rollout(system, policy, episode_length, first_action=first_action)
            returns = discounted_returns(costs, gamma)
            for i in range(episode_length//100):
                replay_bank_states.append(states[i])
                replay_bank_actions.append(actions[i])
                replay_bank_returns.append([returns[i]])

    # learning V^pi 
    print("Learning Q Network...")
    replay_bank_states = np.array(replay_bank_states)
    replay_bank_actions = np.array(replay_bank_actions)
    replay_bank_returns = np.array(replay_bank_returns)
    init_loss = 0
    pbar = tqdm(range(training_steps))
    for step in pbar:
        idx = np.random.choice(np.arange(len(replay_bank_states)), size=batch_size)
        batch_input_states = torch.tensor(replay_bank_states[idx])
        batch_input_actions = torch.tensor(replay_bank_actions[idx])
        batch_outputs = torch.tensor(replay_bank_returns[idx])

        model_outputs = qmodel(batch_input_states, batch_input_actions)
        loss = F.smooth_l1_loss(model_outputs, batch_outputs, reduction='mean')

        if lr == 1e4:
            if step == 0:
                init_loss = loss.item()
            if step == 5000:
                if loss.item() >= init_loss/4:
                    print()
                    print("Supervised Learning Failed. Retrying...")
                    return False 

        if step % log_interval == 0:
            pbar.set_description("Step: {}, LR: {}, Training MSE Loss: {:.2f}, ".format(step, lr, loss.item()))

        qmodel.opt.zero_grad()
        loss.backward()
        qmodel.opt.step()

    print("Final Training MSE Loss: {:.2f}".format(loss.item()))
    print("Supervised Learning Worked")
    return True

def q_model_bootstrapping_ggggg(qmodel, policy, system, gamma, episode_length, max_episodes, log_interval=10, batch_size=100, training_steps=10000, lr=1e-4):
    """
        first do MC evaluation of policy
        bootstrapping
    """
    n = system.n
    value_model = ValueNetwork(n, 2*n*n, 2)
    v_optimizer = optim.Adam(value_model.parameters(), lr=lr)
    while not value_net_bootstrapping(value_model, policy, v_optimizer, system, gamma, episode_length, max_episodes):
        continue

    print("Train Q Network with Value Network")

    for first_action in range(n):
        print("Action: {}".format(first_action))
        flag = False
        while not flag:
            replay_bank_states = []
            replay_bank_returns = []
            for i in range(max_episodes*episode_length//100):
                state = system.reset('Random')
                replay_bank_states.append(state)   
                next_state, _, cost, _ = system.step(first_action)                     
                qval = cost + gamma * value_model(next_state).detach().item()
                replay_bank_returns.append([qval])   

            # learning V^pi 
            print("Learning Value Network...")
            replay_bank_states = np.array(replay_bank_states)
            replay_bank_returns = np.array(replay_bank_returns)
            loss_history = [0] * 10
            init_loss = 0
            pbar = tqdm(range(training_steps))
            for step in pbar:
                idx = np.random.choice(np.arange(len(replay_bank_states)), size=batch_size)
                batch_inputs = torch.tensor(replay_bank_states[idx])
                batch_outputs = torch.tensor(replay_bank_returns[idx])
                model_outputs = qmodel.evaluate(batch_inputs, first_action)
                loss = F.smooth_l1_loss(model_outputs, batch_outputs, reduction='mean')

                if lr == 1e4:
                    if step == 0:
                        init_loss = loss.item()
                    if step == 5000:
                        if loss.item() >= init_loss/4:
                            print()
                            print("Supervised Learning Failed. Retrying...")
                            flag = False
                            break 

                if step % log_interval == 0:
                    pbar.set_description("Step: {}, LR: {}, Training MSE Loss: {:.2f}, ".format(step, lr, loss.item()))
                qmodel.optimizers[first_action].zero_grad()
                loss.backward()
                qmodel.optimizers[first_action].step()
            print("Final Training MSE Loss: {:.2f}".format(loss.item()))
            print("Supervised Learning Worked")
            flag = True

    return True    

def greedy_policy_pretrain(policy_model, optimizer, system, greedy_policy, episode_length, max_episodes, log_interval=10, batch_size=100, training_steps=5000, lr=1e-4):
    """ 
        train policy to greedy policy
        bootstrapping

    """
    print("Learning Policy Network (Mimicking Greedy)...")
    training_states = []
    training_action_dist = []
    for step in range(max_episodes):
        system.reset('Random')
        for t in range(episode_length):
            dist = greedy_policy(system.P)
            training_states.append(normalize_input(system.P))
            training_action_dist.append(dist)
            # print(dist)
            action = np.argmax(dist)
            system.step(action)

    training_states = np.array(training_states)
    training_action_dist = np.array(training_action_dist)
    pbar = tqdm(range(training_steps))
    for step in pbar:
        idx = np.random.randint(0, high=len(training_states), size=batch_size)
        batch_inputs = torch.tensor(training_states[idx])
        batch_outputs = torch.tensor(training_action_dist[idx])
        model_outputs = policy_model(batch_inputs)
        # print(model_outputs)
        loss = F.kl_div(model_outputs, batch_outputs, reduction='batchmean')

        if step == 0:
            init_loss = loss.item()
        if step == 5000:
            if loss.item() >= init_loss/5:
                print()
                print("Supervised Learning Failed. Retrying...")
                return False 

        if step % log_interval == 0:
            pbar.set_description("Step: {}, LR: {}, Training KL Loss: {:.2f}".format(step, lr, loss.item()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print("Final Training KL Loss: {:.2f}".format(loss.item()))
    print("Supervised Learning Worked for Greedy")
    return True

def value_net_bootstrapping(value_model, policy, optimizer, system, gamma, episode_length, max_episodes, policy_model = None, log_interval=10, batch_size=100, training_steps=10000, lr=1e-4):
    """
        first do MC evaluation of policy
        bootstrapping
    """
    print("Generating trajectories for Monte Carlo Method...")
    replay_bank_states = []
    replay_bank_returns = []
    for step in tqdm(range(max_episodes)):
        actions, costs, log_probs, values, total_cost, states = policy_rollout(system, policy, episode_length, model=policy_model)
        returns = discounted_returns(costs, gamma)
        for i in range(episode_length//100):
            replay_bank_states.append(states[i])
            replay_bank_returns.append([returns[i]])

    # learning V^pi 
    print("Learning Value Network...")
    replay_bank_states = np.array(replay_bank_states)
    replay_bank_returns = np.array(replay_bank_returns)
    loss_history = [0] * 10
    init_loss = 0
    pbar = tqdm(range(training_steps))
    for step in pbar:
        idx = np.random.choice(np.arange(len(replay_bank_states)), size=batch_size)
        batch_inputs = torch.tensor(replay_bank_states[idx])
        batch_outputs = torch.tensor(replay_bank_returns[idx])

        model_outputs = value_model(batch_inputs)
        loss = F.smooth_l1_loss(model_outputs, batch_outputs, reduction='mean')

        if lr == 1e4:
            if step == 0:
                init_loss = loss.item()
            if step == 5000:
                if loss.item() >= init_loss/4:
                    print()
                    print("Supervised Learning Failed. Retrying...")
                    return False 

        if step % log_interval == 0:
            pbar.set_description("Step: {}, LR: {}, Training MSE Loss: {:.2f}, ".format(step, lr, loss.item()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Final Training MSE Loss: {:.2f}".format(loss.item()))
    print("Supervised Learning Worked")
    return True

def actor_critic(model, system, gamma, episode_length, max_episodes, gae_lam=0.9, log_interval=10, step_save=50):
    lr = 1e-3
    iir = 0.1
    print("Starting Actor Critic RL")
    model.update_value_optimizer(lr)
    model.update_policy_optimizer(lr)

    history_costs = []
    pbar = tqdm(range(max_episodes))

    episode_costs = np.zeros(max_episodes)

    for step in pbar:
        system.reset('Random')
        
        costs = torch.zeros(episode_length)
        values = torch.zeros(episode_length+1)
        deltas = torch.zeros(episode_length)
        log_dists = torch.zeros(episode_length)
        actions = torch.zeros(episode_length, dtype=torch.int)

        for t in range(episode_length):
            log_dist = model.policy_model(normalize_input(system.P))
            values[t] = model.value_model(system.P)
            actions[t] = Categorical(torch.exp(log_dist)).sample().detach()
            log_dists[t] = log_dist[actions[t]]

            _, _, costs[t], _ = system.step(actions[t])


        episode_costs[step] = torch.mean(costs[100:]).item()

        A_hat = torch.zeros(episode_length+1)
        for t in reversed(range(episode_length)):
            deltas[t] = costs[t] + gamma * values[t+1].detach() - values[t]
            A_hat[t] = deltas[t] + gamma * gae_lam * A_hat[t+1].detach()

        value_loss = 0.5 * torch.mean(A_hat ** 2) 
        policy_loss = torch.mean( A_hat[:-1].detach() * log_dists)
        model.value_optimizer.zero_grad()
        value_loss.backward()
        model.value_optimizer.step()

        model.policy_optimizer.zero_grad()
        policy_loss.backward()
        model.policy_optimizer.step()
                
        if step == 0:
            running_cost = episode_costs[step]
        else:
            running_cost = iir * episode_costs[step] + (1-iir) * running_cost

        history_costs.append(running_cost)
        if step > 20 and lr > 1e-5:
            if np.abs(np.mean(history_costs[:-1]) - np.mean(history_costs[1:])) < lr:
                lr /= 10
                model.update_value_optimizer(lr)
                model.update_policy_optimizer(lr)
            history_costs.pop(0)

        if step % log_interval == 0:
            pbar.set_description('Episode {}, logLR: {:.2f}, Last Policy Cost: {:.2f}, Average Cost: {:.2f}'.format(step, np.log10(lr), episode_costs[step], running_cost))

        # if step % step_save == 0:
            # Saves model checkpoint
            # save_pv_checkpoint(model, step, save_dir="./"+system.name+"_"+model.name, log=False)

    save_pv_checkpoint(model, step, save_dir="./"+system.name+"_"+model.name)

    return episode_costs















