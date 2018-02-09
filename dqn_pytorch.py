import numpy as np
import gym, random, math
from gym.wrappers import Monitor

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable


class DQN(nn.Module):

    def __init__(self, n_dim_states, n_actions):
        super(DQN, self).__init__()
        self.n_dim_states = n_dim_states
        self.n_actions = n_actions

        self.lin1 = nn.Linear(self.n_dim_states, 50)
        self.lin2 = nn.Linear(50, self.n_actions)

        self.optimizer = optim.Adam(self.parameters())

    def forward(self, X):
        X = F.relu(self.lin1(X))
        return self.lin2(X)

    def fit(self, X, y, nb_epoch=1):
        var_X, var_y = Variable(torch.Tensor(X)), Variable(torch.Tensor(y))

        """
        lr = 0.01
        for i in xrange(nb_epoch):
            self.zero_grad()
            loss = criterion(self(var_X), var_y)
            loss.backward()
            for f in self.parameters():
                f.data.sub_(f.grad.data * lr)
        """

        for i in xrange(nb_epoch):
            self.optimizer.zero_grad()
            loss = nn.MSELoss()(self(var_X), var_y)
            loss.backward()
            self.optimizer.step()

    def predict(self, state):
        var_state = Variable(torch.Tensor(state))
        return self(var_state).data.numpy()

    def predict_one(self, state):
        var_state = Variable(torch.Tensor(state.reshape(1, self.n_dim_states)))
        return self(var_state).view(-1).data.numpy()



class Agent:

    def __init__(self, n_dim_states, n_actions, gamma=0.9, min_epsilon=0.05,
     eps_decay=0.001, batch_size=64, mem_capacity=10000):
        self.n_dim_states = n_dim_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.DQN = DQN(n_dim_states, n_actions)
      
        # List of past samples (state, action, reward, next_state)
        self.memory = []
        self.mem_capacity = mem_capacity

        # Important : epsilon decay, otherwise method has high variance
        self.n_steps = 0                # Keep track of number of steps to decrease epsilon
        self.epsilon = 1                # Start by exploring all the time
        self.min_epsilon = min_epsilon  # End up exploiting most of the time
        self.eps_decay = eps_decay      # Speed of decay for epsilon
        
    def epsilon_greedy_policy(self, state):
        # Pick random action with probability epsilon
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions-1)
        # Pick greedy action with probability (1 - epsilon)
        else:
            return np.argmax(self.DQN.predict_one(state))

    def greedy_policy(self, state):
        return np.argmax(self.DQN.predict_one(state))

    def observe(self, sample):
        # Add sample to memory, and delete one sample if capacity exceeded
        self.memory.append(sample)
        if (len(self.memory) > self.mem_capacity):
            self.memory.pop(0)
        # Decrease epsilon to favor exploitation over exploration over time
        self.n_steps += 1
        # TODO : find more info about epsilon decay schemes
        self.epsilon = self.min_epsilon + (
            1 - self.min_epsilon) * math.exp(-self.eps_decay * self.n_steps)

    def experience_replay(self):
        # Sample a batch from memory uniformly at random
        batch_size = min(self.batch_size, len(self.memory))
        batch = random.sample(self.memory, batch_size)

        # Prepare batches of states to predict q-values
        none_state = np.zeros(self.n_dim_states) # Used in place of None for next_state 
        states = np.array([sample[0] for sample in batch])
        next_states = np.array([(none_state if sample[3] is None else sample[3]) for sample in batch])

        # Predict q-values in batches for efficiency
        q_values = self.DQN.predict(states)
        q_values_next = self.DQN.predict(next_states)

        # Fill in our training batch for DQN
        # Important : target is the q_value itself for all actions except the one actually taken
        X = states
        y = q_values
        for i in range(batch_size):
            state, action, reward, next_state = batch[i]
            target = reward if next_state is None else reward + self.gamma * np.amax(q_values_next[i])
            y[i, action] = target

        # Fit network with training batch
        self.DQN.fit(X, y)


class Environment:
    def __init__(self, environment):
        #self.env = Monitor(gym.make(environment), 'CartPole-v1-experiment', force=True)
        self.env = gym.make(environment)
        self.n_episodes = 0
        self.n_successes_in_a_row = 0

    def run_episode_training(self, agent):
        self.n_episodes += 1
        state = self.env.reset()
        total_reward = 0 
        while True:            
            self.env.render()
            action = agent.epsilon_greedy_policy(state)
            next_state, reward, done, info = self.env.step(action)
            # Important to treat next state when done differently 
            if done:
                next_state = None
            agent.observe((state, action, reward, next_state))
            agent.experience_replay()            
            state = next_state
            total_reward += reward
            if done:
                if total_reward == 500.0:
                    self.n_successes_in_a_row += 1
                else:
                    self.n_successes_in_a_row = 0
                break
        print("Episode {} (training), total reward: {}".format(self.n_episodes, total_reward))

    def run_episode(self, agent):
        self.n_episodes += 1
        state = self.env.reset()
        total_reward = 0
        while True:            
            self.env.render()
            action = agent.greedy_policy(state)
            next_state, reward, done, info = self.env.step(action)   
            state = next_state
            total_reward += reward
            if done:
                break
        print("Episode {}, total reward: {}".format(self.n_episodes, total_reward))



if __name__ == "__main__":
    env = Environment('CartPole-v1')

    n_dim_states = env.env.observation_space.shape[0]
    n_actions = env.env.action_space.n

    agent = Agent(n_dim_states, n_actions)

    while(env.n_successes_in_a_row < 5):
        env.run_episode_training(agent)

    while True:
        env.run_episode(agent)