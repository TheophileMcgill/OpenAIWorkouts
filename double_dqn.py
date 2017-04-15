import numpy as np
import gym, random, math
from gym.wrappers import Monitor

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


class DQN:
    def __init__(self, n_dim_states, n_actions):
        self.n_dim_states = n_dim_states
        self.n_actions = n_actions
        self.model = self._createModel()

    def _createModel(self):
        model = Sequential()
        model.add(Dense(50, activation='relu', input_dim=self.n_dim_states))
        model.add(Dense(self.n_actions, activation='linear'))
        adam = Adam(0.001)
        model.compile(loss='mse', optimizer='adam')
        return model

    def fit(self, X, y, batch_size=64, nb_epoch=1, verbose=0):
        self.model.fit(X, y, batch_size=batch_size, nb_epoch=nb_epoch, verbose=verbose)

    def predict(self, state):
        return self.model.predict(state)

    def predict_one(self, state):
        return self.predict(state.reshape(1, self.n_dim_states)).flatten()

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def get_weights(self):
        return self.model.get_weights()


class Agent:
    def __init__(self, n_dim_states, n_actions, gamma=0.99, max_epsilon=1, min_epsilon=0.01,
     eps_decay=0.001, batch_size=64, mem_capacity=30000):
        self.n_dim_states = n_dim_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size

        # Keep target network separated for stability
        self.current_DQN = DQN(n_dim_states, n_actions)
        self.target_DQN = DQN(n_dim_states, n_actions)

        # List of past samples (state, action, reward, next_state)
        self.memory = []
        self.mem_capacity = mem_capacity

        # Important : epsilon decay, otherwise method has high variance
        self.n_steps = 0                # Keep track of number of steps to decrease epsilon
        self.epsilon = max_epsilon      # Start by exploring all the time
        self.min_epsilon = min_epsilon  # End up exploiting most of the time
        self.eps_decay = eps_decay      # Speed of decay for epsilon
        
    def epsilon_greedy_policy(self, state):
        # Pick random action with probability epsilon
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions-1)
        # Pick greedy action with probability (1 - epsilon)
        else:
            return np.argmax(self.current_DQN.predict_one(state))

    def observe(self, sample):
        # Add sample to memory, and delete one sample if capacity exceeded
        self.memory.append(sample)
        if (len(self.memory) > self.mem_capacity):
            self.memory.pop(0)
        # Decrease epsilon to favor exploitation over exploration over time
        self.n_steps += 1
        self.epsilon = self.min_epsilon + (
            1 - self.min_epsilon) * math.exp(-self.eps_decay * self.n_steps)

    def experience_replay(self):
        # Sample a batch from memory uniformly at random
        batch_size = min(self.batch_size, len(self.memory))
        batch = random.sample(self.memory, batch_size)

        # Predict q_values in batches for efficiency
        none_state = np.zeros(self.n_dim_states) # Used in place of None for next_state 
        states = np.array([sample[0] for sample in batch])
        next_states = np.array([(none_state if sample[3] is None else sample[3]) for sample in batch])
        q_values = self.current_DQN.predict(states)
        # Predict q_values_next using target network
        q_values_next_target = self.target_DQN.predict(next_states)
        q_values_next_current = self.current_DQN.predict(next_states)

        # Fill in our training batch
        X = np.zeros((batch_size, self.n_dim_states))
        y = np.zeros((batch_size, self.n_actions))
        for i in range(batch_size):
            state, action, reward, next_state = batch[i]
            # Important : target is the q_value itself for all actions except the one actually taken 
            target = q_values[i]
            if next_state is None:
                target[action] = reward
            else:
                # DQN with frozen target : greedy policy + evaluation by target network
                #target[action] = reward + self.gamma * np.amax(q_values_next_target[i])
                # Double DQN : greedy policy by current network + evaluation by target network
                next_action = np.argmax(q_values_next_current[i])
                target[action] = reward + self.gamma * q_values_next_target[i, next_action]
            X[i] = state
            y[i] = target

        # Fit network with training batch
        self.current_DQN.fit(X, y)

    def update_target_network(self):
        # Update target network weights to reflect current weights
        self.target_DQN.set_weights(self.current_DQN.get_weights())


class Environment:
    def __init__(self, environment):
        #self.env = Monitor(gym.make(environment), 'CartPole-v1-experiment')
        self.env = gym.make(environment)
        self.n_episodes = 0

    def run_episode(self, agent):
        self.n_episodes += 1
        # Every episode, update target network to reflect current network
        agent.update_target_network()
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
                break
        print("Episode {}, total reward: {}".format(self.n_episodes, total_reward))


if __name__ == "__main__":
    env = Environment('CartPole-v1')

    n_dim_states = env.env.observation_space.shape[0]
    n_actions = env.env.action_space.n

    agent = Agent(n_dim_states, n_actions)

    while(True):
        env.run_episode(agent)
