import numpy as np
import gym, random, math
from gym.wrappers import Monitor

from keras.models import Sequential
from keras.layers import Input, Dense
from keras.optimizers import Adam
import tensorflow as tf


class Actor:
    def _createGraph(self, n_dim_states, n_actions):
        self.state = tf.placeholder("float", [None, n_dim_states])
        self.action = tf.placeholder("float", [None, n_actions]) # One hot vector (action taken)
        self.advantage = tf.placeholder("float", [None, 1])
        
        # Compute probabilities of taking actions
        action_probas = Dense(n_actions, activation='softmax')(self.state)

        # The gradient computed w.r.t this loss corresponds to the policy gradient theorem

        # The two lines below is the same as computing the cross-entropy between our predicted actions
        # distribution and the true deterministic distribution of actions that have been taken
        # Could be done with a cross-entropy builtin function, and could even be improved by using 
        # a builtin cross-entropy-with-logits function and removing the softmax activation
        # e.g :
        # logits =  Dense(n_actions)(self.state)
        # negative_likelihoods = tf.nn.softmax_cross_entropy_with_logits(labels=self.action, logits = logits)
        action_taken_proba = tf.reduce_sum(action_probas * self.action, reduction_indices=1)
        negative_likelihoods = -tf.log(action_taken_proba)

        weighted_negative_likelihoods = negative_likelihoods * self.advantage
        loss = tf.reduce_mean(weighted_negative_likelihoods)
        train_step = tf.train.AdamOptimizer(0.01).minimize(loss)

        return action_probas, train_step

    def __init__(self, n_dim_states, n_actions):
        self.policy, self.train_step = self._createGraph(n_dim_states, n_actions)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def partial_fit(self, state, action, advantage):
        self.sess.run(
            self.train_step, 
            feed_dict={self.state: state, self.action: action, self.advantage: advantage})

    def predict_one(self, state):
        policy = self.sess.run(
            self.policy,
            feed_dict={self.state: np.expand_dims(state, axis=0)})
        return policy.flatten()


class Critic:
    def __init__(self, n_dim_states):
        self.model = self._createModel(n_dim_states)

    def _createModel(self, n_dim_states):
        model = Sequential()
        model.add(Dense(50, activation='relu', input_dim=n_dim_states))
        model.add(Dense(1, activation='linear'))
        # Loss is mean squared error between predicted state value and td_target
        adam = Adam(0.01)
        model.compile(loss='mse', optimizer=adam)
        return model

    # For debugging
    def eval_mse(self, state, target):
        return np.mean((self.model.predict(state) - target)**2)

    def partial_fit(self, state, target):
        self.model.fit(state, target, batch_size=64, nb_epoch=1, verbose=0)

    def predict(self, state):
        return self.model.predict(state)


class Agent:
    def __init__(self, n_dim_states, n_actions, gamma=0.9, epsilon=0.1, batch_size=64, 
        mem_capacity=30000):
        self.n_dim_states = n_dim_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon

        self.actor = Actor(n_dim_states, n_actions)
        self.critic = Critic(n_dim_states)
        
        # List of past samples (state, action, reward, next_state)
        self.memory = []
        self.mem_capacity = mem_capacity
        
    def epsilon_greedy_policy(self, state):
        # Pick random action with probability epsilon
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions-1)
        else:
            action_probas = self.actor.predict_one(state)
            action = 0 if random.random() < action_probas[0] else 1
            return action

    def observe(self, sample):
        # Add sample to memory, and delete one sample if capacity exceeded
        self.memory.append(sample)
        if (len(self.memory) > self.mem_capacity):
            self.memory.pop(0)

    def experience_replay(self):
        # Sample a batch from memory uniformly at random 
        batch_size = min(self.batch_size, len(self.memory))
        batch = random.sample(self.memory, batch_size)

        # Samples take the form (state, action, reward, next_state)
        none_state = np.zeros(self.n_dim_states) # Used in place of None for next_state 
        states = np.array([sample[0] for sample in batch])
        actions = np.array([sample[1] for sample in batch])
        rewards = np.array([sample[2] for sample in batch])
        next_states = np.array([(none_state if sample[3] is None else sample[3]) for sample in batch])
        
        # Predict state values in batches for efficiency
        state_values = self.critic.predict(states)
        state_values_next = self.critic.predict(next_states)
        for i in range(batch_size):
            if batch[i][3] is None:
                state_values_next[i] = 0

        # Compute td_errors
        td_targets = rewards + self.gamma * state_values_next
        td_errors = td_targets - state_values

        # Train actor network
        self.actor.partial_fit(states, actions, td_errors)

        # Train critic network
        self.critic.partial_fit(states, td_targets)


class Environment:
    def __init__(self, environment):
        #self.env = Monitor(gym.make(environment), 'CartPole-v1-actorcritic')
        self.env = gym.make(environment)
        self.n_episodes = 0

    def run_episode(self, agent):
        self.n_episodes += 1
        state = self.env.reset()
        total_reward = 0 
        while True:            
            self.env.render()
            action = agent.epsilon_greedy_policy(state)
            next_state, reward, done, info = self.env.step(action)
            if done:
                next_state = None
            action_one_hot = np.zeros(self.env.action_space.n)
            action_one_hot[action] = 1
            agent.observe((state, action_one_hot, [reward], next_state))
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


