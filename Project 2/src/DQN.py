import numpy as np
from tensorflow_core.python.keras.activations import relu, linear
from tensorflow_core.python.keras.losses import mean_squared_error
from tensorflow_core.python.keras.models import Sequential
from tensorflow_core.python.keras.layers.core import Dense
from tensorflow_core.python.keras.optimizer_v2.adam import Adam
import matplotlib.pyplot as plt
from collections import deque
import sys
import random


class DQN:
    def __init__(self, env, alpha=0.0001, gamma=0.9, epsilon=1, max_episodes=1000,
                 rewards_vs_episodes_plot="Rewards_vs_episodes_1.png",
                 analysis_run=False, epsilon_decay=0.995):
        np.random.seed(7)
        self.epsilon = epsilon
        self.alpha = alpha
        self.time_step = 0
        self.env = env
        self.analysis_run = analysis_run
        self.rewards_vs_episodes_plot = rewards_vs_episodes_plot
        self.training_model = NNModel(self.alpha, self.env.observation_space.shape[0], self.env.action_space.n)
        self.prediction_model = NNModel(self.alpha, self.env.observation_space.shape[0], self.env.action_space.n)
        # self.nn_model_2 = NNModel(self.alpha, self.env.observation_space.shape[0], self.env.action_space.n)
        self.max_episodes = max_episodes
        self.memory = deque(maxlen = 10000)
        self.decay_epsilon = epsilon_decay
        self.batch_size = 64
        self.gamma = gamma

    def train(self):
        rewards_list = []
        episode_of_convergence = self.max_episodes
        for episode in range(self.max_episodes):
            print("\nEpisode:", episode)
            rewards = 0
            state = self.env.reset()
            step = 0
            print("epsilon", self.epsilon)
            while True:
                if self.time_step % 100 == 0:
                    self.prediction_model.set_weights(self.training_model.get_weights())
                self.time_step += 1
                sys.stdout.write(str(step) + " ")
                step += 1
                action = self.explore_or_exploit(state)
                state_prime, reward, done, info = self.env.step(action)
                rewards += reward
                self.memory.append((state, action, state_prime, reward, done))
                if done:
                    break
                state = state_prime
                self.train_nn_model()
            rewards_list.append(rewards)
            if self.epsilon > 0.01:
                self.epsilon = self.epsilon * self.decay_epsilon
            if not self.analysis_run:
                self.plot_rewards(rewards_list)
            if len(rewards_list) >= 100:
                print("\nRewards Cumulative:", np.sum(rewards_list[-100:]))
            if np.mean(rewards_list[-100:]) >= 200:
                if self.analysis_run:
                    episode_of_convergence = episode
                else:
                    print("Done")
                    break
        if not self.analysis_run:
            self.training_model.save()
        return rewards_list, episode_of_convergence

    def explore_or_exploit(self, state):
        if np.random.random() < self.epsilon:
            return random.randrange(self.env.action_space.n)
        else:
            return np.argmax(self.prediction_model.predict(state))

    def train_nn_model(self):
        if len(self.memory) > 1000:
            mem_array = np.asarray(self.memory)
            indices = np.random.choice(mem_array.shape[0], self.batch_size)
            batch = mem_array[indices]
            states = np.array([entry[0] for entry in batch])
            states_prime = np.array([entry[2] for entry in batch])
            actions = np.array([entry[1] for entry in batch])
            rewards = np.array([entry[3] for entry in batch])
            done_values = np.array([0 if entry[4] else 1 for entry in batch])
            new_vals = rewards + done_values * self.get_future_rewards(states_prime)

            # Model output
            model_output = self.training_model.predict_multiple(states)
            model_output[[np.array(range(self.batch_size))], [actions]] = new_vals
            self.training_model.fit(states, model_output)

    def get_future_rewards(self, states_prime):
        return np.amax(self.gamma * self.prediction_model.predict_multiple(states_prime), axis=1)

    def plot_rewards(self, rewards_list):
        plt.xlabel("episodes")
        plt.ylabel("rewards")
        plt.plot(rewards_list)
        plt.savefig(self.rewards_vs_episodes_plot)
        plt.clf()

    def load(self):
        self.training_model.load("dqn1.h5")

    def run(self):
        # self.env.seed(121)
        done = False
        state = self.env.reset()
        step = 0
        rewards = 0
        while not done:
            step = step +1
            self.env.render()
            action = np.argmax(self.training_model.predict(state))
            state_prime, reward, done, info = self.env.step(action)
            state = state_prime
            rewards += reward
        return rewards

class NNModel:

    def __init__(self, alpha, input_dim, output_dim):
        self.state_space_count = input_dim
        self.action_space_count = output_dim
        self.model = Sequential()
        self.model.add(Dense(64, input_dim = input_dim, activation = relu))
        self.model.add(Dense(64, activation = relu))
        self.model.add(Dense(output_dim, activation = linear))
        self.model.compile(loss = mean_squared_error, optimizer = Adam(lr = alpha))

    def fit(self, data, output):
        self.model.fit(data, output, epochs=1, verbose=0)

    def save(self):
        self.model.save_weights("dqn1.h5")

    def predict(self, state):
        return self.model.predict(np.reshape(state, [1, self.state_space_count]))

    def predict_multiple(self, states):
        return self.model.predict_on_batch(states).numpy()

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, w):
        return self.model.set_weights(w)

    def load(self, filename):
        self.model.load_weights(filename)
        self.model.compile(loss = mean_squared_error, optimizer = Adam(lr = 0.0001))