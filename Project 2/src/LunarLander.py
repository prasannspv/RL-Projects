import gym
from DQN import DQN
import json
import matplotlib.pyplot as plt
import numpy as np

# Initialize the environment
env = gym.make("LunarLander-v2")

# Create a DQN instance and train
# DQN(env, max_episodes = 1000).train()

# rewards = []
# dqn = DQN(env)
# dqn.load()
# for i in range(100):
#     print(i)
#     rewards.append(dqn.run())
# plt.plot(rewards)
# plt.savefig("100runs.png")


def alpha_analysis():
    rewards_dict = {}
    episodes_of_convergence = {}
    for alpha in [0.0001, 0.0005, 0.001]:
        rewards_dict[alpha], episodes_of_convergence[alpha] = DQN(env, analysis_run=True, alpha=alpha, max_episodes = 500).train()
        with open("alpha_rewards_episodes.json", "w+") as file:
            json.dump(rewards_dict, file)
        with open("alpha_convergence.json", "w+") as file:
            json.dump(episodes_of_convergence, file)


# alpha_analysis()


def gamma_analysis():
    rewards_dict = {}
    episodes_of_convergence = {}
    for gamma in [0, 0.2, 0.4, 0.6, 0.8, 1]:
        rewards_dict[gamma], episodes_of_convergence[gamma] = DQN(env, analysis_run = True, gamma = gamma,
                                                                  max_episodes = 500).train()
        with open("gamma_rewards_episodes.json", "w+") as file:
            json.dump(rewards_dict, file)
        with open("gamma_convergence.json", "w+") as file:
            json.dump(episodes_of_convergence, file)


def _plot_analysis_aggregated(rewards_json, figname):
    with open(rewards_json) as file:
        rewards_json = json.load(file)
        aggregated_rewards_json = {}
        for val, rewards_list in rewards_json.items():
            aggregated_rewards = [np.mean(rewards_list[i:i+100]) for i in range(400)]
            aggregated_rewards_json[val] = aggregated_rewards
        for key, aggregated_rewards in aggregated_rewards_json.items():
            # val, rew = zip(*aggregated_rewards)  # Unpack
            plt.plot(list(range(101, 501)), aggregated_rewards, label = key)
        plt.legend()
        plt.xlabel("Episodes")
        plt.ylabel("Mean of last 100 rewards")
        plt.savefig(figname)


def _plot_analysis(rewards_json, figname):
    with open(rewards_json) as file:
        rewards_json = json.load(file)
        for key, rewards in rewards_json.items():
            plt.plot(rewards, label = key)
        plt.legend()
        plt.xlabel("Episodes")
        plt.ylabel("Rewards")
        plt.savefig(figname)


def epsilon_decay_analysis():
    rewards_dict = {}
    episodes_of_convergence = {}
    for epsilon_decay in [0.9, 0.95, 0.99, 0.995, 1]:
        rewards_dict[epsilon_decay], episodes_of_convergence[epsilon_decay] = \
                DQN(env, analysis_run = True, epsilon_decay=epsilon_decay, max_episodes = 500).train()
        with open("epsilon_decay_episodes.json", "w+") as file:
            json.dump(rewards_dict, file)
        with open("epsilon_decay_convergence.json", "w+") as file:
            json.dump(episodes_of_convergence, file)


def _plot_box_plot(rewards_json, figname):
    with open(rewards_json) as file:
        rewards_json = json.load(file)
        rewards_list = []
        map = {
            "0": [16, 32],
            "1": [32, 64],
            "2": [64, 128],
            "3": [32, 32],
            "4": [64, 64]
        }
        sizes =[ ]
        for key, values in rewards_json.items():
            sizes.append(map[key])
            # rewards_list.append(values)
            rewards_list.append([np.mean(values[i: i+100]) for i in range(400)])
        plt.boxplot(rewards_list)
        plt.xticks([1, 2, 3, 4, 5], list(map.values()))
        plt.xlabel("# Nodes")
        plt.ylabel("Rewards")
        plt.savefig(figname)

# epsilon_decay_analysis()
# gamma_analysis()
# _plot_analysis_aggregated("gamma_rewards_episodes.json", "gamma_rewards_episodes.png")
# _plot_analysis_aggregated("epsilon_decay_episodes.json", "epsilon_decay_episodes.png")
# _plot_box_plot("nodes.json", "nodes.png")


np.random.seed(121)
env.seed(121)
dqn = DQN(env)
dqn.load()
dqn.run()