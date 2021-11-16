from src.game.Soccer import Soccer
import matplotlib.pyplot as plt
import numpy as np


def get_state_value(Q, encoded_state):
    return max([Q.get(encoded_state + str(i) + str(j), 0) for i in range(0, 5) for j in range(0, 5)])


def plot_diff(diff_values, iterations):
    plt.xlabel("Iteration")
    plt.ylabel("Diff_Q")
    plt.ylim(0, 0.5)
    plt.plot(iterations, diff_values)
    plt.savefig("FriendQ.png")
    plt.clf()


def choose_action_from_Q(Q, encoded_state):
    index = np.argmax([Q.get(encoded_state + str(i) + str(j), 0) for i in range(0, 5) for j in range(0, 5)])
    return int(index / 5)


def friend_Q(max_steps=1000000, alpha=1, gamma=0.9):
    step = 0
    env = Soccer((0, 2), (0, 1))
    Q_A = {}
    Q_B = {}
    diff_values = []
    alpha_decay = 0.9995
    iteration = []

    while step < max_steps:
        encoded_state_A = env.encode_state("A")
        encoded_state_B = env.encode_state("B")
        action = env.sample(), env.sample()
        env.act(action)

        s_a_pair_A = encoded_state_A + str(action[0]) + str(action[1])
        s_a_pair_B = encoded_state_B + str(action[0]) + str(action[1])

        q_s_a_A = Q_A.get(s_a_pair_A, 0)
        q_s_a_B = Q_B.get(s_a_pair_B, 0)

        v_s_A = get_state_value(Q_A, env.encode_state("A"))
        v_s_B = get_state_value(Q_B, env.encode_state("B"))

        reward_A = env.get_reward("A")
        reward_B = env.get_reward("B")

        if reward_A != 0:
            env.reset()

        new_q_s_a_A = (1 - alpha) * q_s_a_A + alpha * (((1 - gamma) * reward_A) + (gamma * v_s_A))
        new_q_s_a_B = (1 - alpha) * q_s_a_B + alpha * (((1 - gamma) * reward_B) + (gamma * v_s_B))

        diff_value = abs(q_s_a_A - new_q_s_a_A)

        # Only the same state-action pair will be plotted
        if s_a_pair_A == "T343":
            diff_values.append(diff_value)
            iteration.append(step)

        Q_A[s_a_pair_A] = new_q_s_a_A
        Q_B[s_a_pair_B] = new_q_s_a_B

        step += 1
        if step % 1000 == 0:
            print("Done with step ", step)

        if step % 100000 == 0:
            plot_diff(diff_values, iteration)

        alpha = max(0.001, alpha * alpha_decay)

friend_Q()
