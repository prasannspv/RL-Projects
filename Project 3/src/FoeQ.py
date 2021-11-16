import numpy as np
from game.Soccer import Soccer
from cvxopt.modeling import matrix
from cvxopt import solvers
from cvxopt.solvers import options, lp
import matplotlib.pyplot as plt


def select_action(pi):
    # Different From previous approaches. Actions sampled with probability pi
    return np.random.choice(list(range(5)), p = pi)


def solve_foe_q(q_s_a):
    options['glpk'] = {
        'msg_lev': 'GLP_MSG_OFF',
        'tm_lim': 1000
    }

    temp = [0., 1., 1., 1., 1., 1.]
    c = matrix([-1., 0., 0., 0., 0., 0.])
    A = matrix(
        np.vstack(
            (np.vstack(
                (np.hstack((np.ones((5, 1)), matrix(q_s_a).trans())),
                 -np.hstack((np.zeros((5, 1)), np.eye(5))))),
             temp,
             -np.asarray(temp)
            )
        )
    )
    b = matrix([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., -1.])
    sol = solvers.lp(c, A, b, solver='glpk')
    objective_ = sol['primal objective']
    return objective_ or 0


def plot_diff(diff_values, iterations):
    plt.xlabel("Iteration")
    plt.ylabel("Diff_Q")
    plt.ylim(0, 0.5)
    plt.plot(iterations, diff_values)
    plt.savefig("FoeQ.png")
    plt.clf()


def foe_q(max_step=1000000, alpha=1, epsilon=1, gamma=0.9):
    step = 0
    alpha_decay = 0.999995

    env = Soccer((0, 2), (0, 1))

    # Different Q Table Representation for ease of computation. Each element is an array of 5 X 5
    Q_A = {}
    diff_values = []
    iteration = []

    while step < max_step:
        state_A = env.encode_state("A")
        state_B = env.encode_state("B")
        state = state_A + state_B

        actions = env.sample(), env.sample()

        env.act(actions)
        internal_state = state_A + str(actions[0]) + str(actions[1])

        reward_A = env.get_reward("A")
        q_s_a_A = Q_A.get(state, np.ones((5, 5)))

        value_A = solve_foe_q(q_s_a_A)

        new_q_s_a_A = (1 - alpha) * q_s_a_A[actions[0]][actions[1]] \
                      + alpha * (((1 - gamma) * reward_A) + (gamma * value_A))

        if internal_state == "T114":
            diff_values.append(abs(new_q_s_a_A - q_s_a_A[actions[0]][actions[1]]))
            iteration.append(step)

        if step % 1000 == 0:
            print("Done with step ", step)

        if step % 10000 == 0:
            plot_diff(diff_values, iteration)

        step += 1
        alpha = max(0.001, alpha * alpha_decay)

        q_s_a_A[actions[0]][actions[1]] = new_q_s_a_A
        Q_A[state] = q_s_a_A

foe_q()