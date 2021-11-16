import numpy as np
from game.Soccer import Soccer
from cvxopt.modeling import matrix
from cvxopt.solvers import options, lp
import matplotlib.pyplot as plt


def select_action(pi):
    # Different From previous approaches. Actions sampled with probability pi
    return np.random.choice(list(range(5)), p = pi)


def solve_ce_q(q_s_a, q_s_b):
    options['glpk'] = {
        'msg_lev': 'GLP_MSG_OFF',
        'tm_lim': 10000
    }

    temp = np.append(np.array([0]), np.ones(25))
    c = matrix(-np.hstack(([1.], np.add(q_s_a, q_s_b).flatten())))
    A = np.zeros((40, 25))

    action_combo = [(i, j) for i in range(5) for j in range(5) if i != j]
    for i, val in enumerate(action_combo):
        a, b = val
        A[i, 5*a:5*a+5] = q_s_a[a] - q_s_a[b]
        indices = list(range(a, 25, 5))
        A[i + 20, indices] = q_s_b[:, a] - q_s_b[:, b]

    A = matrix(
        np.vstack(
            (np.vstack(
                (np.hstack((np.ones((40, 1)), A)),
                 -np.hstack((np.zeros((25, 1)), np.eye(25))))),
             temp,
             -np.asarray(temp)
            )
        )
    )

    b = matrix(np.hstack((np.zeros(A.size[0] - 2), [1, -1])))

    sol = lp(c, A, b, solver='glpk')['x']

    if sol:
        solution = sol[1:]
        return np.matmul(q_s_a.flatten(), solution)[0], np.matmul(q_s_b.transpose().flatten(), solution)[0]


def plot_diff(diff_values, iterations):
    plt.xlabel("Iteration")
    plt.ylabel("Diff_Q")
    plt.ylim(0, 0.5)
    plt.plot(iterations, diff_values)
    plt.savefig("CeQ.png")
    plt.clf()


def ce_q(max_step=1000000, alpha=1., gamma=0.9):
    step = 0
    alpha_decay = 0.999993

    env = Soccer((0, 2), (0, 1))

    # Different Q Table Representation for ease of computation. Each element is an array of 5 X 5
    Q_A = {}
    Q_B = {}
    diff_values = []
    iteration = []
    value_A, value_B = 0, 0
    while step < max_step:
        state_A = env.encode_state("A")
        state_B = env.encode_state("B")
        state = state_A + state_B

        actions = env.sample(), env.sample()

        env.act(actions)
        internal_state = state_A + str(actions[0]) + str(actions[1])

        reward_A = env.get_reward("A")
        reward_B = env.get_reward("B")

        q_s_a_A = Q_A.get(state, np.ones((5, 5)))
        q_s_a_B = Q_B.get(state, np.ones((5, 5)))

        result = solve_ce_q(q_s_a_A, q_s_a_B)
        if result is not None:
            value_A, value_B = result

        new_q_s_a_A = (1 - alpha) * q_s_a_A[actions[0]][actions[1]] \
                      + alpha * (((1 - gamma) * reward_A) + (gamma * value_A))
        new_q_s_a_B = (1 - alpha) * q_s_a_B[actions[0]][actions[1]] \
                      + alpha * (((1 - gamma) * reward_B) + (gamma * value_B))

        if internal_state == "T214":
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
        q_s_a_B[actions[0]][actions[1]] = new_q_s_a_B
        Q_B[state] = q_s_a_B

ce_q()