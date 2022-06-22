from copy import deepcopy
import numpy as np


def get_states_list(mdp):
    states = []
    for r in range(mdp.num_row):
        for c in range(mdp.num_col):
            if mdp.board[r][c] != 'WALL':
                states.append((r, c))
    return states


def sum_for_action(mdp, state, action_chosen, U):
    _sum = 0
    actions = ["UP", "DOWN", "RIGHT", "LEFT"]
    for action_taken_idx, action_taken in enumerate(actions):
        next_state = mdp.step(state, action_taken)
        _sum += mdp.transition_function[action_chosen][action_taken_idx] * U[next_state[0]][next_state[1]]
    return _sum


def value_iteration(mdp, U_init, epsilon=10 ** (-3)):
    # TODO:
    # Given the mdp, the initial utility of each state - U_init,
    #   and the upper limit - epsilon.
    # run the value iteration algorithm and
    # return: the U obtained at the end of the algorithms' run.
    #

    # ====== YOUR CODE: ======
    U_curr = deepcopy(U_init)
    states = get_states_list(mdp)
    row, col = 0, 1
    while True:
        U, delta = deepcopy(U_curr), 0
        for state in states:
            if state in mdp.terminal_states:
                U_curr[state[row]][state[col]] = float(mdp.board[state[row]][state[col]])
            else:
                # action_sums = []
                # for action_chosen in mdp.actions:
                #     action_sums.append(sum_for_action(mdp, state, action_chosen, U))
                action_sums = [sum_for_action(mdp, state, action_chosen, U) for action_chosen in mdp.actions]
                U_curr[state[row]][state[col]] = float(mdp.board[state[row]][state[col]]) + mdp.gamma * max(action_sums)

            if abs(U_curr[state[row]][state[col]] - U[state[row]][state[col]]) > delta:
                delta = float(abs(U_curr[state[row]][state[col]] - U[state[row]][state[col]]))

        if mdp.gamma == 1 and delta == 0.0:
            break

        if mdp.gamma != 1 and delta < ((epsilon * (1 - mdp.gamma)) / mdp.gamma):
            break

    return U
    # ========================


def get_policy(mdp, U):
    # TODO:
    # Given the mdp and the utility of each state - U (which satisfies the Belman equation)
    # return: the policy
    #

    # ====== YOUR CODE: ======
    actions = ["UP", "DOWN", "RIGHT", "LEFT"]
    policy = [["-"]*4 for _ in range(3)]
    for r in range(mdp.num_row):
        for c in range(mdp.num_col):

            if (r, c) in mdp.terminal_states or mdp.board[r][c] == "WALL":
                policy[r][c] = mdp.board[r][c]

            else:
                state = (r, c)
                action_sums = [sum_for_action(mdp, state, action_chosen, U) for action_chosen in mdp.actions]
                reward = mdp.board[r][c]
                max_action_sum = (U[r][c] - float(reward)) / mdp.gamma
                policy[r][c] = actions[action_sums.index(max_action_sum)]
    return policy
# ========================


def src_to_dst_probability(mdp, action, src, dst):
    r, c = 0, 1
    actions = ["UP", "DOWN", "RIGHT", "LEFT"]
    if abs(dst[c] - src[c]) + abs(dst[r] - src[r]) >= 2 or src in mdp.terminal_states:
        return 0

    actions_taken = []
    for act in mdp.actions:
        if mdp.step(src, act) == dst:
            actions_taken.append(act)

    if len(actions_taken) == 0:
        return 0
    sum_prob = sum([mdp.transition_function[action][actions.index(act)] for act in actions_taken])
    return sum_prob


def get_probability_mat(mdp, policy, states):
    p_mat = np.zeros((11, 11))
    for src, state_src in enumerate(states):
        for dst, state_dst in enumerate(states):
            if state_src in mdp.terminal_states:
                p_mat[src][dst] = 0
            else:
                p_mat[src][dst] = src_to_dst_probability(mdp, policy[state_src[0]][state_src[1]], state_src, state_dst)
    return p_mat


def policy_evaluation(mdp, policy):
    # TODO:
    # Given the mdp, and a policy
    # return: the utility U(s) of each state s
    #

    # ====== YOUR CODE: ======
    states = get_states_list(mdp)
    rewards = []
    for state in states:
        rewards.append(float(mdp.board[state[0]][state[1]]))
    R = np.array(rewards)
    I = np.identity(11)
    P = get_probability_mat(mdp, policy, states)
    inv_mat = np.linalg.inv(I - np.multiply(P, mdp.gamma))
    U_pi = list(inv_mat.dot(R))
    U_to_ret = [[0]*4 for _ in range(3)]

    for idx, state in enumerate(states):
        U_to_ret[state[0]][state[1]] = U_pi[idx]

    return U_to_ret
    # ========================


def policy_iteration(mdp, policy_init):
    # TODO:
    # Given the mdp, and the initial policy - policy_init
    # run the policy iteration algorithm
    # return: the optimal policy
    #
    # ====== YOUR CODE: ======
    actions = ["UP", "DOWN", "RIGHT", "LEFT"]
    policy = deepcopy(policy_init)
    states = get_states_list(mdp)
    r, c = 0, 1
    while True:
        U = policy_evaluation(mdp, policy)
        unchanged = True
        for state in states:
            if state in mdp.terminal_states:
                continue
            action_sums = [sum_for_action(mdp, state, action_chosen, U) for action_chosen in mdp.actions]
            if max(action_sums) > sum_for_action(mdp, state, policy[state[r]][state[c]], U):
                policy[state[r]][state[c]] = actions[action_sums.index(max(action_sums))]
                unchanged = False
        if unchanged:
            break

    return policy
    # ========================
