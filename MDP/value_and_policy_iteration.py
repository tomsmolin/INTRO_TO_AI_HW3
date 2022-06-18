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
                action_sums = []
                for action_chosen in mdp.actions:
                    action_sums.append(sum_for_action(mdp, state, action_chosen, U))

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
    raise NotImplementedError
    # ========================


def policy_evaluation(mdp, policy):
    # TODO:
    # Given the mdp, and a policy
    # return: the utility U(s) of each state s
    #

    # ====== YOUR CODE: ======
    raise NotImplementedError
    # ========================


def policy_iteration(mdp, policy_init):
    # TODO:
    # Given the mdp, and the initial policy - policy_init
    # run the policy iteration algorithm
    # return: the optimal policy
    #

    # ====== YOUR CODE: ======
    raise NotImplementedError
    # ========================
