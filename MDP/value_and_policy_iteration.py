from copy import deepcopy
import numpy as np


def get_probability(mdp, next_state, state, action_chosen):
    action_made = action_chosen
    for action in mdp.actions:
        if mdp.step(state, action) == next_state:
            action_made = action
            break

    action_made_idx = ["UP", "DOWN", "RIGHT", "LEFT"].index(action_made)
    return mdp.transition_function[action_chosen][action_made_idx]


def reward(mdp, state):
    if mdp.board[state[0]][state[1]] == 'WALL':
        return None
    return float(mdp.board[state[0]][state[1]])


def get_states_list(mdp):
    states = []
    for r in range(mdp.num_row):
        for c in range(mdp.num_col):
            if mdp.board[r][c] != 'WALL':
                    states.append((r, c))
    return states


def possible_next_states(mdp, state):
    next_states = []
    for action in mdp.actions:
        next_state = mdp.step(state, action)
        if next_state not in next_states:
            next_states.append(next_state)

    return next_states


def value_iteration(mdp, U_init, epsilon=10 ** (-3)):
    # TODO:
    # Given the mdp, the initial utility of each state - U_init,
    #   and the upper limit - epsilon.
    # run the value iteration algorithm and
    # return: the U obtained at the end of the algorithms' run.
    #

    # ====== YOUR CODE: ======
    # raise NotImplementedError
    U_curr = U_init
    states = get_states_list(mdp)
    row, col = 0, 1
    while True:
        U, delta = U_curr, 0
        for state in states:

            bellman_rhs_max_options = []
            next_states = possible_next_states(mdp, state)

            for act_chosen in mdp.actions:
                p_u_elements = []
                for next_state in next_states:
                    p_u_elements.append(get_probability(mdp, next_state, state, act_chosen) * U[state[row]][state[col]])
                action_sum = sum(p_u_elements)
                bellman_rhs_max_options.append(action_sum)

            U_curr[state[row]][state[col]] = reward(mdp, state) + mdp.gamma * max(bellman_rhs_max_options)
            if abs(U_curr[state[row]][state[col]] - U[state[row]][state[col]]) > delta:
                delta = abs(U_curr[state[row]][state[col]] - U[state[row]][state[col]])

        if mdp.gamma == 1 and delta == 0:
            break

        if mdp.gamma != 1 and delta < ((epsilon * (1 - mdp.gamma))/mdp.gamma):
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
