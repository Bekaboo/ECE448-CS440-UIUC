"""
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
"""
import numpy as np
import itertools

epsilon = 1e-3


def compute_transition_matrix(model):
    """
    Parameters:
    model - the MDP model returned by load_MDP()

    Output:
    P - An M x N x 4 x M x N numpy array. P[r, c, a, r', c'] is the probability that the agent will move from cell (r, c) to (r', c') if it takes action a, where a is 0 (left), 1 (up), 2 (right), or 3 (down).
    """

    def reachable(row, col, model):
        return (
            row >= 0
            and row < model.M
            and col >= 0
            and col < model.N
            and not model.W[row, col]
        )

    transition_matrix = np.zeros((model.M, model.N, 4, model.M, model.N))
    actions = ([0, -1], [-1, 0], [0, 1], [1, 0])
    for row, col in itertools.product(range(model.M), range(model.N)):
        if model.T[row, col]:  # terminal state
            continue
        for actionnr in range(4):
            action = actions[actionnr]
            for flag in range(3):
                if flag == 1:
                    action = actions[(actionnr + 3) % 4]  # rotate left
                elif flag == 2:
                    action = actions[(actionnr + 1) % 4]  # rotate right
                new_row = row + action[0]
                new_col = col + action[1]
                if reachable(new_row, new_col, model):
                    transition_matrix[
                        row, col, actionnr, new_row, new_col
                    ] += model.D[row, col, flag]
                else:
                    transition_matrix[row, col, actionnr, row, col] += model.D[
                        row, col, flag
                    ]
    return transition_matrix


def update_utility(model, P, U_current):
    """
    Parameters:
    model - The MDP model returned by load_MDP()
    P - The precomputed transition matrix returned by compute_transition_matrix()
    U_current - The current utility function, which is an M x N array

    Output:
    U_next - The updated utility function, which is an M x N array
    """
    U_next = np.zeros((model.M, model.N))
    for row, col in itertools.product(range(model.M), range(model.N)):
        max_score = -np.inf
        for actionnr in range(4):
            score = 0
            for row2, col2 in itertools.product(
                range(model.M), range(model.N)
            ):
                score += (
                    P[row, col, actionnr, row2, col2] * U_current[row2, col2]
                )
            max_score = max(max_score, score)
        U_next[row, col] = model.R[row, col] + model.gamma * max_score
    return U_next


def value_iteration(model):
    """
    Parameters:
    model - The MDP model returned by load_MDP()

    Output:
    U - The utility function, which is an M x N array
    """
    eps = 1e-3
    numiter = 400
    U = np.zeros((model.M, model.N))
    transition_matrix = compute_transition_matrix(model)
    for cnt in range(numiter):
        should_break = True
        U_next = update_utility(model, transition_matrix, U)
        for row, col in itertools.product(range(model.M), range(model.N)):
            if abs(U_next[row, col] - U[row, col]) >= eps:
                should_break = False
        U = U_next
        if should_break:
            break
    return U


if __name__ == "__main__":
    import utils

    model = utils.load_MDP("models/small.json")
    model.visualize()
    U = value_iteration(model)
    model.visualize(U)
