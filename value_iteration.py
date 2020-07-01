import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix
from constants import *

from value_iteration_cy import optimize_T_sparse, next_values_sparse


def value_iteration(transition_matrix, reward_matrix, terminal_states, horizon=100, gamma=0.9, action_mask=None, q_default=-1, use_sparse_matrices=False):
    num_states = terminal_states.shape[0]
    if use_sparse_matrices:
        num_actions = int(transition_matrix.shape[0] / num_states)
    else:
        num_actions = transition_matrix.shape[1]

    if action_mask is None:
        action_mask = np.ones([num_states, num_actions])

    values = np.zeros([num_states], dtype=DTYPE)
    q_values = np.zeros([num_states, num_actions], dtype=DTYPE)

    if use_sparse_matrices:
        terminal_state_actions = (1 - terminal_states)[:, np.newaxis].repeat(num_actions, axis=1).flatten()
        
        for h in range(horizon):
            T = transition_matrix
            next_values = next_values_sparse(T.data, T.indices, T.indptr, values, num_states * num_actions)
            q_values = reward_matrix.flatten() + gamma * next_values * terminal_state_actions
            q_values = q_values.reshape([num_states, num_actions])
            values = np.max(np.where(action_mask == 1, q_values, q_default), axis=1)
    else:
        for h in range(horizon):
            q_values = reward_matrix + \
                       gamma * np.sum(transition_matrix * values[np.newaxis, np.newaxis, :], axis=2) * (1 - terminal_states)[:, np.newaxis]
            values = np.max(np.where(action_mask == 1, q_values, q_default), axis=1)

    return values, q_values


def rmax_value_iteration(transition_matrix, reward_matrix, terminal_states,
                         confident_actions, vmax,
                         horizon=100, gamma=0.9,
                         action_mask=None, q_default=-1, use_sparse_matrices=False):
    num_states = terminal_states.shape[0]
    if use_sparse_matrices:
        num_actions = int(transition_matrix.shape[0] / num_states)
    else:
        num_actions = transition_matrix.shape[1]
    
    if action_mask is None:
        action_mask = np.ones([num_states, num_actions])
    
    values = np.zeros([num_states], dtype=DTYPE)
    q_values = np.zeros([num_states, num_actions], dtype=DTYPE)
    q_values = np.where(confident_actions == 1, q_values, vmax)
    
    if use_sparse_matrices:
        non_terminal_sa_pairs = (1 - terminal_states)[:, np.newaxis].repeat(num_actions, axis=1).flatten()
        
        for h in range(horizon):
            T = transition_matrix
            next_values = next_values_sparse(T.data, T.indices, T.indptr, values,
                                                                num_states * num_actions)
            q_values = reward_matrix.flatten() + gamma * next_values * non_terminal_sa_pairs
            q_values = q_values.reshape([num_states, num_actions])
            q_values = np.where(np.logical_or(confident_actions == 1, terminal_states[:, np.newaxis]), q_values, vmax)
            q_values = np.where(action_mask == 1, q_values, q_default)
            values = np.max(q_values, axis=1)
    else:
        for h in range(horizon):
            q_values = reward_matrix + \
                       gamma * np.sum(transition_matrix * values[np.newaxis, np.newaxis, :], axis=2) * (1 - terminal_states)[:, np.newaxis]
            q_values = np.where(confident_actions == 1, q_values, vmax)
            q_values = np.where(action_mask == 1, q_values, q_default)
            values = np.max(q_values, axis=1)
    
    return values, q_values


def policy_value_iteration(transition_matrix, reward_matrix, terminal_states, policy, horizon=100, gamma=0.9, action_mask=None, q_default=-1, use_sparse_matrices=False):
    num_states = terminal_states.shape[0]
    if use_sparse_matrices:
        num_actions = int(transition_matrix.shape[0] / num_states)
    else:
        num_actions = transition_matrix.shape[1]

    if action_mask is None:
        action_mask = np.ones([num_states, num_actions], dtype=DTYPE)

    values = np.zeros([num_states], dtype=DTYPE)
    q_values = np.zeros([num_states, num_actions], dtype=DTYPE)

    if use_sparse_matrices:
        non_terminal_sa_pairs = (1 - terminal_states)[:, np.newaxis].repeat(num_actions, axis=1).flatten()
        T = transition_matrix
        
        for h in range(horizon):
            next_values = next_values_sparse(T.data, T.indices, T.indptr, values, num_states * num_actions)
            q_values = reward_matrix.flatten() + gamma * next_values * non_terminal_sa_pairs
            q_values = q_values.reshape([num_states, num_actions])
            q_values = np.where(action_mask == 1, q_values, q_default)
            values = q_values[range(num_states), policy]
    else:
        for h in range(horizon):
            q_values = reward_matrix + \
                       gamma * np.sum(transition_matrix * values[np.newaxis, np.newaxis, :], axis=2) * (1 - terminal_states)[:, np.newaxis]
            q_values = np.where(action_mask == 1, q_values, q_default)
            values = q_values[range(num_states), policy]

    return values, q_values


def optimistic_value_iteration(transition_matrix, reward_matrix, terminal_states, eps_R, eps_T,
                               horizon=100, gamma=0.9, pessimistic=False, action_mask=None, q_default=-1,
                               support_mask=None, use_sparse_matrices=False):
    num_states = terminal_states.shape[0]
    if use_sparse_matrices:
        num_actions = int(transition_matrix.shape[0] / num_states)
    else:
        num_actions = transition_matrix.shape[1]

    if action_mask is None:
        action_mask = np.ones([num_states, num_actions], dtype=DTYPE)

    if support_mask is None:
        support_mask = np.ones_like(transition_matrix, dtype=DTYPE)

    # TODO: include optimistic initialization
    values = np.zeros([num_states], dtype=DTYPE)
    q_values = np.zeros([num_states, num_actions], dtype=DTYPE)

    if pessimistic:
        reward_matrix -= eps_R
    else:
        reward_matrix += eps_R

    if use_sparse_matrices:
        non_terminal_sa_pairs = np.logical_not(terminal_states)[:, np.newaxis].repeat(num_actions, axis=1).flatten()

        for h in range(horizon):
            assert transition_matrix.nnz == support_mask.nnz
            
            if pessimistic:
                T = transition_matrix.copy()
                optimize_T_sparse(T.data, T.indices, T.indptr, support_mask.data, -values, eps_T.flatten(), num_states * num_actions)
            else:
                T = transition_matrix.copy()
                optimize_T_sparse(T.data, T.indices, T.indptr, support_mask.data, values, eps_T.flatten(), num_states * num_actions)

            assert (T.indices == support_mask.indices).all()
            assert (T.indptr == support_mask.indptr).all()

            next_values = next_values_sparse(T.data, T.indices, T.indptr, values, num_states * num_actions)
            q_values = reward_matrix.flatten() + gamma * next_values * non_terminal_sa_pairs
            q_values = q_values.reshape([num_states, num_actions])
            q_values = np.where(action_mask == 1, q_values, q_default)
            values = np.max(q_values, axis=1)
    else:
        xv, yv = np.meshgrid(np.arange(num_actions), np.arange(num_states))
        for h in range(horizon):
            if pessimistic:
                T = optimize_T_fast_2(transition_matrix, -values, eps_T, xv, yv, support_mask)
            else:
                T = optimize_T_fast_2(transition_matrix, values, eps_T, xv, yv, support_mask)

            q_values = reward_matrix + \
                       gamma * np.sum(T * values[np.newaxis, np.newaxis, :], axis=2) * (1 - terminal_states)[:,
                                                                                       np.newaxis]
            q_values = np.where(action_mask == 1, q_values, q_default)
            values = np.max(q_values, axis=1)

    return values, q_values, T


def calculate_state_visitation(transition_matrix, terminal_states, policy, state, horizon=100, gamma=0.9,
                               use_sparse_matrices=False):
    num_states = terminal_states.shape[0]

    if use_sparse_matrices:
        num_actions = int(transition_matrix.shape[0] / num_states)

        rho = np.zeros(num_states)
        rho[state] = 1
        visitations = rho.copy()

        policy_i = np.ravel_multi_index([range(num_states), policy], [num_states, num_actions])
        T_policy = transition_matrix[policy_i, :].T

        for h in range(horizon):
            rho *= (1 - terminal_states)  # we can't transition anywhere after a terminal state
            rho = gamma * csr_matrix.dot(T_policy, rho.T)
            visitations += rho
    else:
        rho = np.zeros(num_states)
        rho[state] = 1
        visitations = rho

        T_policy = transition_matrix[range(num_states), policy, :].T

        for h in range(horizon):
            rho = gamma * np.dot(T_policy, (rho * (1 - terminal_states[:, np.newaxis])).T)
            visitations += rho

    return visitations


def optimize_T(transition_matrix, values, eps_T, num_states, num_actions):
    T = transition_matrix.copy()
    for s in range(num_states):
        for a in range(num_actions):
            change = 1
            diff = 0
            while change > FLOAT_TOLERANCE and diff < eps_T[s, a] - FLOAT_TOLERANCE:
                opt = np.argmax(values)

                worst = np.argmin(np.where(T[s, a, :] > FLOAT_TOLERANCE, values, np.inf))

                change = min(T[s, a, worst], (eps_T[s, a] - diff) / 2)
                T[s, a, opt] += change
                T[s, a, worst] -= change

                diff += 2 * change
    return T


def optimize_T_fast(transition_matrix, values, eps_T, num_states, num_actions, xv, yv):
    T = transition_matrix.copy()
    T_prev = np.ones_like(T)
    diff = np.zeros([num_states, num_actions])

    iters = 0
    opt = np.argmax(values)

    while np.any(np.abs(T - T_prev) > FLOAT_TOLERANCE):
        T_prev = T.copy()

        worst = np.argmin(np.where(T > FLOAT_TOLERANCE, np.reshape(values, [1, 1, -1]), np.inf), axis=2)
        worst_i = (yv, xv, worst)

        change = np.where(T[worst_i] < (eps_T - diff) / 2, T[worst_i], (eps_T - diff) / 2)
        T[:, :, opt] += change
        # np.put(T, worst_i, T[worst_i] - change)
        np.put_along_axis(T, worst.reshape([num_states, num_actions, 1]),
                          (T[worst_i] - change).reshape([num_states, num_actions, 1]), axis=2)

        diff = np.linalg.norm(T - transition_matrix, 1, axis=2)

        iters += 1

    return T


def optimize_T_fast_2(transition_matrix, values, eps_T, xv, yv, support_mask=None):
    T = transition_matrix.copy()

    num_support = np.sum(support_mask, axis=2).astype(dtype=np.int)

    worst_to_best = np.argsort(np.where(support_mask, values[np.newaxis, np.newaxis, :], np.inf), axis=2)
    delta = np.zeros_like(T)
    delta_sum = np.zeros_like(delta[:, :, 0])

    for j in range(int(np.max(num_support)) - 1):
        i = worst_to_best[:, :, j]
        diff = np.minimum((eps_T - delta_sum) / 2, T[yv, xv, i])
        diff = np.where(j < num_support - 1, diff, 0)
        delta[yv, xv, i] += diff
        delta_sum += diff

    T = T - delta
    i = worst_to_best[yv, xv, num_support - 1]
    T[yv, xv, i] += np.sum(delta, axis=2)

    return T


def optimize_T_fast_3(transition_matrix, values, eps_T, num_support, support_row, support_col):
    vmin = min(values.min(), -1)

    num_state_actions = transition_matrix.shape[0]

    T = transition_matrix.copy()

    k = np.max(num_support)

    valid_values = csr_matrix((values[support_col] - 2*vmin, (support_row, support_col)))

    delta_sum = np.zeros([num_state_actions])
    eps_T = eps_T.reshape(num_state_actions)
    i_vector = np.arange(num_state_actions)

    worst_to_best, num_args = _sparse_nz_argsort(valid_values, axis=1, k=k)

    temp = np.maximum(worst_to_best[:, :-1], 0)
    diffs = np.zeros([num_state_actions, k - 1])
    T_slices = T[i_vector.repeat(k - 1), temp.flatten()].reshape([num_state_actions, -1])

    for j in range(k - 1):
        diff = np.minimum(T_slices[:, j].flatten(), (eps_T - delta_sum) / 2)
        diff = np.where(j < num_args - 1, diff, 0).flatten()
        diffs[:, j] = diff
        delta_sum += diff

    T = T - csr_matrix((diffs.flatten(), (i_vector.repeat(k - 1), temp.flatten())))
    best_i = worst_to_best[i_vector, num_args - 1]
    T[i_vector, best_i] += delta_sum

    return T


# adapted from https://github.com/scipy/scipy/blob/v1.2.1/scipy/sparse/data.py#L352-L373
def _sparse_nz_argsort(mat, axis, k):
    if mat.shape[axis] == 0:
        raise ValueError("Can't apply the operation along a zero-sized "
                         "dimension.")

    if axis < 0:
        axis += 2

    ret_size, line_size = mat._swap(mat.shape)
    ret = -np.ones([ret_size, k], dtype=int)
    num_args = -np.ones([ret_size], dtype=np.int)

    nz_lines, = np.nonzero(np.diff(mat.indptr))
    for i in nz_lines:
        p, q = mat.indptr[i:i + 2]
        data = mat.data[p:q]
        indices = mat.indices[p:q]
        am = np.argsort(data)
        num_args[i] = am.shape[0]
        ret[i, :am.shape[0]] = indices[am]

    return ret, num_args


def optimize_T_fast_4(T_data,
                      T_indices,
                      T_indprt,
                      V,
                      eps_T,
                      num_state_actions):
    for i in range(num_state_actions):
        T_ind = T_indices[T_indprt[i]:T_indprt[i+1]]
        sorted_T_ind = np.argsort(V[T_ind])

        budget = eps_T[i] / 2
        j = 0
        best_i = sorted_T_ind.shape[0] - 1
        while budget > 1e-12:
            transfer = min(budget, T_data[T_indprt[i] + sorted_T_ind[j]])
            T_data[T_indprt[i] + sorted_T_ind[best_i]] += transfer
            T_data[T_indprt[i] + sorted_T_ind[j]] -= transfer
            budget -= transfer
            j += 1
        assert budget >= 0


# def optimize_T_cvxpy(transition_matrix, values, eps_T, num_states, num_actions):
#     values_tiled = np.repeat(np.reshape(values, [1, -1]), num_actions, axis=0)
#     full_T = transition_matrix.copy()
#
#     for s in range(num_states):
#         T = cp.Variable((num_actions, num_states))
#
#         q_values_var = cp.sum(cp.multiply(T, values_tiled), axis=1)
#         objective = cp.Maximize(cp.sum(q_values_var))
#         constraints = [cp.sum(cp.abs(T - transition_matrix[s, :, :]), axis=1) <= eps_T[s],
#                        T >= 0, T <= 1, cp.sum(T, axis=1) == 1]
#         prob = cp.Problem(objective, constraints)
#         prob.solve()
#
#         full_T[s, :, :] = T
#
#     return full_T
