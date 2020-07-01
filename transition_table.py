import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix, lil_matrix

import value_iteration


DTYPE = np.float32


class TransitionTable:
    def __init__(self, states, actions, reward_func=None, use_sparse_matrices=False):
        self.states = list(states)
        self.num_states = len(states)
        self.state_to_id = dict()
        for s_id, state in enumerate(self.states):
            self.state_to_id[state] = s_id

        self.actions = list(actions)
        self.num_actions = len(actions)
        self.action_to_id = dict()
        for i, a in enumerate(self.actions):
            self.action_to_id[a] = i

        self.sa_counts = np.zeros([self.num_states, self.num_actions], dtype=DTYPE)
        self.reward_sums = np.zeros([self.num_states, self.num_actions], dtype=DTYPE)

        self.use_sparse_matrices = use_sparse_matrices
        if self.use_sparse_matrices:
            self.transition_counts = lil_matrix((self.num_states * self.num_actions, self.num_states), dtype=DTYPE)
        else:
            self.transition_counts = np.zeros([self.num_states, self.num_actions, self.num_states], dtype=DTYPE)

        self.terminal_states = np.zeros([self.num_states], dtype=DTYPE)  # stores all terminal states

        self.predefined_rewards = reward_func is not None
        if self.predefined_rewards:
            self.rewards = np.zeros([self.num_states, len(self.actions)], dtype=DTYPE)
            for s, state in enumerate(self.states):
                for a, action in enumerate(self.actions):
                    self.rewards[s, a] = reward_func(state, action)
                if state.is_terminal():
                    self.terminal_states[s] = 1

    def insert(self, state, action, reward, next_state):
        s = self.state_to_id[state]
        a = self.action_to_id[action]
        sp = self.state_to_id[next_state]

        self.reward_sums[s, a] += reward
        self.sa_counts[s, a] += 1

        if self.use_sparse_matrices:
            sa = np.ravel_multi_index([s, a], [self.num_states, self.num_actions])
            self.transition_counts[sa, sp] += 1
        else:
            self.transition_counts[s, a, sp] += 1

    def prepare_for_vi(self):
        if self.use_sparse_matrices:
            # TODO: make this more efficient
            transition_matrix = np.where((self.sa_counts == 0)[:, :, np.newaxis],
                                         1. / self.num_states,
                                         self.transition_counts.toarray() / self.sa_counts[:, :, np.newaxis])
            transition_matrix = csr_matrix(transition_matrix)
        else:
            transition_matrix = np.where((self.sa_counts == 0)[:, :, np.newaxis],
                                         1./self.num_states,
                                         self.transition_counts/self.sa_counts[:, :, np.newaxis])

        if self.predefined_rewards:
            reward_matrix = self.rewards
        else:
            reward_matrix = np.where((self.sa_counts == 0),
                                     0,
                                     self.reward_sums/self.sa_counts)

        return transition_matrix, reward_matrix, self.terminal_states

    def reward_CI(self, beta_R, r_max=1):
        return np.minimum(r_max, beta_R * np.power(self.sa_counts, -1/2))

    def transition_CI(self, beta_T):
        return np.minimum(2, beta_T * np.power(self.sa_counts, -1/2))

    def id_for_state(self, state):
        return self.state_to_id[state]
