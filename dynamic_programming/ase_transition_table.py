import os
import pickle
from scipy.sparse import csr_matrix

from dynamic_programming.transition_table import TransitionTable
from constants import *


class AnalogousStateTransitionTable(TransitionTable):

    def __init__(self, states, actions, similarity_function, analagous_state_function, initial_safe_sa,
                 reward_func,
                 support_function=None, use_sparse_matrices=False, beta_T=0.5):
        TransitionTable.__init__(self, states, actions, reward_func=reward_func, use_sparse_matrices=use_sparse_matrices)

        if support_function is None:
            self.support_mask = None
        else:
            support_mask_file = os.path.join(os.getcwd(), 'support_mask.pkl')
            if os.path.exists(support_mask_file):
                self.support_mask = pickle.load(open(support_mask_file, 'rb'))
            else:
                self.support_mask = None

            if self.support_mask is None or \
                    self.support_mask.shape != (self.num_states * self.num_actions, self.num_states):
                self.support_mask = support_function(list(states), list(actions)).astype(DTYPE)
                # remove all support for Z_0 outside of Z_0
                s_safe = initial_safe_sa.max(axis=1)
                for s in range(self.num_states):
                    for a in range(self.num_actions):
                        if initial_safe_sa[s, a]:
                            sa = np.ravel_multi_index([s, a], [self.num_states, self.num_actions])
                            
                            p = self.support_mask.indptr[sa]
                            q = self.support_mask.indptr[sa + 1]
                            for i, sp in enumerate(self.support_mask.indices[p:q]):
                                if s_safe[sp] == 0:
                                    self.support_mask.data[p + i] = 0
                    
                pickle.dump(self.support_mask, open(support_mask_file, 'wb'))

        if self.use_sparse_matrices:
            assert support_function is not None

            data = np.zeros(self.support_mask.nnz)
            indices = self.support_mask.indices
            indptr = self.support_mask.indptr
            self.transition_sums = csr_matrix((data, indices, indptr), dtype=DTYPE)
            assert self.transition_sums.nnz == self.support_mask.nnz
        else:
            self.transition_sums = np.zeros([self.num_states, self.num_actions, self.num_states], dtype=DTYPE)

        self.T_hat = self.transition_sums.copy()
        self.eps_T = 2 * np.ones([self.num_states, self.num_actions], dtype=DTYPE)
        self.eps_T_tilde = self.eps_T.copy()
        self.beta_T = beta_T
        self.using_analagous_ci = np.zeros([self.num_states, self.num_actions], dtype=np.bool)

        self.similarity_function = similarity_function
        self.analagous_state_function = analagous_state_function

        # load or construct sa_similarity_matrix
        sa_similarity_matrix_file = os.path.join(os.getcwd(), 'sa_similarity_matrix.pkl')
        if os.path.exists(sa_similarity_matrix_file):
            self.sa_similarity_matrix = pickle.load(open(sa_similarity_matrix_file, 'rb'))
        else:
            self.sa_similarity_matrix = None

        if self.sa_similarity_matrix is None or \
                self.sa_similarity_matrix.shape != (self.num_states*self.num_actions, self.num_states*self.num_actions):
            self.sa_similarity_matrix = similarity_function(list(states), list(actions)).astype(DTYPE)
            pickle.dump(self.sa_similarity_matrix, open(sa_similarity_matrix_file, 'wb'))
    
    def insert(self, state, action, reward, next_state):
        TransitionTable.insert(self, state, action, reward, next_state)

        s = self.state_to_id[state]
        a = self.action_to_id[action]
        sp = self.state_to_id[next_state]

        if self.use_sparse_matrices:
            sa = np.ravel_multi_index([s, a], [self.num_states, self.num_actions])
            
            self.transition_sums[sa, sp] += 1
            self.T_hat[sa, sp] += 1
            sa_count = self.transition_sums[sa, :].sum()
            assert self.support_mask.nnz == self.transition_sums.nnz
        else:
            self.transition_sums[s, a, sp] += 1
            self.T_hat[s, a, sp] += 1
            sa_count = self.transition_sums[s, a, :].sum()

        self.eps_T[s, a] = np.minimum(2, self.beta_T * np.power(sa_count, -1/2))
        if self.eps_T[s, a] < self.eps_T_tilde[s, a]:
            self.eps_T_tilde[s, a] = self.eps_T[s, a]
            if self.using_analagous_ci[s, a]:
                self.using_analagous_ci[s, a] = False
                self.transfer_T_hat(state, action, state, action, s, a, s, a)

        # update the confidence intervals of all similar state-actions
        if self.use_sparse_matrices:
            sa = np.ravel_multi_index([s, a], [self.num_states, self.num_actions])

            p = self.sa_similarity_matrix.indptr[sa]
            q = self.sa_similarity_matrix.indptr[sa + 1]
            for i, sa2 in enumerate(self.sa_similarity_matrix.indices[p:q]):
                delta = 1 - self.sa_similarity_matrix.data[p + i]
                s2, a2 = np.unravel_index(sa2, [self.num_states, self.num_actions])
                if self.eps_T[s, a] + 2 * delta < self.eps_T_tilde[s2, a2]:
                    state2 = self.states[s2]
                    action2 = self.actions[a2]
                    self.using_analagous_ci[s2, a2] = True
                    self.eps_T_tilde[s2, a2] = self.eps_T[s, a] + 2 * delta
                    self.transfer_T_hat(state, action, state2, action2, s, a, s2, a2)
        else:
            for s2, state2 in enumerate(self.states):
                for a2, action2 in enumerate(self.actions):
                    delta = 1 - self.sa_similarity_matrix[s, a, s2, a2]
                    if self.eps_T[s, a] + 2 * delta < self.eps_T[s2, a2]:
                        self.using_analagous_ci[s2, a2] = True
                        self.eps_T_tilde[s2, a2] = self.eps_T[s, a] + 2 * delta
                        self.transfer_T_hat(state, action, state2, action2, s, a, s2, a2)
    
    def transfer_T_hat(self, state1, action1, state2, action2, s1, a1, s2, a2):
        if self.use_sparse_matrices:
            sa1 = np.ravel_multi_index([s1, a1], [self.num_states, self.num_actions])
            sa2 = np.ravel_multi_index([s2, a2], [self.num_states, self.num_actions])

            p2 = self.transition_sums.indptr[sa2]
            q2 = self.transition_sums.indptr[sa2 + 1]
            sp2s = self.transition_sums.indices[p2:q2]
            self.T_hat.data[p2:q2] = 0

            p1 = self.transition_sums.indptr[sa1]
            q1 = self.transition_sums.indptr[sa1 + 1]
            for i in np.argwhere(self.transition_sums.data[p1:q1] > 0).flatten():
                sp1 = self.transition_sums.indices[p1 + i]
                state1_prime1 = self.states[sp1]
                state_prime2 = self.analagous_state_function(state1, action1, state1_prime1, state2, action2)
                if state_prime2 is not None and state_prime2 in self.state_to_id:
                    sp2 = self.state_to_id[state_prime2]
                    j = np.argwhere(sp2s == sp2).flatten()
                    if j.shape[0] == 0:
                        ValueError('Analogous state outside of initial support!')
                    else:
                        self.T_hat.data[p2 + j[0]] = self.transition_sums.data[p1 + i]
            
            assert self.support_mask.nnz == self.T_hat.nnz
        else:
            self.T_hat[s2, a2, :] = 0
            for sp1, state1_prime1 in enumerate(self.num_states):
                if self.transition_sums[s1, a1, sp1] > 0:
                    state_prime2 = self.analagous_state_function(state1, action1, state1_prime1, state2, action2)
                    if state_prime2 is not None and state_prime2 in self.state_to_id:
                        sp2 = self.state_to_id[state_prime2]
                        self.T_hat[s2, a2, sp2] = self.transition_sums[s1, a1, sp1]
    
    def prepare_for_vi(self, tau=None):
        if self.use_sparse_matrices:
            assert self.support_mask.nnz == self.T_hat.nnz
            data = np.zeros(self.support_mask.nnz)
            indices = self.support_mask.indices
            indptr = self.support_mask.indptr
            
            if tau is not None:
                for s in range(self.num_states):
                    for a in range(self.num_actions):
                        if self.eps_T_tilde[s, a] < tau:
                            sa = np.ravel_multi_index([s, a], [self.num_states, self.num_actions])
                            p = indptr[sa]
                            q = indptr[sa + 1]
                            self.support_mask.data[p:q] = np.logical_and(self.support_mask.data[p:q], self.T_hat.data[p:q] > 0)
            
            total_support = np.maximum(np.asarray(self.support_mask.sum(axis=1)).flatten(), 1)
            state_action_totals = np.asarray(self.T_hat.sum(axis=1)).flatten()
            
            for sa in range(self.num_states * self.num_actions):
                p = indptr[sa]
                q = indptr[sa + 1]
                if state_action_totals[sa] == 0:
                    data[p:q] = 1/total_support[sa] * self.support_mask.data[p:q]
                else:
                    data[p:q] = self.T_hat.data[p:q]/state_action_totals[sa]
            
            transition_matrix = csr_matrix((data, indices, indptr), dtype=DTYPE)
        else:
            transition_sums = np.where(np.logical_and((np.sum(self.T_hat, axis=2) == 0)[:, :, np.newaxis],
                                                      self.support_mask),
                                       1, self.T_hat)

            transition_matrix = transition_sums / np.sum(transition_sums, axis=2)[:, :, np.newaxis]

        return transition_matrix, self.rewards, self.terminal_states, self.eps_T_tilde
