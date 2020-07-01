# Python imports.

# Other imports.=
from agents.safe_agent import SafeAgent

from constants import *
from dynamic_programming import value_iteration


class ASEAgent(SafeAgent):
    def __init__(self, actions, states, reward_func, initial_safe_states, initial_safe_actions,
                 similarity_function, analagous_state_function,
                 transition_support_function=None,
                 gamma=0.99, vi_horizon=100, name='ase-agent', beta_T=0.5,
                 tau=0.1, update_frequency=100,
                 directed=True,
                 use_sparse_matrices=False):
        SafeAgent.__init__(self, actions, states, reward_func, initial_safe_states, initial_safe_actions,
                           similarity_function, analagous_state_function,
                           transition_support_function, gamma, vi_horizon, name, beta_T,
                           tau, update_frequency, use_sparse_matrices)
        
        self.q_goal = np.zeros([self.num_states, self.num_actions], dtype=DTYPE)
        self.q_explore = np.zeros([self.num_states, self.num_actions], dtype=DTYPE)
        self.q_z_goal = np.zeros([self.num_states, self.num_actions], dtype=DTYPE)

        self.z_unsafe = self.transition_table.rewards < 0
        self.z_goal = np.zeros([self.num_states, self.num_actions], dtype=np.bool)
        self.z_explore = np.zeros([self.num_states, self.num_actions], dtype=np.bool)
        
        self.directed = directed
    
    def act(self, state, reward, learning=True):
        s = self.state_to_id[state]
        if self.s0 is None:
            self.s0 = s
        
        if reward < 0:
            print('FAILURE: safe agent hit unsafe state')
        
        if learning:
            self.update(self.prev_state, self.prev_action, reward, state)
        
        safe_actions = self.z_safe[s, :]
        if safe_actions.sum() == 0 and not state.is_terminal():
            print('UNSAFE: safe policy is not defined for state %s' % state)
        
        if np.all(np.logical_and(self.z_goal, self.z_safe) == self.z_goal):  # Is z_goal a subset of z_safe?
            if self.z_goal[s, :].any():  # Is s in z_goal?
                qs = self.q_goal[s, :]
            else:
                qs = self.q_z_goal[s, :]
        else:
            qs = self.q_explore[s, :]
        
        masked_qs = np.where(safe_actions, qs, -np.inf)
        a = np.argmax(masked_qs)
        action = self.actions[a]
        
        self.prev_state = None if state.is_terminal() else state
        
        self.prev_action = action
        
        return action
    
    def update(self, state, action, reward, next_state):
        # If this is the first state, just return.
        if state is None:
            return
        
        self.transition_table.insert(state, action, reward, next_state)
        
        if self.step_number % self.update_frequency == 0:
            transition_matrix, reward_matrix, terminal_states, eps_T = self.transition_table.prepare_for_vi(tau=self.tau)
            
            self.z_safe = self.calculate_z_safe(transition_matrix, eps_T, terminal_states)
            self.z_goal, self.z_explore = self.calculate_z_goal(transition_matrix, reward_matrix, eps_T, terminal_states)
            
            if not np.all(np.logical_and(self.z_goal, self.z_safe) == self.z_goal):  # Is z_goal a subset of z_safe?
                _, self.q_explore, _ = value_iteration.optimistic_value_iteration(transition_matrix, self.z_explore.astype(np.float32),
                                                                                  terminal_states,
                                                                                  eps_R=0, eps_T=eps_T,
                                                                                  gamma=self.gamma,
                                                                                  horizon=self.vi_horizon,
                                                                                  pessimistic=False,
                                                                                  action_mask=self.z_safe, q_default=-100,
                                                                                  support_mask=self.transition_table.support_mask,
                                                                                  use_sparse_matrices=self.use_sparse_matrices)
            else:
                _, self.q_z_goal, _ = value_iteration.optimistic_value_iteration(transition_matrix, self.z_goal.astype(np.float32),
                                                                                 terminal_states,
                                                                                 eps_R=0, eps_T=eps_T,
                                                                                 gamma=self.gamma,
                                                                                 horizon=self.vi_horizon,
                                                                                 pessimistic=False,
                                                                                 action_mask=self.z_safe, q_default=-100,
                                                                                 support_mask=self.transition_table.support_mask,
                                                                                 use_sparse_matrices=self.use_sparse_matrices)
        
        self.step_number += 1
    
    def calculate_z_explore(self, z_edge, transition_matrix, eps_T, terminal_states):
        T_indptr = transition_matrix.indptr
        T_indices = transition_matrix.indices
        T_data = transition_matrix.data
        
        z_explore = np.zeros_like(self.z_safe)
        z_ret = np.zeros_like(self.z_safe)
        
        z_next = np.zeros_like(self.z_safe)
        while z_explore.sum() == 0 and z_edge.sum() > 0:
            for s in range(self.num_states):
                for a in range(self.num_actions):
                    if z_edge[s, a]:
                        sa = np.ravel_multi_index([s, a], [self.num_states, self.num_actions])
                        z_ret[s, a] = 1
                        if eps_T[s, a] > self.tau:
                            p = self.transition_table.sa_similarity_matrix.indptr[sa]
                            q = self.transition_table.sa_similarity_matrix.indptr[sa+1]
                            for i, sa2 in enumerate(self.transition_table.sa_similarity_matrix.indices[p:q]):
                                s2, a2 = np.unravel_index(sa2, [self.num_states, self.num_actions])
                                if self.z_safe[s2, a2] and not terminal_states[s2]:
                                    delta = 1 - self.transition_table.sa_similarity_matrix.data[p + i]
                                    if delta <= self.tau / 2:
                                        z_explore[s2, a2] = 1
                        else:
                            p = T_indptr[sa]
                            q = T_indptr[sa + 1]
                            for i, sp in enumerate(T_indices[p:q]):
                                if T_data[p + i] > 0 and not terminal_states[sp]:
                                    for ap in range(self.num_actions):
                                        if z_ret[sp, ap] == 0 \
                                                and self.z_safe[sp, ap] == 0 \
                                                and self.z_unsafe[sp, ap] == 0:
                                            z_next[sp, ap] = 1
            
            z_edge = z_next
            z_next = np.zeros_like(self.z_safe)
        return z_explore
    
    def calculate_z_goal(self, transition_matrix, reward_matrix, eps_T, terminal_states):
        z_explore = np.zeros_like(self.z_safe)
        
        while True:
            _, self.q_goal, T_opt = value_iteration.optimistic_value_iteration(transition_matrix, reward_matrix,
                                                                               terminal_states,
                                                                               eps_R=0, eps_T=eps_T,
                                                                               gamma=self.gamma,
                                                                               horizon=self.vi_horizon,
                                                                               pessimistic=False,
                                                                               support_mask=self.transition_table.support_mask,
                                                                               use_sparse_matrices=self.use_sparse_matrices,
                                                                               action_mask=1 - self.z_unsafe,
                                                                               q_default=-100)
            pi_goal = np.argmax(self.q_goal, axis=1)
            rho_goal_state = value_iteration.calculate_state_visitation(T_opt, terminal_states,
                                                                        policy=pi_goal, state=self.s0,
                                                                        gamma=self.gamma,
                                                                        horizon=self.vi_horizon,
                                                                        use_sparse_matrices=self.use_sparse_matrices)
            rho_goal = np.zeros([self.num_states, self.num_actions])
            rho_goal[range(self.num_states), pi_goal] = rho_goal_state
            z_goal = rho_goal > 0
            
            if np.all(np.logical_and(z_goal, self.z_safe) == z_goal):  # z_goal is a subset of z_safe
                break

            if self.directed:
                z_edge = np.where(np.max(self.z_safe, axis=1)[:, np.newaxis], np.logical_and(z_goal, 1 - self.z_safe), 0)
            else:
                z_edge = np.where(np.max(self.z_safe, axis=1)[:, np.newaxis], 1 - self.z_safe, 0)
            if z_edge.sum() == 0:
                raise ValueError('Z_edge should not be empty')
            
            z_explore = self.calculate_z_explore(z_edge, transition_matrix, eps_T, terminal_states)
            if z_explore.sum() > 0:
                break
            else:
                self.z_unsafe = np.logical_or(self.z_unsafe, z_edge)
        
        return z_goal, z_explore
