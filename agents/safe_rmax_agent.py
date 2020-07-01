# Python imports.

# Other imports.=
from agents.safe_agent import SafeAgent

from constants import *
from dynamic_programming import value_iteration


class SafeRMaxAgent(SafeAgent):
    
    def __init__(self, actions, states, reward_func, initial_safe_states, initial_safe_actions,
                 similarity_function, analagous_state_function,
                 transition_support_function=None,
                 gamma=0.99, vi_horizon=100, name='safe-rmax-agent', beta_T=0.5,
                 tau=0.1, update_frequency=100, rmax=1.0,
                 use_sparse_matrices=False, safe=True):
        SafeAgent.__init__(self, actions, states, reward_func, initial_safe_states, initial_safe_actions,
                           similarity_function, analagous_state_function,
                           transition_support_function, gamma, vi_horizon, name, beta_T,
                           tau, update_frequency, use_sparse_matrices)
        
        self.safe = safe
        self.q = np.zeros([self.num_states, self.num_actions], dtype=DTYPE)
        self.vmax = rmax  # Since we know all goal states are terminal
        if self.use_sparse_matrices:
            self.min_delta = 1 - np.where(self.transition_table.sa_similarity_matrix.data > 0, self.transition_table.sa_similarity_matrix.data, np.infty).min()
        else:
            self.min_delta = 1 - np.where(self.transition_table.sa_similarity_matrix > 0, self.transition_table.sa_similarity_matrix, np.infty).min()
    
    def act(self, state, reward, learning=True):
        s = self.state_to_id[state]
        if self.s0 is None:
            self.s0 = s
            
        if self.prev_state is not None:
            ps = self.state_to_id[self.prev_state]
            pa = self.action_to_id[self.prev_action]
            psa = np.ravel_multi_index([ps, pa], [self.num_states, self.num_actions])
            t_count = self.transition_table.T_hat[psa, s]
        
        if learning:
            self.update(self.prev_state, self.prev_action, reward, state)
        
        if self.safe:
            if reward < 0:
                print('FAILURE: safe agent hit unsafe state')
            
            safe_actions = self.z_safe[s, :]
            if safe_actions.sum() == 0:
                print('UNSAFE: safe policy is not defined for state %s' % state)
            
            qs = np.where(safe_actions, self.q[s, :], -np.inf)
        else:
            qs = self.q[s, :]

        a = np.argmax(qs)
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
            
            # find all actions that are unknown and all similar actions
            unknown_actions = (eps_T >= self.tau).flatten()
            similar_actions = unknown_actions.copy()
            terminal_state_actions = np.repeat(terminal_states[:, np.newaxis], self.num_actions, axis=1).flatten()
            for sa2 in np.argwhere(1 - unknown_actions).flatten():
                p = self.transition_table.sa_similarity_matrix.indptr[sa2]
                q = self.transition_table.sa_similarity_matrix.indptr[sa2 + 1]
                for i in np.argwhere(self.transition_table.sa_similarity_matrix.data[p:q] > 0):
                    sa1 = self.transition_table.sa_similarity_matrix.indices[p + i]
                    if unknown_actions[sa1] and not terminal_state_actions[sa1]:
                        similar_actions[sa2] = 1
                        break

            # unknown_actions = (eps_T >= self.tau).flatten()
            # similar_actions = unknown_actions.copy()
            # for sa1 in np.argwhere(unknown_actions).flatten():
            #     p = self.transition_table.sa_similarity_matrix.indptr[sa1]
            #     q = self.transition_table.sa_similarity_matrix.indptr[sa1 + 1]
            #     for i in np.argwhere(self.transition_table.sa_similarity_matrix.data[p:q] > 0):
            #         sa2 = self.transition_table.sa_similarity_matrix.indices[p + i]
            #         s2, a2 = np.unravel_index(sa2, [self.num_states, self.num_actions])
            #         if not terminal_states[s2]:
            #             similar_actions[sa2] = 1
            
            similar_actions.resize([self.num_states, self.num_actions])
            confident_actions = 1 - similar_actions
            
            action_mask = None
            if self.safe:
                self.z_safe = self.calculate_z_safe(transition_matrix, eps_T, terminal_states)
                action_mask = self.z_safe
            else:
                # ensure that the agent learns the safe-optimal policy
                reward_matrix = np.where(reward_matrix < 0, -100, reward_matrix)
            
            _, self.q = value_iteration.rmax_value_iteration(transition_matrix, reward_matrix, terminal_states,
                                                             confident_actions, self.vmax,
                                                             horizon=self.vi_horizon,
                                                             gamma=self.gamma,
                                                             action_mask=action_mask,
                                                             q_default=-100,
                                                             use_sparse_matrices=True)
        
        self.step_number += 1
