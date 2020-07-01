# Python imports.
import numpy as np

# Other imports.=
from safe_agent import SafeAgent

from constants import *
import value_iteration


class SafeEGreedyAgent(SafeAgent):
    
    def __init__(self, actions, states, reward_func, initial_safe_states, initial_safe_actions,
                 similarity_function, analagous_state_function,
                 epsilon=0.1, annealing_time=1000,
                 transition_support_function=None,
                 gamma=0.99, vi_horizon=100, name='safe-e-greedy-agent', beta_T=0.5,
                 tau=0.1, update_frequency=100,
                 use_sparse_matrices=False, safe=True):
        SafeAgent.__init__(self, actions, states, reward_func, initial_safe_states, initial_safe_actions,
                           similarity_function, analagous_state_function,
                           transition_support_function, gamma, vi_horizon, name, beta_T,
                           tau, update_frequency, use_sparse_matrices)
        
        self.safe = safe
        self.epsilon = epsilon
        self.annealing_time = annealing_time
        self.q = np.zeros([self.num_states, self.num_actions], dtype=DTYPE)
    
    def act(self, state, reward, learning=True):
        s = self.state_to_id[state]
        if self.s0 is None:
            self.s0 = s
        
        if self.prev_state is not None:
            ps = self.state_to_id[self.prev_state]
            pa = self.action_to_id[self.prev_action]
            psa = np.ravel_multi_index([ps, pa], [self.num_states, self.num_actions])
            T_hat = self.transition_table.T_hat[psa, :].toarray().flatten()
            T_count = T_hat[s]
        
        if learning:
            self.update(self.prev_state, self.prev_action, reward, state)

        if self.safe:
            if reward < 0:
                print('FAILURE: safe agent hit unsafe state')
            
            safe_actions = self.z_safe[s, :]
            if safe_actions.sum() == 0:
                print('UNSAFE: safe policy is not defined for state %s' % state)
                safe_actions = np.ones(self.num_actions)
    
            qs = np.where(safe_actions, self.q[s, :], -np.inf)
        else:
            safe_actions = np.ones_like(self.actions)
            qs = self.q[s, :]

        epsilon = self.epsilon + max(0, (self.annealing_time - self.step_number)/self.annealing_time)*(1 - self.epsilon)
        if np.random.rand() < epsilon:
            a = np.random.choice(np.argwhere(safe_actions).flatten())
        else:
            a = np.argmax(qs)
        action = self.actions[a]
        
        self.prev_state = None if state.is_terminal() else state
        self.prev_action = action
        
        self.step_number += 1
        
        return action
    
    def update(self, state, action, reward, next_state):
        # If this is the first state, just return.
        if state is None:
            return
        
        self.transition_table.insert(state, action, reward, next_state)
        
        if self.step_number % self.update_frequency == 0:
            transition_matrix, reward_matrix, terminal_states, eps_T = self.transition_table.prepare_for_vi(
                tau=self.tau)
            
            action_mask = None
            if self.safe:
                self.z_safe = self.calculate_z_safe(transition_matrix, eps_T, terminal_states)
                action_mask = self.z_safe
            else:
                # ensure that the agent learns the safe-optimal policy
                reward_matrix = np.where(reward_matrix < 0, -100, reward_matrix)
            
            _, self.q = value_iteration.value_iteration(transition_matrix, reward_matrix, terminal_states,
                                                           horizon=self.vi_horizon,
                                                           gamma=self.gamma,
                                                           action_mask=action_mask,
                                                           q_default=-100,
                                                           use_sparse_matrices=True)
