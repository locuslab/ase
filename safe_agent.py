# Python imports.
import numpy as np

# Other imports.
from simple_rl.agents.AgentClass import Agent

from constants import *
from ase_transition_table import AnalogousStateTransitionTable
import value_iteration


class SafeAgent(Agent):
    
    def __init__(self, actions, states, reward_func, initial_safe_states, initial_safe_actions,
                 similarity_function, analagous_state_function,
                 transition_support_function=None,
                 gamma=0.99, vi_horizon=100, name='safe-agent', beta_T=0.5,
                 tau=0.1, update_frequency=100,
                 use_sparse_matrices=False):
        
        self.use_sparse_matrices = use_sparse_matrices
        self.gamma = gamma
        self.vi_horizon = vi_horizon
        self.beta_T = beta_T
        self.tau = tau
        self.update_frequency = update_frequency
        
        self.states = states
        self.num_states = len(states)
        self.actions = actions
        self.num_actions = len(actions)
        self.s0 = None
        
        self.step_number = 0
        
        self.state_to_id = dict()
        for s_id, state in enumerate(self.states):
            self.state_to_id[state] = s_id
        self.action_to_id = dict()
        for i, a in enumerate(self.actions):
            self.action_to_id[a] = i
        self.initial_safe_sa = np.zeros([self.num_states, self.num_actions], dtype=np.bool)
        for state in initial_safe_states:
            s = self.state_to_id[state]
            for action in initial_safe_actions(state):
                a = self.action_to_id[action]
                self.initial_safe_sa[s, a] = 1
        
        self.transition_table = AnalogousStateTransitionTable(actions=actions,
                                                              similarity_function=similarity_function,
                                                              analagous_state_function=analagous_state_function,
                                                              initial_safe_sa=self.initial_safe_sa,
                                                              reward_func=reward_func, states=states,
                                                              support_function=transition_support_function,
                                                              use_sparse_matrices=self.use_sparse_matrices,
                                                              beta_T=beta_T)
        
        self.z_safe = self.initial_safe_sa
        
        Agent.__init__(self, name=name, actions=actions, gamma=gamma)

    def act(self, state, reward, learning=True):
        s = self.state_to_id[state]
        if self.s0 is None:
            self.s0 = s
    
        if learning:
            self.update(self.prev_state, self.prev_action, reward, state)
        
        if reward < 0:
            print('FAILURE: safe agent hit unsafe state')
    
        safe_actions = self.z_safe[s, :]
        if safe_actions.sum() == 0:
            print('UNSAFE: safe policy is not defined for state %s' % state)
        
        a = np.random.choice(safe_actions)
        action = self.actions[a]
    
        self.prev_state = None if state.is_terminal() else state
        self.prev_action = action
    
        return action

    # ---------------------------------
    # ---- Q VALUES AND PARAMETERS ----
    # ---------------------------------
    def update(self, state, action, reward, next_state):
        # If this is the first state, just return.
        if state is None:
            return
    
        self.transition_table.insert(state, action, reward, next_state)
    
        if self.step_number % 100 == 0:
            transition_matrix, reward_matrix, terminal_states, eps_T = self.transition_table.prepare_for_vi(tau=self.tau)
        
            self.z_safe = self.calculate_z_safe(transition_matrix, eps_T, terminal_states)
    
        self.step_number += 1
    
    def calculate_z_safe(self, transition_matrix, eps_T, terminal_states):
        T_indptr = transition_matrix.indptr
        T_indices = transition_matrix.indices
        T_data = transition_matrix.data
        
        z_candidate = np.logical_and(eps_T < self.tau, self.z_safe == 0)
        goals = np.logical_and(terminal_states, (self.transition_table.rewards >= 0).all(axis=1))
        failures = np.logical_and(terminal_states, (self.transition_table.rewards < 0).any(axis=1))
        
        while True:
            # Add all reachable states to z_reach
            z_reach = np.where(np.max(self.z_safe, axis=1)[:, np.newaxis], z_candidate, 0)
            changed = True
            while changed:
                changed = False
                z_candidate_next = np.logical_and(z_candidate, 1 - z_reach)
                for s in range(self.num_states):
                    for a in range(self.num_actions):
                        if self.z_safe[s, a] or z_reach[s, a]:
                            sa = np.ravel_multi_index([s, a], [self.num_states, self.num_actions])
                            p = T_indptr[sa]
                            q = T_indptr[sa + 1]
                            sp = T_indices[p:q]
                            can_be_reached = np.logical_and(T_data[p:q] > 0, 1 - failures[sp])[:, np.newaxis]
                            new_z_reachable = np.logical_and(can_be_reached,
                                                             z_candidate_next[sp, :])
                            # check if this will change z_reach[sp, :]
                            if (np.logical_or(z_reach[sp, :], new_z_reachable) != z_reach[sp, :]).any():
                                z_reach[sp, :] = np.logical_or(z_reach[sp, :], new_z_reachable)
                                changed = True
            
            # Add all returnable states from z_reach to z_ret
            z_ret = np.where(goals[:, np.newaxis], 1, np.zeros_like(self.z_safe))
            changed = True
            while changed:
                changed = False
                for s in range(self.num_states):
                    for a in range(self.num_actions):
                        if z_reach[s, a] and z_ret[s, a] == 0:
                            sa = np.ravel_multi_index([s, a], [self.num_states, self.num_actions])
                            p = T_indptr[sa]
                            q = T_indptr[sa + 1]
                            sp = T_indices[p:q]
                            can_be_reached = np.logical_and(T_data[p:q] > 0, 1 - failures[sp])[:, np.newaxis]
                            if np.logical_and(can_be_reached,
                                              np.logical_or(self.z_safe[sp, :], z_ret[sp, :])).max() > 0:
                                z_ret[s, a] = 1
                                changed = True
                                break
            
            # Remove all state-action pairs that have support outside of z_ret (z_closed)
            changed = True
            while changed:
                changed = False
                for s in range(self.num_states):
                    if goals[s]:
                        continue
                    for a in range(self.num_actions):
                        if z_ret[s, a]:
                            sa = np.ravel_multi_index([s, a], [self.num_states, self.num_actions])
                            p = T_indptr[sa]
                            q = T_indptr[sa + 1]
                            for i in np.argwhere(T_data[p:q] > 0).flatten():
                                sp = T_indices[p + i]
                                if failures[sp] or not (goals[sp] or np.logical_or(self.z_safe[sp, :], z_ret[sp, :]).any()):
                                    z_ret[s, a] = 0
                                    changed = True
                                    break
            
            if np.all(z_candidate == z_ret):
                break
            else:
                z_candidate = z_ret.copy()
        
        return np.logical_or(z_ret, self.z_safe)
