# Python imports.
import numpy as np

# Other imports.=
from simple_rl.agents.AgentClass import Agent

from constants import *
from ase_transition_table import AnalogousStateTransitionTable
import value_iteration


class MBIEAgent(Agent):
    
    def __init__(self, actions, states, reward_func, initial_safe_states, initial_safe_actions,
                 similarity_function, analagous_state_function,
                 transition_support_function=None,
                 epsilon_s=0.01, gamma=0.99, vi_horizon=100, name='mbie-agent', beta_T=0.5,
                 tau=0.1, update_frequency=100,
                 use_sparse_matrices=False):
        
        self.use_sparse_matrices = use_sparse_matrices
        self.gamma = gamma
        self.vi_horizon = vi_horizon
        self.epsilon_s = epsilon_s
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

        self.q = np.zeros([self.num_states, self.num_actions], dtype=DTYPE)
        
        Agent.__init__(self, name=name, actions=actions, gamma=gamma)
    
    def act(self, state, reward, learning=True):
        s = self.state_to_id[state]
        if self.s0 is None:
            self.s0 = s
        
        if learning:
            self.update(self.prev_state, self.prev_action, reward, state)
        
        if reward < 0:
            print('FAILURE: safe agent hit unsafe state')
        
        qs = self.q[s, :]
        a = np.argmax(qs)
        action = self.actions[a]
        
        self.prev_state = None if state.is_terminal() else state
        self.prev_action = action
        
        return action
    
    # ---------------------------------
    # ---- Q VALUES AND PARAMETERS ----
    # ---------------------------------
    
    def update(self, state, action, reward, next_state):
        """
        Args:
            state (simple_rl)
            action (str)
            reward (float)
            next_state (simple_rl)

        Summary:
            Updates the internal Q Function according to the Bellman Equation. (Classic Q Learning update)
        """
        # If this is the first state, just return.
        if state is None:
            return
        
        self.transition_table.insert(state, action, reward, next_state)
        
        if self.step_number % self.update_frequency == 0:
            transition_matrix, reward_matrix, terminal_states, eps_T = self.transition_table.prepare_for_vi(tau=self.tau)
            
            # ensure that the agent learns the safe-optimal policy
            reward_matrix = np.where(reward_matrix < 0, -100, reward_matrix)
            
            _, self.q, _ = value_iteration.optimistic_value_iteration(transition_matrix, reward_matrix, terminal_states,
                                                                      eps_R=0, eps_T=eps_T,
                                                                      gamma=self.gamma,
                                                                      horizon=self.vi_horizon,
                                                                      pessimistic=False,
                                                                      support_mask=self.transition_table.support_mask,
                                                                      use_sparse_matrices=self.use_sparse_matrices)
        
        self.step_number += 1
