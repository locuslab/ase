''' QLearningAgentClass.py: Class for a basic QLearningAgent '''

# Python imports.
import random
import numpy as np

# Other imports.
from simple_rl.agents.AgentClass import Agent

from dynamic_programming.transition_table import TransitionTable
from dynamic_programming import value_iteration


class UnsafeAgent(Agent):
    ''' Implementation for a Q Learning Agent '''

    def __init__(self, states, actions, epsilon=0.1, gamma=0.99, vi_horizon=100, name='unsafe-agent'):

        self.transition_table = TransitionTable(states, actions)
        self.epsilon = epsilon
        self.gamma = gamma
        self.vi_horizon = vi_horizon

        Agent.__init__(self, name=name, actions=actions, gamma=gamma)

    # --------------------------------
    # ---- CENTRAL ACTION METHODS ----
    # --------------------------------

    def act(self, state, reward, learning=True):
        '''
        Args:
            state (simple_rl)
            reward (float)

        Returns:
            (str)

        Summary:
            The central method called during each time step.
            Retrieves the action according to the current policy
            and performs updates given (s=self.prev_state,
            a=self.prev_action, r=reward, s'=state)
        '''

        if learning:
            self.update(self.prev_state, self.prev_action, reward, state)

        transition_matrix, reward_matrix, terminal_matrix = self.transition_table.prepare_for_vi()
        values, q_values = value_iteration.value_iteration(transition_matrix, reward_matrix, terminal_matrix,
                                                           gamma=self.gamma, horizon=self.vi_horizon)
        q_values = q_values[self.transition_table.state_to_id[state], :]

        if q_values is None or random.random() < self.epsilon:
            action = random.choice(self.actions)
        else:
            a_id = random.choice(np.sum(np.argwhere(q_values == np.max(q_values)), axis=1))
            action = self.actions[a_id]

        self.prev_state = state
        self.prev_action = action

        return action

    # ---------------------------------
    # ---- Q VALUES AND PARAMETERS ----
    # ---------------------------------

    def update(self, state, action, reward, next_state):
        '''
        Args:
            state (simple_rl)
            action (str)
            reward (float)
            next_state (simple_rl)

        Summary:
            Updates the internal Q Function according to the Bellman Equation. (Classic Q Learning update)
        '''
        # If this is the first state, just return.
        if state is None:
            return

        self.transition_table.insert(state, action, reward, next_state)
