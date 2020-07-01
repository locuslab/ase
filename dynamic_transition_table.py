import numpy as np
import cvxpy as cp

import value_iteration


class TransitionTable:
    def __init__(self, actions, num_states=None, predefined_rewards=False, reward_func=None, states=None):
        self.sa_counts = dict()
        self.transition_counts = dict()
        self.reward_sums = dict()

        self.terminal_states = set()  # stores all terminal states
        self.actions_for_state = dict()  # all actions for state
        self.transitions = dict()  # (state_prime, probability, reward, terminal_probability) pairs indexed by (s, a)

        self.actions = actions
        self.action_to_id = dict()
        for i, a in enumerate(self.actions):
            self.action_to_id[a] = i

        self.predefined_rewards = predefined_rewards
        if states is None and predefined_rewards:
            raise ValueError('Predefined rewards needs a set of all states')
        elif predefined_rewards:
            self.states = states
            self.num_states = len(states)
            self.rewards = np.zeros([self.num_states, len(self.actions)])
            for s, state in enumerate(self.states):
                for a, action in enumerate(self.actions):
                    self.rewards[s, a] = reward_func(state, action)
                if state.is_terminal():
                    self.terminal_states.add(state)
        else:
            self.num_states = num_states
            self.states = set()  # stores all possible states

    def insert(self, state, action, reward, next_state, terminal):
        if self.num_states is None or self.num_states != len(self.states):
            self.states.add(state)
            self.states.add(next_state)
            if terminal:
                self.terminal_states.add(next_state)

        sa_key = (state, action)

        if sa_key not in self.sa_counts:
            self.reward_sums[sa_key] = 0
            self.sa_counts[sa_key] = 0
            self.transition_counts[sa_key] = dict()

        if next_state not in self.transition_counts[sa_key]:
            self.transition_counts[sa_key][next_state] = 0

        self.reward_sums[sa_key] += reward
        self.sa_counts[sa_key] += 1
        self.transition_counts[sa_key][next_state] += 1

    def prepare_for_vi(self):
        num_states = self.num_states if self.num_states is not None else len(self.states)

        transition_matrix = np.zeros([num_states, len(self.actions), num_states])
        if self.predefined_rewards:
            reward_matrix = self.rewards
        else:
            reward_matrix = np.zeros([num_states, len(self.actions)])
        terminal_matrix = np.zeros([num_states])

        self.state_to_id = dict()
        for s_id, state in enumerate(self.states):
            self.state_to_id[state] = s_id

        state_list = list(self.states)
        for s_id in range(num_states):
            if s_id < len(self.states) and state_list[s_id] in self.terminal_states:
                terminal_matrix[s_id] = 1
            else:
                for a_id, action in enumerate(self.actions):
                    if s_id < len(self.states):
                        key = (state_list[s_id], action)
                    else:
                        key = None
                    if key in self.sa_counts:
                        if not self.predefined_rewards:
                            reward_matrix[s_id, a_id] = self.reward_sums[key] / self.sa_counts[key]

                        for sp in self.transition_counts[key].keys():
                            transition_matrix[s_id, a_id, self.state_to_id[sp]] = self.transition_counts[key][sp] / self.sa_counts[key]
                    else:
                        transition_matrix[s_id, a_id, :] = np.ones(num_states) / num_states

        return transition_matrix, reward_matrix, terminal_matrix

    def id_for_state(self, state):
        return self.state_to_id[state]
