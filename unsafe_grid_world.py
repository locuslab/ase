import random
import numpy as np
import os
from scipy.sparse import csr_matrix, lil_matrix
import pygame

import mdp_visualizer
from simple_rl.tasks import GridWorldMDP
from simple_rl.tasks.grid_world.GridWorldStateClass import GridWorldState
from simple_rl.tasks.grid_world.grid_visualizer import _draw_state as draw_state

SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 1000


class UnsafeGridWorldMDP(GridWorldMDP):

    # Static constants.
    ACTIONS = ["up", "down", "left", "right", "jump up", "jump down", "jump left", "jump right"]

    def __init__(self,
                 width=5,
                 height=3,
                 init_loc=(1, 1),
                 rand_init=False,
                 goal_locs=[(5, 3)],
                 lava_locs=[()],
                 walls=[],
                 safe_locs=[],
                 is_goal_terminal=True,
                 gamma=0.99,
                 slip_prob=0.0,
                 step_cost=0.0,
                 lava_cost=1,
                 name="gridworld",
                 gui=False):

        GridWorldMDP.__init__(self, width, height, init_loc, rand_init, goal_locs, lava_locs, walls, is_goal_terminal, gamma, slip_prob, step_cost, lava_cost, name)

        self.jump_dist = 2

        self.actions = UnsafeGridWorldMDP.ACTIONS
        self.safe_states = set()
        for state in self.get_all_states():
            if (state.x, state.y) in safe_locs:
                self.safe_states.add(state)
        
        self.gui = gui
        if gui:
            self.screen = pygame.display.set_mode((SCREEN_HEIGHT, SCREEN_HEIGHT))
            self.agent_shape = None

            # Pygame setup.
            pygame.init()
            self.screen.fill((255, 255, 255))
            pygame.display.update()
            
            self.agent_shape = self._draw_state(self.init_state, draw_statics=True)

    def _reward_func(self, state, action):
        '''
        Args:
            state (simple_rl)
            action (str)

        Returns
            (float)
        '''

        # next_state = self._transition_func(state, action)
        if (int(state.x), int(state.y)) in self.goal_locs:
            return 1.0 - self.step_cost
        elif (int(state.x), int(state.y)) in self.lava_locs:
            return -self.lava_cost - self.step_cost
        else:
            return 0 - self.step_cost
    
    def reward_function(self, state, action): return self._reward_func(state, action)
    
    def transition_function(self, state, action): return self.transition_func(state, action)
    
    def get_current_state(self): return self.get_curr_state()

    def _transition_func(self, state, action):
        '''
        Args:
            state (simple_rl)
            action (str)

        Returns
            (State)
        '''
        if state.is_terminal():
            return [state], [1]

        dx = [0, 0, 0]
        dy = [0, 0, 0]
        if action == "up":
            dx = [-1, 0, 1]
            dy = [1, 1, 1]
        elif action == "down":
            dx = [-1, 0, 1]
            dy = [-1, -1, -1]
        elif action == "right":
            dx = [1, 1, 1]
            dy = [-1, 0, 1]
        elif action == "left":
            dx = [-1, -1, -1]
            dy = [-1, 0, 1]
        elif action == "jump up":
            dx = [-1, 0, 1]
            dy = [2, 2, 2]
        elif action == "jump down":
            dx = [-1, 0, 1]
            dy = [-2, -2, -2]
        elif action == "jump right":
            dx = [2, 2, 2]
            dy = [-1, 0, 1]
        elif action == "jump left":
            dx = [-2, -2, -2]
            dy = [-1, 0, 1]
        
        next_states = []
        for delta_x, delta_y in zip(dx, dy):
            x = np.clip(state.x + delta_x, 1, self.width)
            y = np.clip(state.y + delta_y, 1, self.height)
            if self.is_wall(x, y):
                next_state = GridWorldState(state.x, state.y)
            else:
                next_state = GridWorldState(x, y)
            next_state.set_terminal(self._terminal_function(next_state))
            next_states.append(next_state)

        p = [self.slip_prob / 2., 1 - self.slip_prob, self.slip_prob / 2.]
        assert len(next_states) == len(p)
        return next_states, p

    def _terminal_function(self, state):
        if ((state.x, state.y) in self.goal_locs and self.is_goal_terminal) or \
                (state.x, state.y) in self.lava_locs:
            return True
        else:
            return False

    def get_all_states(self):
        states = set()
        for x in range(1, self.width+1):
            for y in range(1, self.height+1):
                state = GridWorldState(x, y)
                state.set_terminal(self._terminal_function(state))
                states.add(state)
        return states

    def safe_actions(self, state):
        actions = []
        for a in self.get_actions():
            safe = True
            next_states, p = self._transition_func(state, a)
            for next_state in next_states:
                if next_state not in self.safe_states:
                    safe = False
                    break
            if safe:
                actions.append(a)
        assert len(actions) > 0
        return actions

    def get_safe_states(self):
        return self.safe_states

    def get_safe_actions_function(self):
        return lambda state: self.safe_actions(state)
    
    def execute_agent_action(self, action):
        next_states, p = self._transition_func(self.cur_state, action)
        i = np.random.choice(len(next_states), p=p)
        self.cur_state = next_states[i]
        reward = self._reward_func(self.cur_state, action)

        # Update GUI
        if self.gui:
            self.update_gui()

        return reward, self.cur_state

    def reset(self):
        self.cur_state = self.init_state

        # Update GUI
        if self.gui:
            self.update_gui(0)
    
    def update_gui(self):
        cur_state = self.get_curr_state()
        cur_state = self.get_init_state() if cur_state is None else cur_state
        self.agent_shape = self._draw_state(cur_state, draw_statics=True,
                                            agent_shape=self.agent_shape)
        
        pygame.display.update()

    def _draw_state(self,
                    state,
                    agent=None,
                    draw_statics=False,
                    agent_shape=None):
    
        # Prep some dimensions to make drawing easier.
        scr_width, scr_height = self.screen.get_width(), self.screen.get_height()
        width_buffer = 0 #  scr_width / 10.0
        height_buffer = 0 #  30 + (scr_height / 10.0)  # Add 30 for title.
        cell_width = (scr_width - width_buffer * 2) / self.width
        cell_height = (scr_height - height_buffer * 2) / self.height
        goal_locs = self.get_goal_locs()
        lava_locs = self.get_lava_locs()
        font_size = int(min(cell_width, cell_height) / 4.0)
        reg_font = pygame.font.SysFont("CMU Serif", font_size)
        cc_font = pygame.font.SysFont("Courier", font_size * 2 + 2)
    
        # Draw the static entities.
        if draw_statics:
            # For each row:
            for i in range(self.width):
                # For each column:
                for j in range(self.height):
                
                    top_left_point = width_buffer + cell_width * i, height_buffer + cell_height * j
                    r = pygame.draw.rect(self.screen, (46, 49, 49), top_left_point + (cell_width, cell_height), 3)
                    
                    if self.is_wall(i + 1, self.height - j):
                        # Draw the walls.
                        top_left_point = width_buffer + cell_width * i + 5, height_buffer + cell_height * j + 5
                        r = pygame.draw.rect(self.screen, (94, 99, 99), top_left_point + (cell_width - 10, cell_height - 10),
                                             0)
                
                    if (i + 1, self.height - j) in goal_locs:
                        # Draw goal.
                        circle_center = int(top_left_point[0] + cell_width / 2.0), int(
                            top_left_point[1] + cell_height / 2.0)
                        circler_color = (154, 195, 157)
                        pygame.draw.circle(self.screen, circler_color, circle_center,
                                           int(min(cell_width, cell_height) / 2.5))
                
                    if (i + 1, self.height - j) in lava_locs:
                        # Draw lava.
                        circle_center = int(top_left_point[0] + cell_width / 2.0), int(
                            top_left_point[1] + cell_height / 2.0)
                        circler_color = (224, 145, 157)
                        pygame.draw.circle(self.screen, circler_color, circle_center,
                                           int(min(cell_width, cell_height) / 3.0))
                
                    # Current state.
                    if (i + 1, self.height - j) == (state.x, state.y) and agent_shape is None:
                        tri_center = int(top_left_point[0] + cell_width / 2.0), int(
                            top_left_point[1] + cell_height / 2.0)
                        agent_shape = self._draw_agent(tri_center, base_size=min(cell_width, cell_height) / 2.5 - 4)
    
        if agent_shape is not None:
            # Clear the old shape.
            pygame.draw.rect(self.screen, (255, 255, 255), agent_shape)
            top_left_point = width_buffer + cell_width * (state.x - 1), height_buffer + cell_height * (
                        self.height - state.y)
            tri_center = int(top_left_point[0] + cell_width / 2.0), int(top_left_point[1] + cell_height / 2.0)

            # Draw new.
            agent_shape = self._draw_agent(tri_center, base_size=min(cell_width, cell_height) / 2.5 - 4)
    
        pygame.display.flip()
    
        return agent_shape

    def _draw_agent(self, center_point, base_size=20):
        '''
        Args:
            center_point (tuple): (x,y)
            screen (pygame.Surface)

        Returns:
            (pygame.rect)
        '''
        tri_bot_left = center_point[0] - base_size, center_point[1] + base_size
        tri_bot_right = center_point[0] + base_size, center_point[1] + base_size
        tri_top = center_point[0], center_point[1] - base_size
        tri = [tri_bot_left, tri_top, tri_bot_right]
        tri_color = (98, 140, 190)
        return pygame.draw.polygon(self.screen, tri_color, tri)

    def map_image(self):
        imgdata = pygame.surfarray.array3d(self.screen)
        imgdata = imgdata.transpose([1, 0, 2])
        return imgdata

    def location_invariance(self, state_list, action_list, distance=np.infty):
        num_states = len(state_list)
        num_actions = len(action_list)
    
        xs = np.array([state.x for state in state_list])
        ys = np.array([state.y for state in state_list])
    
        similarity_matrix = lil_matrix((num_states * num_actions, num_states * num_actions))
        
        for s, state in enumerate(state_list):
            similarity = np.logical_and(np.abs(state.x - xs) <= distance, np.abs(state.y - ys) <= distance)
            for s2 in np.argwhere(similarity > 0).flatten():
                for a in range(num_actions):
                    sa = np.ravel_multi_index([s, a], [num_states, num_actions])
                    sa2 = np.ravel_multi_index([s2, a], [num_states, num_actions])
                    similarity_matrix[sa, sa2] = similarity[s2]
    
        return similarity_matrix.tocsr()

    def location_invariance_equivalency(self, state1, action1, state_prime1, state2, action2):
        state_prime2 = None
        if action1 == action2:
            x_diff = state_prime1.x - state1.x
            y_diff = state_prime1.y - state1.y
            
            x = state2.x + x_diff
            y = state2.y + y_diff
            state_prime2 = GridWorldState(x, y)
            state_prime2.set_terminal(self._terminal_function(state_prime2))
    
        return state_prime2


def transition_support_function(state_list, action_list, max_x_distance=3, max_y_distance=3):
    num_states = len(state_list)
    num_actions = len(action_list)

    xs = np.array([state.x for state in state_list])
    ys = np.array([state.y for state in state_list])

    support_matrix = lil_matrix((num_states * num_actions, num_states), dtype=np.bool)

    for s, state in enumerate(state_list):
        support = csr_matrix(
            np.logical_and(np.abs(state.x - xs) < max_x_distance, np.abs(state.y - ys) < max_y_distance))
        i_start = np.ravel_multi_index([s, 0], [num_states, num_actions])
        i_end = np.ravel_multi_index([s, num_actions - 1], [num_states, num_actions])
        for i in range(i_start, i_end + 1):
            support_matrix[i, :] = support

    return support_matrix.tocsr()


def make_grid_world_from_file(file_name, randomize=False, num_goals=1, name=None, goal_num=None, slip_prob=0.0, gui=False):
    '''
    Args:
        file_name (str)
        randomize (bool): If true, chooses a random agent location and goal location.
        num_goals (int)
        name (str)

    Returns:
        (GridWorldMDP)

    Summary:
        Builds a GridWorldMDP from a file:
            'w' --> wall
            'a' --> agent
            'g' --> goal
            '-' --> empty
    '''

    if name is None:
        name = file_name.split(".")[0]

    # grid_path = os.path.dirname(os.path.realpath(__file__))
    wall_file = open(os.path.join(os.getcwd(), file_name))
    wall_lines = wall_file.readlines()

    # Get walls, agent, goal loc.
    num_rows = len(wall_lines)
    num_cols = len(wall_lines[0].strip())
    empty_cells = []
    agent_x, agent_y = 1, 1
    walls = []
    goal_locs = []
    lava_locs = []
    safe_locs = []

    for i, line in enumerate(wall_lines):
        line = line.strip()
        for j, ch in enumerate(line):
            if ch == "w":
                walls.append((j + 1, num_rows - i))
            elif ch == "g":
                goal_locs.append((j + 1, num_rows - i))
            elif ch == "l":
                lava_locs.append((j + 1, num_rows - i))
            elif ch == "a":
                agent_x, agent_y = j + 1, num_rows - i
                safe_locs.append((j + 1, num_rows - i))
            elif ch == "-":
                empty_cells.append((j + 1, num_rows - i))
            elif ch == "s":
                empty_cells.append((j + 1, num_rows - i))
                safe_locs.append((j + 1, num_rows - i))
            else:
                raise ValueError("Unrecognized character %s in map" % (ch))

    if goal_num is not None:
        goal_locs = [goal_locs[goal_num % len(goal_locs)]]

    if randomize:
        agent_x, agent_y = random.choice(empty_cells)
        if len(goal_locs) == 0:
            # Sample @num_goals random goal locations.
            goal_locs = random.sample(empty_cells, num_goals)
        else:
            goal_locs = random.sample(goal_locs, num_goals)

    if len(goal_locs) == 0:
        goal_locs = [(num_cols, num_rows)]

    return UnsafeGridWorldMDP(width=num_cols, height=num_rows, init_loc=(agent_x, agent_y), goal_locs=goal_locs,
                              lava_locs=lava_locs, walls=walls, name=name, slip_prob=slip_prob, safe_locs=safe_locs,
                              gui=gui)
