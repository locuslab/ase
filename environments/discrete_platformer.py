"""
Sample Python/Pygame Programs
Simpson College Computer Science
http://programarcadegames.com/
http://simpson.edu/computer-science/

From:
http://programarcadegames.com/python_examples/f.php?file=platform_scroller.py

Explanation video: http://youtu.be/QplXBw_NK5Y

Part of a series:
http://programarcadegames.com/python_examples/f.php?file=move_with_walls_example.py
http://programarcadegames.com/python_examples/f.php?file=maze_runner.py
http://programarcadegames.com/python_examples/f.php?file=platform_jumper.py
http://programarcadegames.com/python_examples/f.php?file=platform_scroller.py
http://programarcadegames.com/python_examples/f.php?file=platform_moving.py
http://programarcadegames.com/python_examples/sprite_sheets/

"""

import pygame
import os
import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix, lil_matrix

from simple_rl.mdp.StateClass import State

# Global constants

# Colors
IMAGE_DIR = 'images'
PLAYER_IMAGE = 'player.png'
GOAL_IMAGE = 'goal.png'
PLATFORM_IMAGE_1 = 'ice.png'
PLATFORM_IMAGE_2 = 'concrete.png'
PLATFORM_IMAGE_3 = 'sand.png'
PLATFORM_COLOR = (30, 30, 30)
PLATFORM_COLOR_2 = (20, 20, 60)
PLATFORM_COLOR_3 = (60, 20, 20)
PLAYER_COLOR = (255, 140, 0)
BG_COLOR = (255, 255, 255)
GOAL_COLOR = (200, 255, 200)

# GUI constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
LEFT_SHIFT_LIMIT = 120
RIGHT_SHIFT_LIMIT = 500
TILE_WIDTH = 20
TILE_HEIGHT = 20
PLAYER_WIDTH = 1
PLAYER_HEIGHT = 1

# Game constants
PLAYER_ACC = 1
PLAYER_MAX_SPEED = 2
JUMP_IMPULSE = 2
JUMP_STD = 1
GRAVITY = 1
TERMINAL_VEL = JUMP_IMPULSE

CAN_MOVE_IN_AIR = False
STOPS_ON_EDGES = False


# Actions
LEFT = (-PLAYER_MAX_SPEED, 0)
RIGHT = (PLAYER_MAX_SPEED, 0)
STOP = (0, 0)
JUMP_LEFT = (-PLAYER_MAX_SPEED, 1)
JUMP_RIGHT = (PLAYER_MAX_SPEED, 1)
JUMP = (0, 1)
ACTIONS = []
for x_dot_desired in range(-PLAYER_MAX_SPEED, PLAYER_MAX_SPEED+1):
    for jump in [0, 1]:
        ACTIONS.append((x_dot_desired, jump))


class PlatformerState(State):
    def __init__(self, x, y, x_dot, y_dot, on_platform_type, is_terminal=False):
        self.x = x
        self.y = y
        self.x_dot = x_dot
        self.y_dot = y_dot
        self.on_platform_type = on_platform_type

        State.__init__(self, data=[self.x, self.y, self.x_dot, self.y_dot, self.on_platform_type], is_terminal=is_terminal)

    def __str__(self):
        return 's: %s, %s' % ((self.x, self.y), (self.x_dot, self.y_dot))

    def __hash__(self):
        return hash(tuple(self.data))

    def __eq__(self, other):
        return isinstance(other, PlatformerState) and \
               self.x == other.x and \
               self.y == other.y and \
               self.x_dot == other.x_dot and \
               self.y_dot == other.y_dot


class DiscreteSprite(pygame.sprite.Sprite):
    def __init__(self, x, y, image, width=1, height=1):
        super().__init__()

        self.x = x
        self.y = y

        self.image = pygame.Surface([width*TILE_WIDTH, height*TILE_HEIGHT])
        self.image = pygame.image.load(os.path.join(os.getcwd(), IMAGE_DIR, image))
        # self.image.fill(color)

        self.rect = self.image.get_rect()
        self.update_rect(0)

    def update_rect(self, shift_x, screen_height=SCREEN_HEIGHT):
        self.rect.x = (self.x + 1) * TILE_WIDTH - shift_x
        self.rect.y = screen_height - (self.y + 1) * TILE_HEIGHT


class PlatformerGUI():
    def __init__(self, state, platform_locs, platform_types, goal_loc):
        pygame.init()

        # Set the height and width of the screen
        size = [SCREEN_WIDTH, SCREEN_HEIGHT]
        self.screen = pygame.display.set_mode(size)

        pygame.display.set_caption("Discrete Platformer")

        self.player = DiscreteSprite(state.x, state.y, PLAYER_IMAGE)
        # colors = [PLATFORM_COLOR, PLATFORM_COLOR_2, PLATFORM_COLOR_3]
        images = [PLATFORM_IMAGE_1, PLATFORM_IMAGE_2, PLATFORM_IMAGE_3]
        self.platforms = [DiscreteSprite(loc[0], loc[1], images[int(platform_type) - 1]) for loc, platform_type in zip(platform_locs, platform_types)]
        self.goal = DiscreteSprite(goal_loc[0], goal_loc[1], GOAL_IMAGE)

        self.player_sprite_group = pygame.sprite.Group()
        self.player_sprite_group.add(self.player)

        self.platform_list = pygame.sprite.Group()
        for platform in self.platforms:
            self.platform_list.add(platform)

        self.goal_sprite_group = pygame.sprite.Group()
        self.goal_sprite_group.add(self.goal)

        self.world_shift = 0
        self.update(state)

    def update(self, state):
        self.player.x = state.x
        self.player.y = state.y

        # If the player gets near the right side, shift the world left (-x)
        if self.player.rect.right >= RIGHT_SHIFT_LIMIT:
            diff = self.player.rect.right - RIGHT_SHIFT_LIMIT
            self.world_shift += diff
        # If the player gets near the left side, shift the world right (+x)
        elif self.player.rect.left <= LEFT_SHIFT_LIMIT:
            diff = self.player.rect.left - LEFT_SHIFT_LIMIT
            self.world_shift += diff

        self.player.update_rect(self.world_shift)
        self.goal.update_rect(self.world_shift)
        for platform in self.platforms:
            platform.update_rect(self.world_shift)

        self.draw()
        pygame.display.flip()

    def draw(self):
        # Draw the background
        self.screen.fill(BG_COLOR)

        # Draw all the sprite lists that we have
        self.player_sprite_group.draw(self.screen)
        self.platform_list.draw(self.screen)
        self.goal_sprite_group.draw(self.screen)
    
    def map_image(self, min_x, max_x, min_y, max_y):
        width = max_x - min_x + 1
        height = max_y - min_y + 1
        surf = pygame.Surface((width * TILE_WIDTH, height * TILE_HEIGHT))

        world_shift = min_x * TILE_WIDTH
        self.player.update_rect(world_shift, screen_height=height * TILE_HEIGHT)
        self.goal.update_rect(world_shift, screen_height=height * TILE_HEIGHT)
        for platform in self.platforms:
            platform.update_rect(world_shift, screen_height=height * TILE_HEIGHT)
        
        surf.fill(BG_COLOR)
        self.player_sprite_group.draw(surf)
        self.platform_list.draw(surf)
        self.goal_sprite_group.draw(surf)

        imgdata = pygame.surfarray.array3d(surf)
        imgdata = imgdata.transpose([1, 0, 2])
        return imgdata


class DiscretePlatformer:
    def __init__(self, platform_grid, init_loc, goal_loc, gui=True):
        self.platform_grid = platform_grid
        init_on_platform_type = self.platform_grid[init_loc[0], init_loc[1] - 1]
        self.init_state = PlatformerState(init_loc[0], init_loc[1], 0, 0, init_on_platform_type)
        self.state = self.init_state
        self.goal_loc = goal_loc

        self.y_min = 1
        self.y_max = int(self.y_min + np.sum(np.arange(JUMP_IMPULSE, 0, -GRAVITY)))

        self.platform_locs = []
        self.platform_types = []
        for x in range(self.platform_grid.shape[0]):
            for y in range(self.platform_grid.shape[1]):
                if self.platform_grid[x, y] > 0:
                    self.platform_locs.append([x, y])
                    self.platform_types.append(self.platform_grid[x, y])

        if gui:
            self.gui = PlatformerGUI(self.init_state, self.platform_locs, self.platform_types, goal_loc)
        else:
            self.gui = None

    def reached_goal(self, x, y):
        if x == self.goal_loc[0] and y == self.goal_loc[1]:
            return True

    def collision(self, x, y):
        if 0 <= x < self.platform_grid.shape[0] and 0 <= y < self.platform_grid.shape[1]:
            return self.platform_grid[x, y]
        else:
            return 0

    def on_platform(self, x, y):
        return self.collision(x, y-1)

    def _get_win_loss(self, x, y):
        # Check win/loss
        terminal = False
        reward = 0
        # Check if fell off
        if y <= self.y_min and self.on_platform(x, y) == 0:
            terminal = True
            reward = -1
        # Check if reached goal
        if self.reached_goal(x, y):
            terminal = True
            reward = 1

        return reward, terminal

    def reward_function(self, state, action):
        reward, terminal = self._get_win_loss(state.x, state.y)
        return reward

    def transition_function(self, state, action):
        if state.is_terminal():
            return [state], [1]

        x, y = state.x, state.y
        x_dot, y_dot = state.x_dot, state.y_dot

        x_dot_desired = action[0]
        jump = action[1]
        
        if self.on_platform(x, y) > 0:
            if jump:
                y_dots = [JUMP_IMPULSE, JUMP_IMPULSE - 1]
                
                if self.on_platform(x, y) == 1:
                    p = [0.5, 0.5]
                elif self.on_platform(x, y) == 2:
                    p = [1, 0]
                elif self.on_platform(x, y) == 3:
                    p = [0.0, 1.0]
                else:
                    p = [0.5, 0.5]
                    ValueError('Platform type not defined')
            else:
                y_dots = [y_dot]
                p = [1]
        else:
            # Apply gravity
            y_dots = [max(-TERMINAL_VEL, y_dot - GRAVITY)]
            p = [1]

        next_states = []
        for y_dot in y_dots:
            x, y = state.x, state.y
            x_dot = state.x_dot
            
            if CAN_MOVE_IN_AIR or self.on_platform(x, y) > 0:
                if x_dot_desired > x_dot:
                    x_dot = min(x_dot_desired, x_dot + PLAYER_ACC)
                else:  # x_dot_desired <= self.x_dot:
                    x_dot = max(x_dot_desired, x_dot - PLAYER_ACC)
    
            # Move up/down
            y += y_dot
            # See if we hit anything
            while self.collision(x, y):
                if y_dot > 0:
                    y -= 1
                else:  # y_dot < 0:
                    y += 1
    
                # Stop our vertical movement
                y_dot = 0
    
            # Move left/right
            x += x_dot
            # See if we hit anything
            while self.collision(x, y):
                if x_dot > 0:
                    x -= 1
                else:  # x_dot < 0:
                    x += 1
    
            # Check if we landed on platform. Needs to be after x movement
            if self.on_platform(x, y) > 0:
                y_dot = 0
    
            reward, terminal = self._get_win_loss(x, y)
            if terminal:
                y_dot = 0
    
            next_state = PlatformerState(x, y, x_dot, y_dot, self.on_platform(x, y), is_terminal=terminal)
            next_states.append(next_state)
        
        assert len(p) == len(next_states)
        assert sum(p) == 1
        return next_states, p

    def execute_agent_action(self, action):
        next_states, p = self.transition_function(self.state, action)
        i = np.random.choice(len(next_states), p=p)
        self.state = next_states[i]
        reward, terminal = self._get_win_loss(self.state.x, self.state.y)

        # Update GUI
        if self.gui is not None:
            self.gui.update(self.state)

        return reward, self.state

    def reset(self):
        self.state = self.init_state

        # Update GUI
        if self.gui is not None:
            self.gui.update(self.state)

    def get_actions(self):
        return ACTIONS

    def get_current_state(self):
        return self.state

    def get_all_states(self):
        states = set()
        total_air_time = JUMP_IMPULSE*2+1
        for x in range(-PLAYER_MAX_SPEED * total_air_time, self.platform_grid.shape[0] + PLAYER_MAX_SPEED * total_air_time):
            for x_dot in range(-PLAYER_MAX_SPEED, PLAYER_MAX_SPEED + 1):
                y = self.y_min
                y_dot = 0
                reward, terminal = self._get_win_loss(x, y)
                state = PlatformerState(x, y, x_dot, y_dot, self.on_platform(x, y), is_terminal=terminal)
                states.add(state)

                # go over every trajectory
                for jump in [JUMP_IMPULSE, JUMP_IMPULSE - 1]:
                    y = self.y_min
                    y_dot = jump
                    y += y_dot
                    while y > self.y_min:
                        reward, terminal = self._get_win_loss(x, y)
                        state = PlatformerState(x, y, x_dot, y_dot, self.on_platform(x, y), is_terminal=terminal)
                        states.add(state)
    
                        y_dot -= GRAVITY
                        y += y_dot

        return states

    def horizontal_invariance(self, state_list, action_list, distance=np.infty):
        platform_similarity = 0.0
        
        num_states = len(state_list)
        num_actions = len(action_list)
    
        xs = np.array([state.x for state in state_list])
        ys = np.array([state.y for state in state_list])
        x_dots = np.array([state.x_dot for state in state_list])
        y_dots = np.array([state.y_dot for state in state_list])
        on_platform_type = np.array([state.on_platform_type for state in state_list])
    
        similarity_matrix = lil_matrix((num_states * num_actions, num_states * num_actions))
    
        for s, state in enumerate(state_list):
            similarity = ((abs(state.x == xs) < distance) * (state.y == ys) * (state.x_dot == x_dots) * (state.y_dot == y_dots)).astype(dtype=np.float)
            same_platform_type = on_platform_type == on_platform_type[s]
            similarity *= same_platform_type + (1 - same_platform_type) * platform_similarity
            for s2 in np.argwhere(similarity > 0).flatten():
                for a in range(num_actions):
                    sa = np.ravel_multi_index([s, a], [num_states, num_actions])
                    sa2 = np.ravel_multi_index([s2, a], [num_states, num_actions])
                    similarity_matrix[sa, sa2] = similarity[s2]
    
        return similarity_matrix.tocsr()

#
# def horizontal_invariance(state1, action1, state2, action2):
#     if (state1.y, state1.x_dot, state1.y_dot) == (state2.y, state2.x_dot, state2.y_dot) \
#             and action1 == action2:
#         return 1
#     else:
#         return 0

    def horizontal_invariance_equivalency(self, state1, action1, state_prime1, state2, action2):
        state_prime2 = None
        if (state1.y, state1.x_dot, state1.y_dot) == (state2.y, state2.x_dot, state2.y_dot) \
                and action1 == action2:
            x_diff = state_prime1.x - state1.x
            x = state2.x + x_diff
            _, terminal = self._get_win_loss(x, state_prime1.y)
            state_prime2 = PlatformerState(x,
                                           state_prime1.y,
                                           state_prime1.x_dot,
                                           state_prime1.y_dot,
                                           self.on_platform(x, state_prime1.y),
                                           is_terminal=terminal)
        
        return state_prime2


def transition_support_function(state_list, action_list, max_x_distance=3, max_y_distance=3):
    num_states = len(state_list)
    num_actions = len(action_list)

    xs = np.array([state.x for state in state_list])
    ys = np.array([state.y for state in state_list])

    support_matrix = lil_matrix((num_states * num_actions, num_states), dtype=np.bool)

    for s, state in enumerate(state_list):
        support = csr_matrix(np.logical_and(np.abs(state.x - xs) < max_x_distance, np.abs(state.y - ys) < max_y_distance))
        i_start = np.ravel_multi_index([s, 0], [num_states, num_actions])
        i_end = np.ravel_multi_index([s, num_actions-1], [num_states, num_actions])
        for i in range(i_start, i_end+1):
            support_matrix[i, :] = support

    return support_matrix.tocsr()


# def transition_support_function(state, action, state_prime, max_x_distance=3, max_y_distance=3):
#     if abs(state.x - state_prime.x) <= max_x_distance and abs(state.y - state_prime.y) <= max_y_distance:
#         return True
#     return False


def env_from_file(file_name, gui=True):
    level_file = open(os.path.join(os.getcwd(), file_name))
    level_lines = level_file.read().splitlines()
    num_lines = len(level_lines)

    platform_grid = np.zeros([len(level_lines[0]), len(level_lines)])
    safe_locs = []

    player_location = None
    goal_location = None
    for i, line in enumerate(level_lines):
        y = num_lines - i - 1
        line = line.strip()
        for j, ch in enumerate(line):
            x = j
            if ch == '1' or ch == '2' or ch == '3':
                platform_grid[x, y] = int(ch)
            elif ch == 'G':
                goal_location = (x, y)
            elif ch == 'a':
                player_location = (x, y)
                safe_locs.append((x, y))
            elif ch == 's':
                safe_locs.append((x, y))
            elif ch != '0':
                raise ValueError('Unrecognized char: %s' % ch)

    return DiscretePlatformer(platform_grid, player_location, goal_location, gui=gui), safe_locs


if __name__ == "__main__":
    # player = Player()
    # level = Level_01(player)
    # env = Platformer(player, level)
    env, safe_locs = env_from_file('../discrete_platformer_levels/level6.txt')

    # Loop until the user clicks the close button.
    done = False

    # Used to manage how fast the screen updates
    clock = pygame.time.Clock()

    left = False
    right = False
    running = False

    x_dot_desired = 0
    jump = 0

    # -------- Main Program Loop -----------
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    env.reset()
                if event.key == pygame.K_LEFT:
                    left = True
                if event.key == pygame.K_RIGHT:
                    right = True
                if event.key == pygame.K_LALT:
                    running = True
                if event.key == pygame.K_UP:
                    jump = 1

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT:
                    left = False
                if event.key == pygame.K_RIGHT:
                    right = False
                if event.key == pygame.K_LALT:
                    running = False
                if event.key == pygame.K_UP:
                    jump = 0

            if left:
                if running:
                    x_dot_desired = -2
                else:
                    x_dot_desired = -1
            elif right:
                if running:
                    x_dot_desired = 2
                else:
                    x_dot_desired = 1
            else:
                x_dot_desired = 0

        r, s = env.execute_agent_action([x_dot_desired, jump])
        print(s)

        # Limit to 60 frames per second
        clock.tick(15)

    # Be IDLE friendly. If you forget this line, the program will 'hang'
    # on exit.
    pygame.quit()
