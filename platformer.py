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

# Global constants

# Actions
from simple_rl.mdp import State

LEFT = 'left'
RIGHT = 'right'
NOOP = 'no-op'
JUMP_LEFT = 'jump left'
JUMP_RIGHT = 'jump right'
JUMP = 'jump'
ACTIONS = [LEFT, RIGHT, NOOP, JUMP]

# Colors
PLATFORM_COLOR = (30, 30, 30)
PLAYER_COLOR = (255, 140, 0)
BG_COLOR = (0, 100, 100)
GOAL_COLOR = (200, 255, 200)

# Screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

# Game constants
LEFT_SHIFT_LIMIT = 120
RIGHT_SHIFT_LIMIT = 500
TILE_WIDTH = 100
TILE_HEIGHT = 100
GOAL_WIDTH = 50
GOAL_HEIGHT = TILE_HEIGHT*2
PLAYER_SPEED = 6
JUMP_IMPULSE = 12
JUMP_STD = 1.2
GRAVITY = 0.35

# State discretization constants
NUM_STATES_PER_TILE = 1
NUM_STATES_PER_JUMP = 3


CAN_MOVE_IN_AIR = False
STOPS_ON_EDGES = False


class Player(pygame.sprite.Sprite):
    """
    This class represents the bar at the bottom that the player controls.
    """

    # -- Methods
    def __init__(self):
        """ Constructor function """

        # Call the parent's constructor
        super().__init__()

        # Create an image of the block, and fill it with a color.
        # This could also be an image loaded from the disk.
        width = 40
        height = 60
        self.image = pygame.Surface([width, height])
        self.image.fill(PLAYER_COLOR)

        # Set a referance to the image rect.
        self.rect = self.image.get_rect()

        # Set speed vector of player
        self.change_x = 0
        self.change_y = 0

        # List of sprites we can bump against
        self.level = None

    def update(self):
        """ Move the player. """
        # Gravity
        self.calc_grav()

        # Move left/right
        self.rect.x += self.change_x

        # See if we hit anything
        block_hit_list = pygame.sprite.spritecollide(self, self.level.platform_list, False)
        for block in block_hit_list:
            # If we are moving right,
            # set our right side to the left side of the item we hit
            if self.change_x > 0:
                self.rect.right = block.rect.left
                if not CAN_MOVE_IN_AIR:
                    self.change_x = 0
            elif self.change_x < 0:
                # Otherwise if we are moving left, do the opposite.
                self.rect.left = block.rect.right
                if not CAN_MOVE_IN_AIR:
                    self.change_x = 0

        # See if we hit edges
        if STOPS_ON_EDGES:
            if self.rect.left - self.level.world_shift < 0:
                self.rect.left = self.level.world_shift
                if not CAN_MOVE_IN_AIR:
                    self.change_x = 0
            elif self.rect.right - self.level.world_shift > self.level.level_limit:
                self.rect.right = self.level.level_limit + self.level.world_shift
                if not CAN_MOVE_IN_AIR:
                    self.change_x = 0

        # Move up/down
        self.rect.y += self.change_y

        # Check and see if we hit anything
        block_hit_list = pygame.sprite.spritecollide(self, self.level.platform_list, False)
        for block in block_hit_list:

            # Reset our position based on the top/bottom of the object.
            if self.change_y > 0:
                self.rect.bottom = block.rect.top
            elif self.change_y < 0:
                self.rect.top = block.rect.bottom

            # Stop our vertical movement
            self.change_y = 0

    def calc_grav(self):
        """ Calculate effect of gravity. """
        if self.change_y == 0:
            self.change_y = 1
        else:
            self.change_y += GRAVITY

    def on_platform(self):
        """ Is the user on a platform (True) or in the air (False)? """
        # move down a bit and see if there is a platform below us.
        # Move down 2 pixels because it doesn't work well if we only move down 1
        # when working with a platform moving down.
        self.rect.y += 2
        platform_hit_list = pygame.sprite.spritecollide(self, self.level.platform_list, False)
        self.rect.y -= 2

        return len(platform_hit_list) > 0

    def jump(self):
        """ Called when user hits 'jump' button. """

        # If it is ok to jump, set our speed upwards
        if self.on_platform():
            self.change_y = -np.random.normal(JUMP_IMPULSE, JUMP_STD)

    # Player-controlled movement:
    def go_left(self):
        """ Called when the user hits the left arrow. """
        if CAN_MOVE_IN_AIR or self.on_platform():
            self.change_x = -PLAYER_SPEED

    def go_right(self):
        """ Called when the user hits the right arrow. """
        if CAN_MOVE_IN_AIR or self.on_platform():
            self.change_x = PLAYER_SPEED

    def stop(self):
        """ Called when the user lets off the keyboard. """
        if CAN_MOVE_IN_AIR or self.on_platform():
            self.change_x = 0


class Platform(pygame.sprite.Sprite):
    """ Platform the user can jump on """

    def __init__(self, width, height):
        """ Platform constructor. Assumes constructed with user passing in
            an array of 5 numbers like what's defined at the top of this code.
            """
        super().__init__()

        self.image = pygame.Surface([width, height])
        self.image.fill(PLATFORM_COLOR)

        self.rect = self.image.get_rect()


class Goal(pygame.sprite.Sprite):
    """ Goal """

    def __init__(self):
        super().__init__()

        self.image = pygame.Surface([GOAL_WIDTH, GOAL_HEIGHT])
        self.image.fill(GOAL_COLOR)

        self.rect = self.image.get_rect()


class Level:
    def __init__(self, platform_list, goal_location, level_limit):
        """ Constructor. Pass in a handle to player. Needed for when moving
            platforms collide with the player. """
        self.platform_list = pygame.sprite.Group()
        for platform in platform_list:
            self.platform_list.add(platform)

        self.enemy_list = pygame.sprite.Group()

        self.goal = Goal()
        self.goal.rect.x = goal_location[0]
        self.goal.rect.bottom = goal_location[1]
        self.goal_sprite_group = pygame.sprite.Group()
        self.goal_sprite_group.add(self.goal)

        self.level_limit = level_limit

        # How far this world has been scrolled left/right
        self.world_shift = 0

    # Update everythign on this level
    def update(self):
        """ Update everything in this level."""
        self.platform_list.update()
        self.enemy_list.update()

    def draw(self, screen):
        """ Draw everything on this level. """

        # Draw the background
        screen.fill(BG_COLOR)

        # Draw all the sprite lists that we have
        self.platform_list.draw(screen)
        self.enemy_list.draw(screen)
        self.goal_sprite_group.draw(screen)

    def shift_world(self, shift_x):
        """ When the user moves left/right and we need to scroll
        everything: """

        # Keep track of the shift amount
        self.world_shift += shift_x

        # Go through all the sprite lists and shift
        for platform in self.platform_list:
            platform.rect.x += shift_x

        for enemy in self.enemy_list:
            enemy.rect.x += shift_x

        self.goal.rect.x += shift_x


class PlatformerState(State):
    def __init__(self, agent_loc, agent_vel, is_terminal=False):
        self.agent_x = agent_loc[0]
        self.agent_y = agent_loc[1]
        self.x_dot = agent_vel[0]
        self.y_dot = agent_vel[1]

        State.__init__(self, data=[self.agent_x, self.agent_y, self.x_dot, self.y_dot], is_terminal=is_terminal)

    def descretize(self):
        return PlatformerState((int(self.agent_x*NUM_STATES_PER_TILE/TILE_WIDTH),
                                int(self.agent_y*NUM_STATES_PER_TILE/TILE_HEIGHT)),
                               (int(self.x_dot/PLAYER_SPEED),
                                min(NUM_STATES_PER_JUMP-1,
                                    int((s.y_dot+JUMP_IMPULSE)*NUM_STATES_PER_JUMP/(2*JUMP_IMPULSE)))))


def discrete_horizontal_invariance(state1, action1, state2, action2, platformer):
    if platformer.platform_at_tile[state1.agent_x] \
            and platformer.platform_at_tile[state2.agent_x] \
            and (state1.agent_y, state1.x_dot, state1.y_dot) == (state2.agent_y, state2.x_dot, state2.y_dot) \
            and action1 == action2:
        return 1
    else:
        return 0


def discrete_horizontal_invariance_equivalency(state1, action1, state_prime1, state2, action2, all_states):
    state_prime2 = None
    if (state1.agent_y, state1.x_dot, state1.y_dot) == (state2.agent_y, state2.x_dot, state2.y_dot) \
            and action1 == action2:
        x_diff = state_prime1.agent_x - state1.agent_x
        x = state2.agent_x + x_diff
        for s in all_states:
            if s.agent_x == x \
                    and (state_prime1.agent_y, state_prime1.x_dot, state_prime1.y_dot) == (s.agent_y, s.x_dot, s.y_dot):
                state_prime2 = s

    return state_prime2


class Platformer:
    def __init__(self, player, level, platform_at_tile):
        """ Main Program """
        pygame.init()

        # Set the height and width of the screen
        size = [SCREEN_WIDTH, SCREEN_HEIGHT]
        self.screen = pygame.display.set_mode(size)

        pygame.display.set_caption("Side-scrolling Platformer")

        # Set player
        self.player = player
        self.init_player_loc = (self.player.rect.x, self.player.rect.y)

        # Set the current level
        self.level = level
        self.player.level = self.level

        self.active_sprite_list = pygame.sprite.Group()

        self.active_sprite_list.add(self.player)

        self.terminal = False
        self.reward = 0
        self.platform_at_tile = platform_at_tile

        self._set_curr_state()

    def execute_agent_action(self, action, action_repeat=1):
        r_sum = 0
        for i in range(action_repeat):
            r_sum += self._act(action)
        self._set_curr_state()
        return r_sum, self.state

    def _act(self, action):
        if self.terminal:
            return 0

        if action in [LEFT, JUMP_LEFT]:
            self.player.go_left()
        elif action in [RIGHT, JUMP_RIGHT]:
            self.player.go_right()
        elif action in [NOOP]:
            self.player.stop()

        if action in [JUMP, JUMP_RIGHT, JUMP_LEFT]:
            self.player.jump()

        # Update the player.
        self.active_sprite_list.update()

        # Update items in the level
        self.level.update()

        # If the player gets near the right side, shift the world left (-x)
        if self.player.rect.right >= RIGHT_SHIFT_LIMIT:
            if STOPS_ON_EDGES:
                if -self.level.world_shift + SCREEN_WIDTH < self.level.level_limit:
                    diff = min(self.player.rect.right - RIGHT_SHIFT_LIMIT,
                               self.level.level_limit - (-self.level.world_shift + SCREEN_WIDTH))
                    self.player.rect.right = RIGHT_SHIFT_LIMIT
                    self.level.shift_world(-diff)
            else:
                diff = self.player.rect.right - RIGHT_SHIFT_LIMIT
                self.player.rect.right = RIGHT_SHIFT_LIMIT
                self.level.shift_world(-diff)

        # If the player gets near the left side, shift the world right (+x)
        if self.player.rect.left <= LEFT_SHIFT_LIMIT:
            if STOPS_ON_EDGES:
                if -self.level.world_shift > 0:
                    diff = min(LEFT_SHIFT_LIMIT - self.player.rect.left,
                               -self.level.world_shift)
                    self.player.rect.left = LEFT_SHIFT_LIMIT
                    self.level.shift_world(diff)
            else:
                diff = LEFT_SHIFT_LIMIT - self.player.rect.left
                self.player.rect.left = LEFT_SHIFT_LIMIT
                self.level.shift_world(diff)

        # Check if fell off
        if self.player.rect.bottom >= SCREEN_HEIGHT:
            self.terminal = True
            self.reward = -1

        # Check if reached goal
        if pygame.sprite.spritecollide(self.player, self.level.goal_sprite_group, False):
            self.terminal = True
            self.reward = 1

        # Draw
        self.level.draw(self.screen)
        self.active_sprite_list.draw(self.screen)

        return self.reward

    def get_curr_state(self):
        return self.state

    def _set_curr_state(self):
        player_x = self.player.rect.x - self.level.world_shift
        player_y = self.player.rect.y
        self.state = PlatformerState((player_x, player_y),
                                     (self.player.change_x, self.player.change_y),
                                     is_terminal=self.terminal)

    def reset(self):
        self.terminal = False
        self.level.shift_world(-self.level.world_shift)
        self.player.rect.x = self.init_player_loc[0]
        self.player.rect.y = self.init_player_loc[1]
        self.player.change_x = 0
        self.player.change_y = 0

        self._set_curr_state()

        # Draw
        self.level.draw(self.screen)
        self.active_sprite_list.draw(self.screen)


def env_from_file(file_name):
    level_file = open(os.path.join(os.getcwd(), file_name))
    level_lines = level_file.readlines()
    num_lines = len(level_lines)

    platform_at_tile = np.zeros([len(level_lines[0])])

    platform_dims = []
    platform_x = None
    player_location = None
    goal_location = None
    level_limit = None
    for i, line in enumerate(level_lines):
        line = line.strip()
        line_length = len(line)
        for j, ch in enumerate(line):
            if ch == '1':
                if platform_x is None:
                    platform_x = j * TILE_WIDTH
                platform_at_tile[j] = 1
            else:
                if platform_x is not None:
                    platform_dims.append([j * TILE_WIDTH - platform_x, TILE_HEIGHT,
                                          platform_x, SCREEN_HEIGHT - (num_lines - i) * TILE_HEIGHT])
                    platform_x = None

            if ch == 'G':
                goal_location = (j * TILE_WIDTH, SCREEN_HEIGHT - (num_lines - (i + 1)) * TILE_HEIGHT)
            elif ch == 'a':
                player_location = (j * TILE_WIDTH, SCREEN_HEIGHT - (num_lines - (i + 1)) * TILE_HEIGHT)

        if platform_x is not None:
            platform_dims.append([line_length * TILE_WIDTH - platform_x, TILE_HEIGHT,
                                  platform_x, SCREEN_HEIGHT - (num_lines - i) * TILE_HEIGHT])
            platform_x = None

        level_limit = line_length * TILE_WIDTH

    # Create player
    player = Player()
    player.rect.x = player_location[0]
    player.rect.bottom = player_location[1]

    # Go through the array above and add platforms
    platform_list = []
    for dims in platform_dims:
        block = Platform(dims[0], dims[1])
        block.rect.x = dims[2]
        block.rect.y = dims[3]
        block.player = player
        platform_list.append(block)

    level = Level(platform_list, goal_location, level_limit)
    env = Platformer(player, level)
    return env


if __name__ == "__main__":
    # player = Player()
    # level = Level_01(player)
    # env = Platformer(player, level)
    env = env_from_file('platformer_levels/simple_level.txt')

    # Loop until the user clicks the close button.
    done = False

    # Used to manage how fast the screen updates
    clock = pygame.time.Clock()

    left = False
    right = False
    action = NOOP

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
                    action = LEFT
                if event.key == pygame.K_RIGHT:
                    right = True
                    action = RIGHT
                if event.key == pygame.K_UP:
                    action = JUMP

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT:
                    left = False
                    if action == LEFT:
                        action = NOOP
                if event.key == pygame.K_RIGHT:
                    right = False
                    if action == RIGHT:
                        action = NOOP
                if event.key == pygame.K_UP and action == JUMP:
                    if left:
                        action = LEFT
                    elif right:
                        action = RIGHT
                    else:
                        action = NOOP

        r, s = env.execute_agent_action(action)

        # Limit to 60 frames per second
        clock.tick(60)

        # Go ahead and update the screen with what we've drawn.
        pygame.display.flip()

    # Be IDLE friendly. If you forget this line, the program will 'hang'
    # on exit.
    pygame.quit()
