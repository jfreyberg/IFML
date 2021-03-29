import numpy as np
import copy
import torch
from collections import deque


class StateProcessor():
    def __init__(self, window_size):
        self.window_size = window_size

        self.set_up_for_new_round()

    def set_up_for_new_round(self):
        # this one is only for debugging / finding errors
        self.prev_explosion_map = None

        # here we store who planted bombs when to know when they will have a new bomb again
        self.bomb_timeout = {}

        # initially there is a coin in every area
        self.coin_areas = np.ones((3, 3))

        # keep track of previous coin distribution
        self.coin_history = deque([], maxlen=5)

        # keep track of own position in the maze
        self.agent_move_history = deque([np.zeros((6,)), np.zeros((6,))], maxlen=2)
        self.previous_position = None

        # store previous state and current step for caching
        self.previous_state = None
        self.current_step = 0

    def window_for_position(self, m, pos, N=3, outside=-2):
        x = pos[1] + N
        y = pos[0] + N
        m = np.pad(m, N, 'constant', constant_values=outside)

        x_min = x - N
        x_max = x + N + 1
        y_min = y - N
        y_max = y + N + 1

        return m[x_min:x_max, y_min:y_max]

    def get_blast_coords(self, arena, explosion_map, pos, val):
        power = 4
        y, x = pos

        for i in range(power):
            if arena[x + i, y] == -1:
                break
            explosion_map[x + i, y] = max(val, explosion_map[x + i, y])
        for i in range(power):
            if arena[x - i, y] == -1:
                break
            explosion_map[x - i, y] = max(val, explosion_map[x - i, y])
        for i in range(power):
            if arena[x, y + i] == -1:
                break
            explosion_map[x, y + i] = max(val, explosion_map[x, y + i])
        for i in range(power):
            if arena[x, y - i] == -1:
                break
            explosion_map[x, y - i] = max(val, explosion_map[x, y - i])

        return explosion_map

    def future_explosions(self, field, bombs):

        explosion_map = np.zeros_like(field)
        for bomb in bombs:
            pos = bomb[0]
            danger_level = 4 - bomb[1]
            '''
            computed like this:
            input   steps till explosion    danger level
            0       1                       4
            1       2                       3
            2       3                       2
            3       4                       1
            '''

            explosion_map = self.get_blast_coords(field, explosion_map, pos, danger_level)

        return explosion_map

    def update_bomb_timeout(self, enemy_names, agent_has_bomb, enemy_bombs):
        # only in the first iteration: initialize bomb_timeout list
        if len(self.bomb_timeout.keys()) == 0:
            for name in enemy_names:
                self.bomb_timeout[name] = 6
            self.bomb_timeout['self'] = 6

        for i, name in enumerate(enemy_names):
            has_bomb = enemy_bombs[i]

            # case: enemy dropped a bomb :o
            if self.bomb_timeout[name] == 6 and not has_bomb:
                self.bomb_timeout[name] = 1
            # case: no bomb could be dropped (timeout)
            elif self.bomb_timeout[name] < 6:
                self.bomb_timeout[name] += 1

        # case: we dropped a bomb
        if self.bomb_timeout['self'] == 6 and not agent_has_bomb:
            self.bomb_timeout['self'] = 1
        # case: we could not drop a bomb (timeout)
        elif self.bomb_timeout['self'] < 6:
            self.bomb_timeout['self'] += 1

    def get_explosion_map(self, field, bombs, explosion_map, agent_index):
        future_explosion_map = self.future_explosions(field, bombs)

        # this is how to test code properly!!
        if np.min(np.subtract(field * 1000, future_explosion_map)) < -1000:
            # if a wall would be affected by an explosion, this case should be triggered
            print('alarm! explosion on undestructable field!')

        full_explosion_map = np.maximum(explosion_map, future_explosion_map)
        # print(full_explosion_map)

        if self.prev_explosion_map is not None:
            if np.max(np.subtract(full_explosion_map, self.prev_explosion_map)) > 1:
                # an explosion occured where it should not because it has not been predicted in the previous step!
                print('alarm! unexpected explosion!')

        self.prev_explosion_map = full_explosion_map

        full_explosion_map = self.window_for_position(full_explosion_map,
                                                      agent_index, N=self.window_size, outside=0)

        return full_explosion_map

    def get_can_enter(self, field, enemy_indices, bomb_indices, agent_index):
        can_enter_calculation_field = copy.deepcopy(field)
        for pos in enemy_indices:
            can_enter_calculation_field[pos[1], pos[0]] = -1
        for pos in bomb_indices:
            can_enter_calculation_field[pos[1], pos[0]] = -1
        self.full_can_enter = can_enter_calculation_field
        can_enter = self.window_for_position(can_enter_calculation_field,
                                             agent_index, N=self.window_size, outside=-2)
        can_enter = np.equal(can_enter, np.zeros_like(can_enter)).astype('int')
        return can_enter

    def get_can_destroy(self, field, agent_index):
        return self.window_for_position(np.equal(field,
                                                 np.ones_like(field)).astype('int'), agent_index, N=self.window_size,
                                        outside=0)

    # the map is split into 9 5x5 areas where each area initially contains one coin
    # we encode which of these still contain a coin with a 1 and where there is no
    # coin anymore with 0.

    def update_coin_areas_left_on_map(self, coin_indices, previous_coin_indices):
        # print(previous_coin_indices)
        for c in previous_coin_indices:
            if c not in coin_indices:
                #print('coin collected: {}'.format(c))
                # oh, a coin is gone...
                # let's find out in which area!
                if self.coin_areas[(c[0] - 1) // 5][(c[1] - 1) // 5] == 0:
                    print('ERROR: illegal coin detected!')
                self.coin_areas[(c[0] - 1) // 5][(c[1] - 1) // 5] = 0

    def get_own_last_move(self, own_pos):
        #OPTIONS = ['RIGHT', 'LEFT', 'BOMB', 'UP', 'DOWN', 'WAIT']
        move_vector = np.zeros((6,))
        if self.previous_position == None:
            self.previous_position = own_pos
            return move_vector

        if self.bomb_timeout['self'] == 1:
            move_vector[2] = 1
        elif self.previous_position[0] - own_pos[0] < 0:
            move_vector[0] = 1
        elif self.previous_position[0] - own_pos[0] > 0:
            move_vector[1] = 1
        elif self.previous_position[1] - own_pos[1] < 0:
            move_vector[4] = 1
        elif self.previous_position[1] - own_pos[1] > 0:
            move_vector[3] = 1
        else:
            move_vector[5] = 1
        # @approved by all

        self.previous_position = own_pos
        return move_vector

    def get_feature_space(self, game_state):

        if game_state['step'] == self.current_step:
            #print('cached state')
            return self.previous_state

        self.current_step += 1
        if self.current_step != game_state['step']:
            print('ERROR: unexpected current step (StateProcessor.py: get_feature_space)')
        #print('not cached state')

        field = game_state['field'].T
        bombs = game_state['bombs']
        agent_index = game_state['self'][3]
        agent_has_bomb = game_state['self'][2]
        explosion_map = game_state['explosion_map'].T * 5
        enemies = game_state['others']

        enemy_indices = [x[3] for x in enemies]
        enemy_scores = [x[1] for x in enemies]
        enemy_names = [x[0] for x in enemies]
        enemy_bombs = [x[2] for x in enemies]
        bomb_indices = [x[0] for x in bombs]
        coin_indices = game_state['coins']
        previous_coin_indices = self.coin_history[-1] if len(self.coin_history) > 0 else []

        self.update_bomb_timeout(enemy_names, agent_has_bomb, enemy_bombs)

        full_explosion_map = self.get_explosion_map(field, bombs, explosion_map,
                                                    agent_index)

        can_enter = self.get_can_enter(field, enemy_indices, bomb_indices,
                                       agent_index)

        can_destroy = self.get_can_destroy(field, agent_index)

        coins = np.zeros_like(field)
        for pos in coin_indices:
            coins[pos[1], pos[0]] = 1
        full_coins = coins
        coins = self.window_for_position(coins, agent_index, N=self.window_size, outside=0)

        #########################################
        #########  enemy_score          #########
        #########################################

        enemy_pos = np.zeros_like(field)
        for i, pos in enumerate(enemy_indices):
            enemy_pos[pos[1], pos[0]] = 1 + enemy_scores[i]
        enemy_pos = self.window_for_position(enemy_pos, agent_index,
                                             N=self.window_size, outside=0)

        #########################################
        #########  enemy_bomb_timeout   #########
        #########################################

        enemy_bombs = np.zeros_like(field)
        for i, pos in enumerate(enemy_indices):
            enemy_bombs[pos[1], pos[0]] = self.bomb_timeout[enemy_names[i]]
        full_enemy_bombs = enemy_bombs
        enemy_bombs = self.window_for_position(enemy_bombs, agent_index,
                                               N=self.window_size, outside=0)

        #########################################
        #########  coins left on map    #########
        #########################################

        self.update_coin_areas_left_on_map(coin_indices, previous_coin_indices)
        coin_areas = np.array(self.coin_areas).flatten()

        self.coin_history.append(coin_indices)

        #########################################
        #########  own position in maze #########
        #########################################

        own_position_in_maze = np.zeros((3, 3))
        own_position_in_maze[(agent_index[0] - 1) // 5][(agent_index[1] - 1) // 5] = 1
        own_position_in_maze = np.array(own_position_in_maze).flatten()

        #########################################
        #########  own bomb timeout     #########
        #########################################

        own_bomb_time_out = np.array([self.bomb_timeout['self']])

        #########################################
        #########    get last move      #########
        #########################################

        last_move = self.get_own_last_move(agent_index)
        self.agent_move_history.append(last_move)
        move_history = np.array([self.agent_move_history[0], self.agent_move_history[1]]).flatten()
        # print(move_history)

        res = np.array([
            # enemy_pos,
            enemy_bombs,
            coins,
            can_destroy,
            full_explosion_map,
            can_enter,
        ])

        extra = np.concatenate((coin_areas, own_position_in_maze, own_bomb_time_out, move_history))

        # print(extra)
        self.previous_state = (res, extra)
        return res, extra


def rotate_state(state, extra):
    state = torch.rot90(state, 1, (2,3))

    # inputs
    coin_areas = extra[:None, 0:9]
    position_in_maze = extra[:None, 9:18]
    bomb_timeout = extra[:None, 18].view((-1,1))
    move_history = extra[:None, 19:]

    # processing
    coin_areas = torch.rot90(coin_areas.view((-1, 3, 3)), 1, (1,2)).reshape((-1, 9))
    position_in_maze = torch.rot90(position_in_maze.view((-1, 3, 3)), 1, (1,2)).reshape((-1, 9))
    move_history = rotate_action(move_history)

    extra = torch.cat((coin_areas, position_in_maze, bomb_timeout, move_history), dim=1)

    return state, extra


def rotate_action(actions):
    #OPTIONS = ['RIGHT', 'LEFT', 'BOMB', 'UP', 'DOWN', 'WAIT']
    new_actions = actions.clone()

    new_actions[actions == 0] = 3
    new_actions[actions == 1] = 4
    new_actions[actions == 3] = 0
    new_actions[actions == 4] = 1

    return new_actions
