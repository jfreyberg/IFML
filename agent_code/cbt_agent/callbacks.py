import numpy as np
from time import sleep
from scipy.optimize import curve_fit
import sys 
import traceback
import random
import copy

import settings as s_new
class setting_translate:
    cols = s_new.COLS
    rows = s_new.ROWS
    grid_size = s_new.GRID_SIZE
    crate_density = s_new.CRATE_DENSITY
    actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT']
    max_agents = s_new.MAX_AGENTS
    max_steps = s_new.MAX_STEPS
    bomb_power = s_new.BOMB_POWER
    bomb_timer = s_new.BOMB_TIMER
    explosion_timer = s_new.EXPLOSION_TIMER

    timeout = s_new.TIMEOUT
    reward_kill = s_new.REWARD_KILL
    reward_coin = s_new.REWARD_COIN
    reward_slow = -1
s = setting_translate


moves = np.array([[]])
path = './'
crate_counter = 0
round_number = 1

def func_curve(X, a_1, a_2, a_3, a_4, a_5, a_6, a_7, a_8, a_9, a_10, a_11, a_12, a_13, a_14, a_15, a_16, a_17, a_18, a_19, a_20, a_21, a_22, a_23, a_24, a_25, a_26, a_27, a_28, a_29, a_30, a_31, a_32, a_33, a_34):
    (x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19, x_20, x_21, x_22, x_23, x_24, x_25, x_26, x_27, x_28, x_29, x_30, x_31, x_32, x_33) = X
    return a_1*x_1 + a_2*x_2 + a_3*x_3 + a_4*x_4 + a_5*x_5 + a_6*x_6 + a_7*x_7 + a_8*x_8 + a_9*x_9 + a_10*x_10 + a_11*x_11 + a_12*x_12 + a_13*x_13 + a_14*x_14 + a_15*x_15 + a_16*x_16 + a_17*x_17 + a_18*x_18 + a_19*x_19 + a_20*x_20 + a_21*x_21 + a_22*x_22 + a_23*x_23 + a_24*x_24 + a_25*x_25 + a_26*x_26 + a_27*x_27 + a_28*x_28 + a_29*x_29 + a_30*x_30 + a_31*x_31 + a_32*x_32 + a_33*x_33 + a_34


def positions(self):
    agent = self.game_state['self']
    agent_pos = np.array([agent[0], agent[1]])
    coins = np.array(self.game_state['coins'])
    if len(coins) == 0:
        # return agent_pos three times, otherwise the agent would try to go to the top left corner, this way he "tries to stay where he is"
        return agent_pos, agent_pos, np.zeros(2)
    else:
        dist = np.array([])   
        for i in range(len(coins)):
            dist = np.append(dist, np.linalg.norm(agent_pos - coins[i]))
        coin = coins[np.argmin(dist)]
        mean_coin = []
        
        for i in range(len(coins)):
            if dist[i] != 0:
                mean_coin.append((agent_pos - coins[i])/dist[i]**(3/2))
        if (len(np.shape(mean_coin)) > 1):
            mean_coin = np.sum(mean_coin, axis=0)
        if np.linalg.norm(mean_coin) != 0: 
            mean_coin = (mean_coin)/ np.linalg.norm(mean_coin) 
        if (mean_coin == []):
            mean_coin = np.zeros(2)
        return agent_pos, np.array(coin), mean_coin # agent_pos, absolute pos naechster coin, richtung mean coin

def difference(self):
    agent_pos, coin, mean_coin = positions(self) 
    arena = self.game_state['arena']
    diff = np.linalg.norm(agent_pos - coin)
    if diff != 0 :
        direction = (agent_pos - coin) / diff
    else:
        direction = np.zeros(2)

    return np.array([diff, *direction, *mean_coin])


def crate_positions(self):
    agent = self.game_state['self']
    agent_pos = np.array([agent[0], agent[1]])
    arena = self.game_state['arena']
    crates = []
    for i in range(np.shape(arena)[0]):
        for j in range(np.shape(arena)[1]):
            if arena[i][j] == 1:
                crates.append([i,j])
    crates = np.array(crates) 
    if len(crates) == 0:
        # return agent_pos three times, otherwise the agent would try to go to the top left corner, this way he "tries to stay where he is"
        return agent_pos, agent_pos, np.zeros(2)
    else:
        dist = np.array([])   
        for i in range(len(crates)):
            dist = np.append(dist, np.linalg.norm(agent_pos - crates[i]))
        crate = crates[np.argmin(dist)]
        mean_crate = []
        
        for i in range(len(crates)):
            if dist[i] != 0:
                mean_crate.append((agent_pos - crates[i])/dist[i]**(3/2))
        if (len(np.shape(mean_crate)) > 1):
            mean_crate = np.sum(mean_crate, axis=0)
        if np.linalg.norm(mean_crate) != 0: 
            mean_crate = (mean_crate)/ np.linalg.norm(mean_crate) 
        if (mean_crate == []):
            mean_crate = np.zeros(2)
        return agent_pos, np.array(crate), mean_crate # agent_pos, absolute pos naechster coin, richtung mean coin

def crate_diff(self):
    agent_pos, crate, mean_crate = crate_positions(self)     
    diff = np.linalg.norm(agent_pos - crate)
    if diff != 0 :
        direction = (agent_pos - crate) / diff
        if diff != 1:
            diff = 0
    else:
        direction = np.zeros(2)

    return np.array([*direction, *mean_crate, diff])

def check_even_odd(position):
    # 1 means, that the row/column is free
    # x and y are swapped, because the y direction is free if the x value is odd
    position = np.array(position)[np.r_[1, 0]] #np.r_ does the swap of x and y
    return position % 2
   
def last_move(self):
    forbidden = np.zeros(4)
    try:
        move = [self.game_state['self'][:2][i] - self.last_coords[i] for i in range(2)]
        moves = [[-1, 0], [1, 0], [0, -1], [0, 1]]
        last_move_ = moves.index(move)
        if (self.game_state['step'] != 1):
            forbidden[int(last_move_ + 1 - 2 * (last_move_ % 2))] = 1
    except:
        pass
    return forbidden


def own_bomb_ticking(self):
    #returns 1 if own bomb is ticking, else 0
    bomb_possible = self.game_state['self'][3]
    return (bomb_possible + 1) % 2

def explosion_radius_single_bomb(coordinates):
    check_free = check_even_odd(coordinates)
    x, y, x_, y_ = *coordinates, *check_free
    row = [[item, y] for item in np.unique(np.clip(np.r_[x-3:x+4], 0, 16))] if x_ else [[x, y]]
    column = [[x, item] for item in np.unique(np.clip(np.r_[y-3:y+4], 0, 16))] if y_ else [[x, y]] 
    return row + column

def position_in_danger(position, self):
    danger = 0
    for bomb in self.game_state['bombs']:
        if (list(position) in explosion_radius_single_bomb(bomb[:2])):
            if bomb[2] == 0:
                danger = 5
            else:
                danger = np.max([(0.633-0.133*bomb[2]), danger])
    if self.game_state['explosions'][tuple(position)] == 2:
        danger = 5
    return danger + 4 * np.clip(danger, 0, 0.1)

def number_of_crates_in_explosion_radius(self):
    own_pos = np.array(self.game_state['self'][:2])
    explosion_radius = explosion_radius_single_bomb(own_pos)
    tiles = np.array([self.game_state['arena'][tuple(item)] for item in explosion_radius])
    number = len(np.arange(len(tiles))[tiles == 1])
    number = 0.7 * number if number <= 2 else number
    return number

def next_move_danger(self):
    # returns 0 if no danger is in the corresponding direction or a wall 
    danger = np.zeros(4)
    own_pos = np.array(self.game_state['self'][:2])
    pos_diffs = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
    for i in range(4):
        danger[i] = position_in_danger(own_pos + pos_diffs[i], self)
        pos = own_pos + pos_diffs[i]
        if (np.abs(self.game_state['arena'][tuple([pos[0], pos[1]])]) == 1):
            danger[i] = 0.8
    return danger

def directions_blocked(pos, self):
    pos = np.array(pos)
    # returns 0 if direction is free, 1 if it is blocked 
    blocked = np.zeros(4)
    pos_diffs = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
    for i in range(4):
        blocked[i] = np.abs(self.game_state['arena'][tuple(pos + pos_diffs[i])])
        if list(pos + pos_diffs[i]) in [list(coord[:-1]) for coord in self.game_state['bombs']]:
            blocked[i] = 1
        if list(pos + pos_diffs[i]) in [list(coord[:2]) for coord in self.game_state['others']]:
            blocked[i] = 1
    return blocked
    
def next_move_blocked (self):
    blocked = np.zeros(4)
    own_pos = np.array(self.game_state['self'][:2])
    pos_diffs = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]]) 
    for i in range(4):
        current_pos = np.array(own_pos + pos_diffs[i])
        if (current_pos < 1).any() or (current_pos > 15).any():
            blocked[i] = 1
        else:
            blocked[i] = -3 + np.clip(np.sum(directions_blocked(current_pos, self)), 3, 4)
    return blocked

def tile_blocked(pos, self):
    blocked = np.abs(self.game_state['arena'][tuple(pos)])
    if list(pos) in [list(coord[:-1]) for coord in self.game_state['bombs']]:
        blocked = 1
    if list(pos) in [list(coord[:2]) for coord in self.game_state['others']]:
        blocked = 1
    return blocked

def no_through_road(self, check_no_bomb = False):
    blocked = np.zeros(4)
    own_pos = np.array(self.game_state['self'][:2])
    if (list(own_pos) not in [list(coord[:-1]) for coord in self.game_state['bombs']] and not check_no_bomb):
        return blocked
    pos_diffs = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]]) # left, right, up, down 
    orthogonal_list = np.array([[2, 3], [0, 1]])
    for i in range(4): #all directions
        cur_orthogonal = orthogonal_list[int(np.floor(0.5*i))]
        for j in range(4):
            current_pos = own_pos + (j+1)*pos_diffs[i]
            if (tile_blocked(current_pos, self) == 1):
                blocked[i] = 1
                break
            elif (directions_blocked(current_pos, self)[cur_orthogonal] == 0).any():
                break
    return blocked

def no_bomb(self):
    all_directions_blocked = no_through_road(self, check_no_bomb = True).all()
    return 1 if all_directions_blocked else 0

def killable_opponents(self):
    number = 0
    if own_bomb_ticking(self) == 0:
        explosion_radius = explosion_radius_single_bomb(self.game_state['self'][:2])
        for opponent in self.game_state['others']:
            if list(opponent[:2]) in explosion_radius:
                number += 1
    return number


def q_function(theta_q, features):
    f = theta_q[:,-1]
    for i in range(len(features)):
        f = f + theta_q[:, i] * features[i] 
    return f


def build_features (self):
    current_pos = self.game_state['self'][:2]
    features = difference(self) # [diff, *direction, *mean_coin] also 5 Werte, Indizes 0, 1, 2, 3, 4
    features = np.append(features, check_even_odd(current_pos)) # 2 Werte, Indizes 5, 6
    features = np.append(features, last_move(self)) # 4 Werte, Indizes 7, 8, 9, 10
    features = np.append(features, position_in_danger(current_pos, self)) # 1 Wert, Index 11
    features = np.append(features, own_bomb_ticking(self)) # 1 Wert, Index 12
    features = np.append(features, number_of_crates_in_explosion_radius(self)) # 1 Wert, Index 13
    features = np.append(features, directions_blocked(current_pos, self)) # 4 Werte, Indizes 14, 15, 16, 17
    features = np.append(features, next_move_danger(self)) # 4 Werte, Indizes 18, 19, 20, 21
    features = np.append(features, crate_diff(self)) # 5 Werte, Indizes 22, 23, 24, 25, 26
    features = np.append(features, no_through_road(self)) # 4 Werte, Indizes 27, 28, 29, 30
    features = np.append(features, no_bomb(self)) # 1 Wert, Index 31
    features = np.append(features, killable_opponents(self)) # 1 Wert, Index 32
    self.last_coords = current_pos
    return features





def setup(self):
    try:
        setup_(self)
    except Exception as e:
        print('Exception as e:')
        print('line: ', sys.exc_info()[2].tb_lineno)
        print(type(e), e) 
        print('Traceback:')
        traceback.print_tb(sys.exc_info()[2], file = sys.stdout)
        print('end of traceback.') 
    pass
    
def setup_(self):
    self.last_coords = [0, 0]
    self.theta = np.load('{}thetas/theta_q.npy'.format(path))
    self.q_data = np.load('{}q_data/q_data.npy'.format(path))
    self.all_data = np.load('{}all_data/all_data.npy'.format(path))

    if len(self.q_data) > 6000:
        self.q_data = self.q_data[-6000:]
    if len(self.all_data) > 30000:
        self.all_data = self.all_data[np.array(random.sample(list(np.arange(len(self.all_data))), 30000))]
    pass


def act(self, game_state: dict):
    self.game_state = {
        'step': game_state['round'],
        'arena': game_state['field'],
        'self': [(x, y, n, b) for (n, s, b, (x, y)) in [game_state['self']]][0],
        'others': [(x, y, n, b) for (n, s, b, (x, y)) in game_state['others']],
        'bombs': [(x, y, t) for ((x, y), t) in game_state['bombs']],
        'explosions': game_state['explosion_map'],
        'coins': game_state['coins'],
    }
    try:
        global round_number

        features = build_features(self)

        action = {0:'LEFT', 1:'RIGHT', 2:'UP', 3:'DOWN', 4:'WAIT', 5:'BOMB'}
        q_value = q_function(self.theta, features)

        eps = 0.05
        e = np.random.uniform(0,1)
        if e < eps:
            chosen_action = int(np.random.choice([0,1,2,3,4,5], p = [0.2, 0.2, 0.2, 0.2, 0.1, 0.1]))
            # print('step', self.game_state['step'], 'random: ', action[chosen_action])
        else:
            chosen_action = int(np.argmax(q_value))

        self.q_data = np.append(self.q_data, [np.array([chosen_action, q_value[chosen_action], *features])], axis=0)
        self.all_data = np.append(self.all_data, [np.array([chosen_action, q_value[chosen_action], *features])], axis=0)

        self.next_action = action[chosen_action]
        
    except Exception as e:
        print('Exception as e:')
        print('line: ', sys.exc_info()[2].tb_lineno)
        print(type(e), e) 
        print('Traceback:')
        traceback.print_tb(sys.exc_info()[2], file = sys.stdout)
        print('end of traceback.')

    return self.next_action

def reward_update(self):
    global moves, round_number
    last_event = self.q_data[-1]
    f = last_event[2:]
    
    # sleep(10)

    event_dict = {0 : "MOVED_LEFT",
            1 : "MOVED_RIGHT",
            2 : "MOVED_UP",
            3 : "MOVED_DOWN",
            4 : "WAITED",
            5 : "INTERRUPTED",
            6 : "INVALID_ACTION",
            7 : "BOMB_DROPPED",
            8 : "BOMB_EXPLODED",
            9 : "CRATE_DESTROYED",
            10 : "COIN_FOUND",
            11 : "COIN_COLLECTED",
            12 : "KILLED_OPPONENT",
            13 : "KILLED_SELF",
            14 : "GOT_KILLED",
            15 : "OPPONENT_ELIMINATED",
            16 : "SURVIVED_ROUND"}

    rewards = {"MOVED_LEFT" : -2 + f[1] + 0.2*f[3] + 0.2*f[5] - 3*f[7] - 35*f[18] + 2*f[22] + 0.4*f[24] - 22*f[27], 
                    "MOVED_RIGHT" : -2 - f[1] - 0.2*f[3] + 0.2*f[5] - 3*f[8] - 35*f[19] - 2*f[22] - 0.4*f[24] - 22*f[28],
                    "MOVED_UP" : -2 + f[2] + 0.2*f[4] + 0.2*f[6] - 3*f[9] - 35*f[20] + 2*f[23] + 0.4*f[25] - 22*f[29],
                    "MOVED_DOWN" : -2 - f[2] - 0.2*f[4] + 0.2*f[6] - 3*f[10] - 35*f[21] - 2*f[23] - 0.4*f[25] - 22*f[30],
                    "WAITED" : -6 - 300*f[11], 
                    "INTERRUPTED" : 0,
                    "INVALID_ACTION" : -5 - 5*f[5] - 5*f[6] - 50*f[12] - 10*np.clip(np.sum(f[14:18]), 0, 1),
                    
                    "BOMB_DROPPED" :  -5 + 5*f[13] + 2*f[26] - 30*f[31] + 10*f[32] - 4*f[11],
                    "BOMB_EXPLODED" : 0, 
                    
                    "CRATE_DESTROYED" : 2,
                    "COIN_FOUND" : 0,
                    "COIN_COLLECTED" : 20,

                    "KILLED_OPPONENT" : 0,
                    "KILLED_SELF" : 0,

                    "GOT_KILLED" : 0,
                    "OPPONENT_ELIMINATED" : 0,
                    "SURVIVED_ROUND" : 0}

    reward = 0.667761 * np.sum([rewards[event_dict[item]] for item in self.events])

    global crate_counter
    for event in self.events:
        if event == 9:
            crate_counter += 1

    moves = np.append(moves, [last_event[0], reward, *f])
    moves = np.reshape(moves, (int(len(moves.flat)/(2+len(f))), 2+len(f)))
    return None


def end_of_episode(self):
    try:
        global moves, round_number, path, crate_counter
        alpha = 0.04        
        gamma = 0.4
        n = 7    
        for t in range(len(moves)):
            if (t >= len(moves) - n):
                n = len(moves) - t - 1
            q_next = np.max(q_function(self.theta, moves[t + n, 2:])) 
            y_t = np.sum([gamma**(t_ - t - 1) * moves[t_, 1] for t_ in range(t + 1, t + n + 1)]) + gamma**n * q_next

            q_update = self.q_data[-len(moves) + t, 1] + alpha * (y_t - self.q_data[-len(moves) + t, 1])   
            
            self.q_data[-len(moves) + t, 1] = q_update        

        
        with open('{}moves.txt'.format(path), 'a') as f:
            new_coins = self.game_state['self'][4]
            last_score = self.game_state['self'][4]
            f.write(str(round_number) + ' ' + str(len(moves)) + ' ' + str(new_coins) + ' ' + str(crate_counter) + '\n')
            crate_counter = 0

        if (round_number % 100 == 0):
            try: 
                theta = []
                for i in range(6):
                    regression_data = copy.deepcopy(self.q_data)
                    if len(self.all_data) > 6000:
                        tmp = self.all_data[:-6000]
                        if len(tmp) > 6000:
                            tmp = tmp[np.array(random.sample(list(np.arange(len(tmp))), 6000))]
                        regression_data = np.append(tmp, regression_data, axis = 0)
                    
                    mask = regression_data[:,0]==i
                    regression_data = regression_data[mask][:, 1:]
                    
                    popt, pcov = curve_fit(func_curve, (regression_data[:, 1:].T), regression_data[:, 0], p0=self.theta[i])
                    theta.append(popt)
                self.theta = np.clip(np.array(theta), -90, 20)

            except Exception as e:
                print(e)
                print('theta unverändert')
                print('Exception as e:')
                print('line: ', sys.exc_info()[2].tb_lineno)
                print(type(e), e) 
                print('Traceback:')
                traceback.print_tb(sys.exc_info()[2], file = sys.stdout)
                print('end of traceback.')
                pass
         
        if len(self.q_data) > 6000:
            self.q_data = self.q_data[-6000:]       
        
        if len(self.all_data) > 30000:
            self.all_data = self.all_data[np.array(random.sample(list(np.arange(len(self.all_data))), 30000))]    

        if round_number % 20 == 0:
            np.save('{}thetas/theta_q.npy'.format(path), self.theta)
            np.save('{}q_data/q_data.npy'.format(path), self.q_data)
            np.save('{}all_data/all_data.npy'.format(path), self.all_data)
            np.save('{}thetas/theta_nach_{}_spielen.npy'.format(path, round_number), self.theta)
            if round_number % 60 == 0:
                np.save('{}all_data/all_data_nach_{}_spielen.npy'.format(path, round_number), self.all_data)

        round_number += 1
        moves = np.array([[]])    
        # print('END')
    except Exception as e:
        print('Exception as e:')
        print('line: ', sys.exc_info()[2].tb_lineno)
        print(type(e), e) 
        print('Traceback:')
        traceback.print_tb(sys.exc_info()[2], file = sys.stdout)
        print('end of traceback.')
        
    return None


