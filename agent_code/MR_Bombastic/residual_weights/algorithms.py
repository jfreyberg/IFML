import numpy as np
from random import shuffle
from time import time, sleep
from collections import deque
import pickle
import copy

from settings import s
from settings import e


############# USEFUL FUNCTIONS ##############

def estimate_blast_coords(arena,bombs):
    """
    is only an idea on how the bomb blast ranges may be computed
    """
    for bomb in bombs_xy:
        # compute patches in posible directions
        # up
        if bomb[1]-bomb_power <1:
            #patch = arena[1:bomb[1]+1,bomb[0]]
            patch = arena[bomb[0], 1:bomb[1]+1]
        else:
            #patch = arena[bomb[1]-bomb_power:bomb[1]+1, bomb[0]]
            patch = arena[bomb[0], bomb[1]-bomb_power:bomb[1]+1]

def compute_blast_coords(arena, bomb):
    """
    !! This function may be good for computations, as it takes to much time :/
    to compute
    Retrieve the blast of all bombs acording a given stage.
    The maximal power of the bomb (maximum range in each direction) is
    imported directly from the game settings. The blast range is
    adjusted according to walls, other agents and crates (immutable obstacles).
    Parameters:
    * game_state
    Return Value:
    * Array containing blast range coordinates of all bombs (list of tuples).

    May be useful for feature 2
    """
    bomb_power = s.bomb_power
    x, y = bomb[0], bomb[1] 
    blast_coords = [(x,y)]

    for i in range(1, bomb_power+1):
        if arena[x+i, y] == -1: break
        blast_coords.append((x+i,y))
    for i in range(1, bomb_power+1):
        if arena[x-i, y] == -1: break 
        blast_coords.append((x-i, y))
    for i in range(1, bomb_power+1):
        if arena[x, y+i] == -1: break 
        blast_coords.append((x, y+i))
    for i in range(1, bomb_power+1):
        if arena[x, y-i] == -1: break 
        blast_coords.append((x, y-i))

    return blast_coords

def compute_patch(arena, p1, p2):
    """
    this function computes the patch of the arena between the points p1 and p2
    USEFUL FOR feat1
    """
    patch = arena[min(p1[1], p2[1]):max(p1[1], p2[1])+1,  min(p1[0], p2[0]):max(p1[0], p2[0])+1] 
    return patch

def manhattan_metric(p1, p2):
    """
    p1 or p2 may be an array of points in numpy format
    USEFUL FOR feat1
    """
    absdiff = np.abs(p1 - p2)
    sum_absdiff = np.sum(absdiff, axis=1)
    return sum_absdiff

def feat_1(game_state):
    """
        Feature extraction for coin detection
        Old implementation useful for ideas and maybe use at some point distance computations
    """
    coins = game_state['coins']
    x, y, _, bombs_left = game_state['self']
    directions = [(x,y-1), (x,y+1), (x-1,y), (x+1,y)]
    arena = game_state['arena']

    feature = [] # Desired feature

    # check if there are coins available if not return 0 as feature
    if coins == []: 
        return np.expand_dims(np.zeros(6), axis=0)

    for d in directions:
        # if invalid action set to zero
        if arena[d] != 0:
            feature.append(0)
            continue

        # compute manhattan distance between all visible coins and the agent
        manh_dist = manhattan_metric(np.asarray(coins), np.array(d))

        # find the nearest coins
        min_coin, dist2min_coin = np.argmin(manh_dist), np.min(manh_dist)
        indx_mincoins = list(np.where(manh_dist == dist2min_coin)[0])
        
        ## correct manhattan distance for special cases
        # compute patches between min_coins and agent 
        patches = [] 
        for m in indx_mincoins:
            p = compute_patch(arena, coins[m], d)
            patches.append(p)
        
        # look if there is a fast path to the closes coin
        FAST_PATH = False 
        for patch in patches:
            if patch.shape[0] == 1 or patch.shape[1] == 1:
                if np.count_nonzero(patch) == 0:
                    FAST_PATH=True
                    break
            else:
                FAST_PATH=True
                break
        if not FAST_PATH:
            dist2min_coin += 2

        # fill features
        
        #feature.append(f)
        feature.append(1000-dist2min_coin) #1000 random value
        
    feature.append(0)
    feature.append(0)
    
    # convert the maximum values of feature to 1 and the rest to 0
    help_list = []
    for f in feature:
        if f == max(feature):
            help_list.append(1)
        else:
            help_list.append(0)

    feature = help_list 
    # because this feature doesn't take in consideration using bombs
    #f_bomb = np.expand_dims(np.zeros(feature.shape[1]), axis=0)
    #feature = np.concatenate((feature,f_bomb), axis=0)

    #feature = np.expand_dims(feature, axis=0)
    return feature


def look_for_targets(free_space, start, targets, logger=None):
    """Find direction of closest target that can be reached via free tiles.

    Performs a breadth-first search of the reachable free tiles until a target is encountered.
    If no target can be reached, the path that takes the agent closest to any target is chosen.

    Args:
        free_space: Boolean numpy array. True for free tiles and False for obstacles.
        start: the coordinate from which to begin the search.
        targets: list or array holding the coordinates of all target tiles.
        logger: optional logger object for debugging.
    Returns:
        coordinate of first step towards closest target or towards tile closest to any target.

    USEFUL FOR feature1
    """
    if len(targets) == 0: return None

    frontier = [start]
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best = start
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()

    while len(frontier) > 0:
        current = frontier.pop(0)
        # Find distance from current position to all targets, track closest
        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        if d + dist_so_far[current] <= best_dist:
            best = current
            best_dist = d + dist_so_far[current]
        if d == 0:
            # Found path to a target's exact position, mission accomplished!
            best = current
            break
        # Add unexplored free neighboring tiles to the queue in a random order
        x, y = current
        neighbors = [(x,y) for (x,y) in [(x+1,y), (x-1,y), (x,y+1), (x,y-1)] if free_space[x,y]]
        shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1
    if logger: logger.debug(f'Suitable target found at {best}')
    # Determine the first step towards the best found target tile
    current = best
    while True:
        if parent_dict[current] == start: return current
        current = parent_dict[current]

def look_for_targets_path(free_space, start, targets, logger=None):
    """Reverse the path from a suitable tile to the start tile.

    Parameters:
    * free_space:  Boolean numpy array. True for free tiles and False for obstacles.
    * start:       Coordinate from which to begin the search.
    * targets:     List or array holding the coordinates of all target tiles.
    * logger:      Optional logger object for debugging.
    """
    if len(targets) == 0:
        return None

    frontier = [start]
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best = start
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()

    while len(frontier) > 0:
        current = frontier.pop(0)

        # Find distance from current position to all targets, track closest
        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        if d + dist_so_far[current] <= best_dist:
            best = current
            best_dist = d + dist_so_far[current]
        if d == 0:
            # Found path to a target's exact position, mission accomplished!
            best = current
            break

        # Add unexplored free neighboring tiles to the queue in a random order
        x, y = current
        neighbors = [(x,y) for (x,y) in [(x+1,y), (x-1,y), (x,y+1), (x,y-1)] if free_space[x,y]]

        shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1

    if logger:
        logger.debug(f'Suitable target found at {best}')

    current = best
    path = []

    while True:
        # Insert visited node to the beginning of the path.
        path.insert(0, current)
        current = parent_dict[current]

        # TODO: check if logger works
        if current == start:
            if logger:
                logger.debug(f'Using path {path} to target {best}')
            return path


def get_blast_coords(bomb, arena, arr):
    x, y = bomb[0], bomb[1]
    if len(arr)== 0:
       arr = [(x,y)]
       #np.append(a, [[0,1]], axis=0)
    
    for i in range(1, 3+1):
        if arena[x+i,y] == -1: break
        arr.append((x+i,y))
    for i in range(1, 3+1):
        if arena[x-i,y] == -1: break
        arr.append((x-i,y))           
    for i in range(1, 3+1):
        if arena[x,y+i] == -1: break
        arr.append((x,y+i))            
    for i in range(1, 3+1):
        if arena[x,y-i] == -1: break
        arr.append((x,y-i))
    return arr

############# FEATURES ##############

def feature1(game_state):
    """
    Reward the best possible action to a coin, if it is reachable F(s,a)=1,  otherwise F(s,a)=0.    
    `BOMB' and `WAIT ' are always 0
    """
    coins = game_state['coins']
    x, y, _, bombs_left = game_state['self']
    directions = [(x,y-1), (x,y+1), (x-1,y), (x+1,y)]
    arena = game_state['arena']
    bombs = game_state['bombs']
    bombs_xy = [(x,y) for (x,y,t) in bombs]

    feature = [] # Desired feature
    
    # construct the free_space Boolean numpy_array
    free_space = arena == 0
    for xb, yb in bombs_xy:
        free_space[xb, yb] = False

    best_coord = look_for_targets(free_space, (x,y), coins)

    directions = [(x,y-1), (x,y+1), (x-1,y), (x+1,y)]

    if best_coord is None:
        return np.zeros(6)
    for d in directions:
        if d != best_coord:
            feature.append(0)
        else:
            feature.append(1)
    
   # for 'BOMB' and 'WAIT'
    feature.append(0)
    feature.append(0)
    
    return np.asarray(feature)


def feature2(game_state):
    """
    Penalize if the action follows the agent to death (F,s)=1, F(s,a)=0. otherwise.
    it could be maybe get better
    """
    x, y, _, bombs_left = game_state['self']
    directions = [(x,y-1), (x,y+1), (x-1,y), (x+1,y), (x,y)]
    bombs = game_state['bombs']
    others = [(x,y) for (x,y,n,b) in game_state['others']]
    bombs_xy = [(x,y) for (x,y,t) in bombs]
    explosions = game_state['explosions'] 
    arena = game_state['arena']
    bomb_power = s.bomb_power
    blast_coords = bombs_xy 
    feature = []
    
    bomb_map = np.ones(arena.shape) * 5
    for xb,yb,t in bombs:
        for (i,j) in [(xb+h, yb) for h in range(-3,4)] + [(xb, yb+h) for h in range(-3,4)]:
            if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                bomb_map[i,j] = min(bomb_map[i,j], t)

    danger_zone = [] 
    if len(bombs) != 0:
        '''
        If there are bombs on the game board, we map all the explosions
        that will be caused by the bombs. For this, we use the help function
        get_blast_coords. This is an adjusted version of the function with the
        same name from item.py of the original framework
        '''
        for b in bombs_xy:
            danger_zone += compute_blast_coords(arena, b)
    #danger_zone += bombs_xy

    for d in directions:
        if ((arena[d] != 0) or 
            (d in others) or 
            (d in bombs_xy)):
            d = (x,y)
            
        if ((d in danger_zone) and  bomb_map[d] == 0) or explosions[d] >1:
            feature.append(-1) 
        else:
            feature.append(0)
    
    # BOMB actions should be same as WAIT action
    feature.append(feature[-1])

    return np.asarray(feature)


def feature3(game_state):
    """
    Penalize the agent for going into an area threatened by a bomb.
    """
    x, y, _, bombs_left = game_state['self']
    directions = [(x,y-1), (x,y+1), (x-1,y), (x+1,y), (x,y)]
    bombs = game_state['bombs']
    others = [(x,y) for (x,y,n,b) in game_state['others']]
    bombs_xy = [(x,y) for (x,y,t) in bombs]
    arena = game_state['arena']
    feature = []
    danger_zone = []

    if len(bombs) != 0:
        '''
        If there are bombs on the game board, we map all the explosions
        that will be caused by the bombs. For this, we use the help function
        get_blast_coords. This is an adjusted version of the function with the
        same name from item.py of the original framework
        '''
        for b in bombs_xy:
            danger_zone += compute_blast_coords(arena, b)
    #danger_zone += bombs_xy

    for d in directions:
        if ((arena[d] != 0) or 
            (d in others) or 
            (d in bombs_xy)):
            d = (x,y)

        if d in danger_zone:
            feature.append(-1) 
        else:
            feature.append(0)
    
    # BOMB actions should be same as WAIT action
    feature.append(feature[-1])

    return np.asarray(feature)


def feature4(state):
    """Reward the agent for moving in the shortest direction outside
    the blast range of (all) bombs in the game.
    """
    bombs = state['bombs']
    bombs_xy = [(x,y) for (x,y,t) in bombs]
    arena = state['arena']
    others = state['others']    
    x, y, _, _ = state['self']

    directions = [(x,y-1), (x,y+1), (x-1,y), (x+1,y), (x,y)]
    feature = []


    # Compute the blast range of all bombs in the game ('danger zone')
    danger_zone = []
    for b in bombs_xy:
        danger_zone += compute_blast_coords(arena, b)

    if len(bombs) == 0 or (x,y) not in danger_zone:
        return np.zeros(6)

    # The agent can use any free tile in the arena to escape from a
    # bomb (which may not immediately explode).
    free_space = arena == 0 # boolean np.ndarray
  
    for xb, yb in bombs_xy:
        free_space[xb, yb] = False

    targets = [(x,y) for x in range(1,16) for y in range(1,16) if (arena[x, y] == 0) and (x,y) not in danger_zone]

    # def look_for_targets(free_space, start, targets, logger=None):
    safety_direction = look_for_targets(free_space, (x, y), targets)

    if safety_direction is None:
        return np.zeros(0)
    
    # check if next action moves agent towards safety
    for d in directions:
        if d == safety_direction:
            feature.append(1)
        else:
            feature.append(0)

    # Do not reward placing a bomb at this stage.
    feature.append(feature[-1])
    feature[-2] = 0

    return feature

def feature5(game_state):
    """
    Penalize invalid actions.  F(s,a) = 1, otherwise F(s,a) = 0.  
    """
    x, y, _, bombs_left = game_state['self']
    directions = [(x,y-1), (x,y+1), (x-1,y), (x+1,y)]
    bombs = game_state['bombs']
    bombs_xy = [(x,y) for (x,y,t) in bombs]
    others = [(x,y) for (x,y,n,b) in game_state['others']]
    arena = game_state['arena']

    feature = [] # Desired feature

    for d in directions:
        if ((arena[d] != 0) or 
            (d in others) or 
            (d in bombs_xy)):
            feature.append(-1)
        else:
            feature.append(0)

    # for 'BOMB'
    if bombs_left == 0:
        feature.append(1)
    else:
        feature.append(0)

    # for 'WAIT'
    feature.append(0)

    return np.asarray(feature)

def feature6(game_state):
    """
    Reward when getting a coin F(s,a) = 1, otherwise F(s,a) = 0
    """
    x, y, _, bombs_left = game_state['self']
    directions = [(x,y-1), (x,y+1), (x-1,y), (x+1,y)]
    bombs = game_state['bombs']
    bombs_xy = [(x,y) for (x,y,t) in bombs]
    others = [(x,y) for (x,y,n,b) in game_state['others']]
    coins = game_state['coins']
    arena = game_state['arena']

    feature = [] # Desired feature

    for d in directions:
        if d in coins:
            feature.append(1)
        else:
            feature.append(0)

    # for 'BOMB' and 'WAIT'
    feature.append(0)
    feature.append(0)

    return np.asarray(feature)

def feature7(game_state):
    """
    Reward putting a bomb next to a crate. 
    F(s,a) = 0 for all actions and F(s,a) = 1 for a 'BOMB' if we are next to a block.
    """
    x, y, _, bombs_left = game_state['self']
    directions = [(x,y-1), (x,y+1), (x-1,y), (x+1,y)]
    arena = game_state['arena']
    
    feature = [0, 0, 0, 0] # Desired feature

    # add feature for the action 'BOMB'
    # check if we are next to a crate
    CHECK_FOR_CRATE = False 
    for d in directions:
        if arena[d] == 1:
            CHECK_FOR_CRATE = True 
            break
    if CHECK_FOR_CRATE and bombs_left>0:
        feature.append(1)
    else:
        feature.append(0)

    # add feature for 'WAIT'
    feature.append(0)

    return np.asarray(feature)


def feature8(game_state):
    """
    Reward (if there are no blocks anymore ? and no coins?)  the available movements F(s,a) = 1, 
    otherwise F(s,a) = 0 .   Bombs = 0, WAIT =1 ? 
    """
    x, y, _, bombs_left = game_state['self']
    directions = [(x,y-1), (x,y+1), (x-1,y), (x+1,y)]
    arena = game_state['arena']
    
    feature = [] # Desired feature

    for d in directions:
        if arena[d] == 0:
            feature.append(1)
        else:
            feature.append(0)

    # for 'BOMB' and 'WAIT'
    feature.append(0)
    feature.append(1)

    return np.asarray(feature)

def feature9(game_state):
    """
    Reward going into dead-ends (from simple agent)
    """
    x, y, _, bombs_left = game_state['self']
    directions = [(x,y-1), (x,y+1), (x-1,y), (x+1,y)]
    bombs = game_state['bombs']
    bombs_xy = [(x,y) for (x,y,t) in bombs]
    others = [(x,y) for (x,y,n,b) in game_state['others']]
    arena = game_state['arena']

    dead_ends = [(x,y) for x in range(1,16) for y in range(1,16) if (arena[x,y] == 0)
                    and ([arena[x+1,y], arena[x-1,y], arena[x,y+1], arena[x,y-1]].count(0) == 1)]

    # construct the free_space Boolean numpy_array
    free_space = arena == 0

    for xb, yb in bombs_xy:
        free_space[xb, yb] = False

    best_coord = look_for_targets(free_space, (x,y), dead_ends)
    
    if (x,y) in dead_ends:
        return np.zeros(6)

    feature = []
    if best_coord is None:
        return np.zeros(6)
    for d in directions:
        if d != best_coord:
            feature.append(0)
        else:
            feature.append(1)
    
    # for 'BOMB' and 'WAIT'
    feature += [0,0]

    return feature



def feature10(game_state):
    """
    Reward going to crates 
    """
    x, y, _, bombs_left = game_state['self']
    directions = [(x,y-1), (x,y+1), (x-1,y), (x+1,y)]
    bombs = game_state['bombs']
    bombs_xy = [(x,y) for (x,y,t) in bombs]
    others = [(x,y) for (x,y,n,b) in game_state['others']]
    arena = game_state['arena']
    crates = [(x,y) for x in range(1,16) for y in range(1,16) if (arena[x,y] == 1)]

    # construct the free_space Boolean numpy_array
    free_space = arena == 0
    
    for xc, yc in crates:
        free_space[xc,yc] = True

    #bombs_others = tuple([[c[0], c[1]] for c in bombs_xy + others]) #tuple(bombs_xy + others)

    for xb, yb in bombs_xy :
        free_space[xb, yb] = False
    best_coord = look_for_targets(free_space, (x,y), crates)
    
    feature = []
    
    if best_coord is None:
        return np.zeros(6)

    #if (x,y) == best_coord or (x,y) in bombs_xy:
    #    return np.zeros(6)
    
    for d in directions:
        if d in crates:
            return np.zeros(6)

    for d in directions:
        if d != best_coord:
            feature.append(0)
        else:
            feature.append(1)
    
    # for 'BOMB' 
    feature += [0,0]

    return feature


def feature11(game_state):
    """
    Penalize moving getting into dead_ends if the last action was 'BOMB'
    """
    x, y, _, bombs_left = game_state['self']
    directions = [(x,y-1), (x,y+1), (x-1,y), (x+1,y)]
    bombs = game_state['bombs']
    bombs_xy = [(x,y) for (x,y,t) in bombs]
    others = [(x,y) for (x,y,n,b) in game_state['others']]
    arena = game_state['arena']

    dead_ends = [(x,y) for x in range(1,16) for y in range(1,16) if (arena[x,y] == 0)
                    and ([arena[x+1,y], arena[x-1,y], arena[x,y+1], arena[x,y-1]].count(0) == 1)]

    feature = []  # feature that we want get

    for d in directions:
        if (d in dead_ends) and ((x,y) in bombs_xy):
            feature.append(-1)
        else:
            feature.append(0)

    # For 'BOMB' and 'WAIT'
    feature += [0,0]

    return feature


def feature12(game_state):
    """
    HUNTING MODE: Reward puting a bomb next to an agent

    Nur if len(coins) != 0 or len(crates) !=0:
        return np.zeros(6)
    """
    x, y, _, bombs_left = game_state['self']
    directions = [(x,y-1), (x,y+1), (x-1,y), (x+1,y)]
    arena = game_state['arena']
    others = [(x,y) for (x,y,n,b) in game_state['others']]
    crates = [(x,y) for x in range(1,16) for y in range(1,16) if (arena[x,y] == 1)]
    coins = game_state['coins']
    
    feature = [0, 0, 0, 0] # Desired feature

    if len(coins) != 0 or len(crates) !=0:
        return np.zeros(6)

    # add feature for the action 'BOMB'
    # check if we are next to a crate
    CHECK_FOR_OTHERS = False 
    for d in others:
        if arena[d] == 1:
            CHECK_FOR_OTHERS = True 
            break

    if CHECK_FOR_OTHERS and bombs_left>0:
        feature.append(1)
    else:
        feature.append(0)

    # add feature for 'WAIT'
    feature.append(0)

    return np.asarray(feature)


def feature13(game_state):
    """
    Reward putting a bomb that traps another agent if he is dead-ends
    """
    x, y, _, bombs_left = game_state['self']
    directions = [(x,y-1), (x,y+1), (x-1,y), (x+1,y)]
    arena = game_state['arena']
    others = [(x,y) for (x,y,n,b) in game_state['others']]
    
    feature = [0, 0, 0, 0] # Desired feature

    dead_ends = [(x,y) for x in range(1,16) for y in range(1,16) if (arena[x,y] == 0)
                    and ([arena[x+1,y], arena[x-1,y], arena[x,y+1], arena[x,y-1]].count(0) == 1)]

    CHECK_CONDITION = False
    for o in others:
        if o in dead_ends:
            CHECK_CONDITION = True
            break

    CHECK_OTHERS = False
    for d in directions:
        if d in others:
            CHECK_OTHERS= True
            break

    if CHECK_CONDITION and CHECK_OTHERS:
        feature.append(1)
    else:
        feature.append(0)

    # For 'WAIT'
    feature.append(0)

    return feature


def feature14(game_state):
    """
    Hunting mode:
    If no crates and no coins reward going towards the nearest agent
    """
    
    x, y, _, bombs_left = game_state['self']
    directions = [(x,y-1), (x,y+1), (x-1,y), (x+1,y)]
    arena = game_state['arena']
    others = [(x,y) for (x,y,n,b) in game_state['others']]
    crates = [(x,y) for x in range(1,16) for y in range(1,16) if (arena[x,y] == 1)]
    coins = game_state['coins']
    bombs = game_state['bombs']
    bombs_xy = [(x,y) for (x,y,t) in bombs]
    
    if len(coins) != 0 or len(crates) !=0:
        return np.zeros(6)

    # construct the free_space Boolean numpy_array
    free_space = arena == 0
    for xb, yb in bombs_xy:
        free_space[xb, yb] = False

    best_coord = look_for_targets(free_space, (x,y), others)
    
    feature = []
    
    if best_coord is None:
        return np.zeros(6)
    
    for d in directions:
        if d != best_coord:
            feature.append(0)
        else:
            feature.append(1)

    feature += [0,0]

    return feature

def feature15(game_state):
    """
    if not in Hunting mode:
    If no crates and no coins penalize going towards the nearest agent
    """
    
    x, y, _, bombs_left = game_state['self']
    directions = [(x,y-1), (x,y+1), (x-1,y), (x+1,y)]
    arena = game_state['arena']
    others = [(x,y) for (x,y,n,b) in game_state['others']]
    crates = [(x,y) for x in range(1,16) for y in range(1,16) if (arena[x,y] == 1)]
    coins = game_state['coins']
    bombs = game_state['bombs']
    bombs_xy = [(x,y) for (x,y,t) in bombs]
    
    if len(coins) == 0 and len(crates) ==0:
        return np.zeros(6)

    # construct the free_space Boolean numpy_array
    free_space = arena == 0
    for xb, yb in bombs_xy:
        free_space[xb, yb] = False

    best_coord = look_for_targets(free_space, (x,y), others)
    
    feature = []
    
    if best_coord is None:
        return np.zeros(6)
    
    for d in directions:
        if d != best_coord:
            feature.append(0)
        else:
            feature.append(-1)

    feature += [0,0]

    return feature

def feature_extraction(game_state):

    f0 = np.ones(6)  # for bias
    f1 = feature1(game_state) # reward good action
    f2 = feature2(game_state) # penalization bad action
    f3 = feature3(game_state)
    f4 = feature4(game_state) # reward good action
    f5 = feature5(game_state)  # penalize bad action
    f6 = feature6(game_state)  # reward good action
    f7 = feature7(game_state) # reward action
    f9 = feature9(game_state) # rewards good action
    f10 = feature10(game_state) # rewards good action
    f11 = feature11(game_state) # penalize bad action
    f12 = feature12(game_state) # penalize bad action
    f13 = feature13(game_state) # penalize bad action
    f14 = feature14(game_state) # penalize bad action
    f15 = feature15(game_state) # penalize bad action

    return np.vstack((f1,f2,f3,f4,f5,f6,f7,f9,f10,f11,f12,f13,f14,f15)).T

def new_reward(events):
    reward = 0
    for event in events:
        if event == e.BOMB_DROPPED:
            reward += 1
        elif event == e.COIN_COLLECTED:
            reward += 200
        elif event == e.KILLED_SELF:
            reward -= 100
        elif event == e.CRATE_DESTROYED:
            reward += 10
        elif event == e.COIN_FOUND:
            reward += 100
        elif event == e.KILLED_OPPONENT:
            reward += 300
        elif event == e.GOT_KILLED:
            reward -= 300
        elif event == e.SURVIVED_ROUND:
            reward += 100
        elif event == e.INVALID_ACTION:
            reward -= 2
    return reward

def q_gd_linapprox(next_state, prev_state_a, reward, weights, alpha, gamma):
    next_state_a = next_state[np.argmax(np.dot(next_state, weights)), :]
    weights += alpha * (reward + gamma * np.dot(next_state_a,weights) - np.dot(prev_state_a,weights)) * prev_state_a 
    return weights