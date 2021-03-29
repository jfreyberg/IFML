import numpy as np
from random import shuffle

from settings import s
from settings import e


# Function modified from items.py
def get_blast_coords(arena, x, y):
    """Retrieve the blast range for a bomb.

    The maximal power of the bomb (maximum range in each direction) is
    imported directly from the game settings. The blast range is
    adjusted according to walls (immutable obstacles) in the game
    arena.

    Parameters:
    * arena:  2-dimensional array describing the game arena.
    * x, y:   Coordinates of the bomb.

    Return Value:
    * Array containing each coordinate of the bomb's blast range.
    """
    bomb_power = s.bomb_power
    blast_coords = [(x, y)]

    for i in range(1, bomb_power+1):
        if arena[x+i, y] == -1: break
        blast_coords.append((x+i, y))
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


# Function modified from simple_agent to return the path towards a
# target, instead of only the first step.
def look_for_targets_path(free_space, start, targets, logger=None):
    """Find direction of closest target that can be reached via free tiles.

    Performs a breadth-first search of the reachable free tiles until
    a target is encountered.  If no target can be reached, the path
    that takes the agent closest to any target is chosen.

    Args:
        free_space: Boolean numpy array. True for free tiles and False for obstacles.
        start: the coordinate from which to begin the search.
        targets: list or array holding the coordinates of all target tiles.
        logger: optional logger object for debugging.
    Returns:
        the path towards closest target or towards tile closest to any
        target, beginning at the next step.
    """
    if len(targets) == 0:
        return []

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

    # Determine the path towards the best found target tile, start not included
    current = best
    path = []
    while True:
        path.insert(0, current)
        if parent_dict[current] == start:
            return path
        current = parent_dict[current]


def look_for_targets(free_space, start, targets, logger=None):
    """Returns the coordinate of first step towards closest target, or
    towards tile closest to any target.
    """
    path = look_for_targets_path(free_space, start, targets, logger=None)
    #print("PATH TO TARGET: ", path, sep=" ")

    if len(path):
        return path[0]
    else:
        return None
