import numpy as np
import copy

from agent_code.MR_Bombastic.arena import *


class RLFeatureExtraction:
    # TODO: Allow to set the bias?
    def __init__(self, game_state, bias=0, coin_limit=0, crate_limit=2):
        """
        Extract relevant properties from the environment for feature
        extraction.
        """
        # The actions set here determine the order of the columns in the returned
        # feature matrix. (Take as an argument?)
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT']

        # Set the amount of features / weights
        self.bias = bias
        self.dim = 15

        # Collect commonly used data from the environment.
        self.arena = game_state['field']
        self.bombs = game_state['bombs']
        self.coins = game_state['coins']
        self.explosions = game_state['explosion_map']
        self.others = game_state['others']
        self.name, self.score, self.bombs_left, (self.x, self.y) = game_state['self']

        # Some methods only require the coordinates of bombs and
        # opposing agents.
        self.bombs_xy = [(x,y) for ((x,y),t) in self.bombs]
        self.others_xy = [(x,y) for (n,_,b,(x,y)) in self.others]

        # Map actions to coordinates. Placing a bomb or waiting does
        # not move the agent. Other actions assume that the origin
        # (0,0) is located at the top left of the arena.
        self.agent = (self.x, self.y)
        self.directions = {
            'UP'   : (self.x, (self.y)-1),
            'DOWN' : (self.x, (self.y)+1),
            'LEFT' : ((self.x)-1, self.y),
            'RIGHT': ((self.x)+1, self.y),
            'BOMB' : self.agent,
            'WAIT' : self.agent
        }

        # Check the arena (np.array) for free tiles, and include the comparison
        # result as a boolean np.array.
        self.free_space = self.arena == 0

        # Do not include agents as obstacles, as they are likely to
        # move in the next round.
        for (xb, yb), _ in self.bombs:
            self.free_space[xb, yb] = False

        # The blast range (discounted for walls) of a bomb is only available if the
        # bomb has already exploded. We thus compute the range manually.
        self.danger_zone = []
        if len(self.bombs) != 0:
            for (xb, yb), _ in self.bombs:
                self.danger_zone += get_blast_coords(self.arena, xb, yb)

        # Define a "safe zone" of all free tiles in the arena, which are not part of
        # the above danger zone.
        self.safe_zone = [(x, y) for x in range(1, 16) for y in range(1, 16)
                          if (self.arena[x, y] == 0)
                          and (x, y) not in self.danger_zone]

        # Compute a list of all crates in the arena.
        self.crates = [(x,y) for x in range(1,16) for y in range(1,16)
                       if (self.arena[x,y] == 1)]

        # Compute dead ends, i.e. tiles with only a single neighboring, free
        # tile. Only crates and walls are taken into account; opposing agents and
        # bombs are ignored: moving the agent towards these targets may impose a
        # negative reward, and should be left to other features.
        self.dead_ends = [(x, y) for x in range(1, 16) for y in range(1, 16)
                          if (self.arena[x, y] == 0)
                          and ([self.arena[x+1, y],
                                self.arena[x-1, y],
                                self.arena[x, y+1],
                                self.arena[x, y-1]].count(0) == 1)]

        # bomb_map gives the maximum blast range of a bomb, if walls are not taken
        # into account. The values in this array are set to the timer for each
        # bomb. (taken from simple_agent)
        self.bomb_map = np.ones(self.arena.shape) * 5
        for (xb, yb), t in self.bombs:
            for (i, j) in [(xb+h, yb) for h in range(-3, 4)] + [(xb, yb+h) for h in range(-3, 4)]:
                if (0 < i < self.bomb_map.shape[0]) and (0 < j < self.bomb_map.shape[1]):
                    self.bomb_map[i, j] = min(self.bomb_map[i, j], t)                    

        # Compute the feature matrix with columns F_i(S, A) and rows ordered by the
        # actions defined in self.actions.
        self.feature = np.vstack(
            ([self.bias] * len(self.actions),
             self.feature1(),
             self.feature2(),
             self.feature3(),
             self.feature4(),
             self.feature5(),
             self.feature6(),
             self.feature7(),
             self.feature8(coin_limit, crate_limit),
             self.feature10(),
             self.feature11(coin_limit, crate_limit),
             self.feature12(),
             self.feature13(),
             self.feature14(),
             self.feature15())).T
        # test
        #print(self.feature)


    def state(self):
        """
        Return the feature matrix F, where every column represents an
        a feature F_i(S,A), and rows represent actions A.
        """
        return self.feature


    def state_action(self, action):
        """
        Return the column vector for the feature:
           F(S, A) = F_1(S,A) ... F_n(S,A)
        """
        return self.feature[self.actions.index(action), :]


    def max_q(self, weights):
        """
        Return the maximum Q-value for all possible actions, and the corresponding
        action to this maximum. It may be used to update weights during training, or
        to implement a greedy policy. The required weights are assumed known, and
        taken as a parameter.
        """
        # Compute the dot product (w, F_i(S,A)) for every action.
        Q_lfa = np.dot(self.feature, weights)
        Q_max = np.max(Q_lfa)
        A_max = np.where(Q_lfa == Q_max)[0]

        return Q_max, [self.actions[a] for a in A_max]


    def feature0(self):
        return [self.bias] * len(self.actions)


    def feature1(self):
        """
        Reward the agent to move in a direction towards a coin.
        """
        feature = []

        # Check if there are coins available in the arena, and that they can be
        # reached directly by the agent.
        best_direction = look_for_targets_strict(self.free_space, self.agent, self.coins)
        if best_direction == None:
            return [0] * len(self.actions)

        # Check if the next move action matches the direction of the nearest coin.
        for action in self.actions:
            d = self.directions[action]

            # Check if the next action the agent to a different tile (in particular,
            # not a bomb or wait action).
            if d == self.agent:
                feature.append(0)
            elif d == best_direction:
                feature.append(1)
            else:
                feature.append(0)

        return feature


    def feature2(self):
        """
        Penalize the action if it places the agent into a location
        where it is most likely to die.
        """
        feature = []

        # TODO: 'BOMB' and 'WAIT' are computed twice here
        for action in self.actions:
            d = self.directions[action]

            # Check if the tile reached by the next action is occupied by an
            # object. (Opposing agents may wait, thus we should check them even if
            # they can move away.) This object may be destroyed by bombs, but
            # prevents us from moving into a free tile.
            if (self.arena[d] != 0) or (d in self.others_xy) or (d in self.bombs_xy):
                d = self.agent

            # We first check if the agent moves into the blast range of a bomb which
            # will explode directly after. The second condition checks if the agent
            # moves into an ongoing explosion. In both cases, such a movement causes
            # certain death for the agent (that is, we set F_i(s, a) = 1).
            if ((d in self.danger_zone) and (self.bomb_map[d] == 0)) or (self.explosions[d] > 1):
                feature.append(1)
            else:
                feature.append(0)

        return feature


    def feature3(self):
        """Penalize the agent for going or remaining into an area threatened
        by a bomb.

        The logic used in this feature is very similar to feature2. The
        main difference is that we consider any bomb present in the arena,
        not only those that will explode in the next step.
        """
        feature = []

        # TODO: 'BOMB' and 'WAIT' are computed twice here
        for action in self.actions:
            d = self.directions[action]

            # Check if the tile reached by the next action is occupied by an
            # object. If so, only consider the current location of the agent.
            if (self.arena[d] != 0) or (d in self.others_xy) or (d in self.bombs_xy):
                d = self.agent
            if d in self.danger_zone:
                feature.append(1)
            else:
                feature.append(0)

        return feature


    def feature4(self):
        """
        Reward the agent for moving towards the shortest direction outside
        the blast range of (all) bombs in the game.
        """
        feature = []

        # Check if the arena contains any bombs with a blast radius affecting the agent.
        if len(self.bombs) == 0 or (self.agent not in self.danger_zone):
            return [0] * len(self.actions)
        
        # Check if the agent can move into a safe area.
        best_direction = look_for_targets_strict(self.free_space, self.agent, self.safe_zone)
        if best_direction == None:
            return [0] * len(self.actions)

        for action in self.actions:
            d = self.directions[action]

            if action == 'BOMB':
                # When the agent is retreating from one or several bombs, we do not
                # wish to expand the danger zone by dropping a bomb ourselves.
                feature.append(0)
            elif d == best_direction:
                feature.append(1)
            else:
                feature.append(0)

        return feature


    def feature5(self):
        """
        Penalize the agent taking an invalid action.
        """
        feature = []

        for action in self.actions:
            d = self.directions[action]

            if action == 'WAIT':
                # We should check explicitely if the agent is waiting; when dropping
                # a bomb, the agent may remain in the same tile until either the
                # bomb explodes, or the agent takes a move action (after which he
                # may longer move to the tile containing the placed bomb).
                feature.append(0)
            elif (action == 'BOMB') and (self.bombs_left == 0):
                # An agent may only place a bomb if it has any remaining.
                feature.append(1)
            elif action == 'BOMB':
                feature.append(0)
            elif (self.arena[d] != 0) or (d in self.others_xy) or (d in self.bombs_xy):
                # When checking other objects than walls (immutable), we make the
                # following observations regarding invalid actions. Which agent
                # moves first is decided randomly; an initially free square may thus
                # be later occupied by an agent. Crates may be destroyed by a
                # ticking bomb, but this is done only after all agents have
                # performed their respective agents.
                feature.append(1)
            else:
                feature.append(0)

        return feature


    def feature6(self):
        """
        Reward the agent for collecting a coin.
        """
        feature = []

        for action in self.actions:
            d = self.directions[action]

            if d == self.agent:
                feature.append(0)
            elif d in self.coins:
                feature.append(1)
            else:
                feature.append(0)

        return feature


    def feature7(self):
        """
        Reward the agent for placing a bomb next to a crate, if the
        agent can possibly escape from the blast radius.
        """
        feature = []

        for action in self.actions:
            if action == 'BOMB' and self.bombs_left > 0:
                CHECK_FOR_CRATE = False
                for d in self.directions.values():
                    if self.arena[d] == 1: # d != self.agent
                        CHECK_FOR_CRATE = True
                        break

                # Do not reward the agent for placing a bomb if he
                # cannot possibly escape it in a later step.
                danger_zone = copy.deepcopy(self.danger_zone)
                danger_zone += get_blast_coords(self.arena, self.x, self.y)

                safe_zone = [(x, y) for x in range(1, 16) for y in range(1, 16)
                             if self.arena[x, y] == 0 and (x, y) not in danger_zone]
                best_coord = look_for_targets_strict(self.free_space, self.agent, safe_zone)

                if CHECK_FOR_CRATE and best_coord != None:
                    feature.append(1)
                else:
                    feature.append(0)
            else:
                feature.append(0)

        return feature


    def feature8(self, coins_limit, crates_limit):
        """Hunting mode

        Reward the agent for placing a bomb next to an opponent, if
        the agent can possibly escape from the blast radius.
        """
        feature = []

        if len(self.coins) > coins_limit or len(self.crates) > crates_limit:
            return [0] * len(self.actions)

        for action in self.actions:
            if action == 'BOMB' and self.bombs_left > 0:
                CHECK_FOR_OTHERS = False
                for d in self.directions.values():
                    if d in self.others_xy: # d != self.agent
                        CHECK_FOR_OTHERS = True
                        break

                # Do not reward the agent for placing a bomb if he
                # cannot possibly escape it in a later step.
                danger_zone = copy.deepcopy(self.danger_zone)
                danger_zone += get_blast_coords(self.arena, self.x, self.y)

                safe_zone = [(x, y) for x in range(1, 16) for y in range(1, 16)
                             if self.arena[x, y] == 0 and (x, y) not in danger_zone]
                best_coord = look_for_targets_strict(self.free_space, self.agent, safe_zone)

                if CHECK_FOR_OTHERS and best_coord != None:
                    feature.append(1)
                else:
                    feature.append(0)
            else:
                feature.append(0)

        return feature


    # TODO: Rearding moving towards a dead end might encourage the agent to walk
    # into a trap by other agents...
    # def feature9(self):
    #     """
    #     Reward the agent taking a step towards a dead end (a tile with
    #     only a single free, neighboring tile).
    #     """
    #     feature = []
    #     # Here we make no distinction between dead ends that are
    #     # reachable by the agent, and those that are not.
    #     best_direction = look_for_targets(self.free_space, self.agent, self.dead_ends)

    #     # Do not reward if the agent is already in a dead-end, or if there
    #     # are none in the arena.
    #     if self.agent in self.dead_ends or best_direction is None:
    #         return [0] * len(self.actions)

    #     for action in self.actions:
    #         d = self.directions[action]

    #         # Only a move action can bring the agent towards a tile.
    #         if d == self.agent:
    #             feature.append(0)
    #         elif d == best_direction:
    #             feature.append(1)
    #         else:
    #             feature.append(0)

    #     return feature


    def feature10(self):
        """
        Reward the agent for going towards a crate.
        """
        feature = []

        # Observation: look_for_targets requires that any targets are
        # considered free space. (feature 10)
        free_space = copy.deepcopy(self.free_space)
        for xc, yc in self.crates:
            free_space[xc, yc] = True
            
        # Here we make no distinction between crates that are
        # reachable by the agent, and those that are not.
        best_direction = look_for_targets(free_space, self.agent, self.crates)
        if best_direction == None: # in look_for_targets, checks len(self.crates) == 0
            return [0] * len(self.actions)

        # If we are directly next to a create, look_for_targets will
        # return the tile where the agent is located in, rewarding an
        # (unnecessary) wait action.
        # if best_direction == self.agent:
        #     return np.zeros(6)

        for action in self.actions:
            d = self.directions[action]

            # Only a move action can bring the agent towards a crate.
            if d == self.agent:
                feature.append(0)
            else:
                # Give no reward if the agent is already next to a crate.
                # TODO: too specific, already covered by other feature???
                if d in self.crates:
                    return [0] * len(self.actions)
                if d == best_direction:
                    feature.append(1)
                else:
                    feature.append(0)

        return feature


    def feature11(self, coins_limit, crates_limit):
        """Hunting mode

        Reward moving towards opposing agents when the arena contains less than a
        given amount of coins and crates.
        """
        feature = []

        if len(self.coins) > coins_limit or len(self.crates) > crates_limit:
            return [0] * len(self.actions)

        # Check the arena for opponents.
        best_direction = look_for_targets(self.free_space, self.agent, self.others_xy)
        if best_direction == None: # in look_for_targets, checks len(self.crates) == 0
            return[0] * len(self.actions)

        for action in self.actions:
            d = self.directions[action]

            # Only a move action can bring the agent towards an opponent.
            if d == self.agent:
                feature.append(0)
            else:
                # Give no reward if the agent is already next to an agent.
                # TODO: too specific, already covered by other feature???
                if d in self.others_xy:
                    return [0] * len(self.actions)

                if d == best_direction:
                    feature.append(1)
                else:
                    feature.append(0)

        return feature


    def feature12(self):
        """
        Reward placing a bomb that traps another agent located in a dead end.
        """
        feature = []

        for action in self.actions:        
            if action == 'BOMB' and self.bombs_left > 0:
                # Check if other agents are in direct vicinity.
                target_agent = []
                for d in self.directions.values():
                    if d != self.agent and d in self.others_xy:
                        target_agent.append(d)

                # Check if a targets are located in a dead end.
                CHECK_PLACE_BOMB = False
                if len(target_agent) != 0:
                    for target in target_agent:
                        if target in self.dead_ends:
                            CHECK_PLACE_BOMB = True
                            break

                if CHECK_PLACE_BOMB:
                    feature.append(1)
                else:
                    feature.append(0)
            else:
                feature.append(0)

        return feature


    def feature13(self):
        """
        Penalize the agent for moving into a dead end when it placed a bomb previously.
        """
        feature = []

        for action in self.actions:
            d = self.directions[action]

            # Only a move action can trap the agent into a dead end.
            if d == self.agent:
                feature.append(0)
            elif d in self.dead_ends and self.agent in self.bombs_xy:
                feature.append(1)
            else:
                feature.append(0)

        return feature

    
    def feature14(self):
        """
        Reward putting a bomb if it can kill an agent and we can scape
        """

        feature = []  # feature that we want get

        danger_zone = copy.deepcopy(self.danger_zone)
        my_bomb_zone = get_blast_coords(self.arena, self.x, self.y)
        danger_zone += my_bomb_zone 
        safe_zone = [(x, y) for x in range(1, 16) for y in range(1, 16)
                  if (self.arena[x, y] == 0)
                  and (x, y) not in danger_zone]
    
        best_path = look_for_targets_path(self.free_space, self.agent, safe_zone)

        for action in self.actions:
            if action == 'BOMB':
                CHECK_COND = False
                for others in self.others_xy:
                    if len(best_path) == 0:
                        return [0] * len(self.actions)
                    if (others in my_bomb_zone) and (best_path[-1] in safe_zone):
                        CHECK_COND = True
                        break
                if CHECK_COND and (self.bombs_left > 0):
                    feature.append(1)
                else:
                    feature.append(0)
            else:
                feature.append(0)
    
        #print("Feature 14: ", feature)
        return feature


    def feature15(self):
        """
        Reward going to an agent if he is nearby 
        """

        feature = []

        best_direction = look_for_targets(self.free_space, self.agent, self.others_xy)
        best_path = look_for_targets_path(self.free_space, self.agent, self.others_xy)

        for action in self.actions:
            if action == 'BOMB' or action == 'WAIT':
                # The feature rewards movement towards an agent. In particular,
                # placing a bomb or waiting is given no reward.
                feature.append(0)
            else:
                d = self.directions[action]
                if d in self.others_xy:
                    #print("Feature 15: ", [0] * len(self.actions))
                    return [0] * len(self.actions)
                
                if len(best_path) == 0:
                    #print("Feature 15: ", [0] * len(self.actions))
                    return [0] * len(self.actions)

                if (len(best_path) <= 3) and (best_path[-1] in self.others_xy):
                    if d == best_direction:
                        feature.append(1)
                    else:
                        feature.append(0)
                else:
                    feature.append(0)
        #print("Feature 15", feature)
        return feature
