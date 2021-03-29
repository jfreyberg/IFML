import numpy as np
import os # required for training
import random

from agent_code.MR_Bombastic.feature_extraction import *
import settings as e
import settings as s

## TRAINING PARAMETERS

# Allow to check various combinations of training methods in parallel
# through multiple agent processes by taking values from the
# environment.
t_policy = os.environ.get('MRBOMBASTIC_POLICY')
t_policy_eps = os.environ.get('MRBOMBASTIC_POLICY_EPS')
t_weight_begin = os.environ.get('MRBOMBASTIC_WEIGHTS_BEGIN')

# Default values for training.
if t_policy == None: t_policy = 'greedy'
if t_policy_eps == None: t_policy_eps = 0.15
if t_weight_begin == None: t_weight_begin = 'trained'

# Construct unique id for naming of persistent data.
t_training_id = "{}_{}_{}".format(t_policy, t_policy_eps, t_weight_begin)
if t_policy == "greedy":
    t_training_id = "{}_{}".format(t_policy, t_weight_begin)

# Save weights to file
print("TRAINING ID:", t_training_id)
weights_file = "./agent_code/MR_Bombastic/models/{}_weights.npy".format(t_training_id)

## REWARDS AND POLICY

def initialize_weights(method):
    best_guess = np.asarray([1, 1.5, -7, -1, 4, -0.5, 1.5, 1, 0.5, 0.8, 0.5, 1, -1, 0.6, 0.6])
    trained_value = np.asarray([1, 2.36889437, -7.40026249, -1.34216494, 3.69025486,
                                -0.64679605, 2.34142438, 1.04329996, 0.50236224,
                                1.25573668, 0.47097264, 1.00000857, -1,
                                0.51097347, 0.22704274])
    if method == 'trained':
        return trained_value
    elif method == 'bestguess':
        return best_guess
    elif method == 'ones':
        return np.ones(len(best_guess))
    elif method == 'zero':
        return np.zeros(len(best_guess))
    elif method == 'random':
        return np.random.rand(1, len(best_guess)).flatten()
    else:
        return None


def policy_select_action(greedy_action, policy, policy_eps, game_ratio, logger=None):
    # Select a random action for use in epsilon-greedy or random-walk
    # exploration.
    random_action = np.random.choice(['RIGHT', 'LEFT', 'UP', 'DOWN', 'BOMB', 'WAIT'])

    if policy == 'greedy':
        logger.info('Pick greedy action %s', greedy_action)
        return greedy_action
    elif policy == 'epsgreedy':
        logger.info('Picking greedy action at probability %s', 1-policy_eps)
        return np.random.choice([greedy_action, random_action],
                                p=[1-policy_eps, policy_eps])
    elif policy == 'diminishing':
        policy_eps_dim = max(0.05, (1 - game_ratio) * policy_eps)
        logger.info('Picking greedy action at probability %s', 1-policy_eps_dim)
        return np.random.choice([greedy_action, random_action],
                                p=[1-policy_eps_dim, policy_eps_dim])
    else:
        return None


def new_reward(events):
    # An action is always punished with a small negative reward due to
    # time constraints on the game.
    reward = 0
    for event in events:
        if event == e.BOMB_DROPPED:
            reward += 1
        elif event == e.COIN_COLLECTED:
            reward += 200
        elif event == e.KILLED_SELF:
            reward -= 1000
        elif event == e.CRATE_DESTROYED:
            reward += 10
        elif event == e.COIN_FOUND:
            reward += 100
        elif event == e.KILLED_OPPONENT:
            reward += 700
        elif event == e.GOT_KILLED:
            reward -= 500
        elif event == e.SURVIVED_ROUND:
            reward += 200            
        elif event == e.INVALID_ACTION:
            reward -= 2

    return reward


## GAME METHODS

def setup(self):
    """Called once before a set of games to initialize data structures etc.

    The 'self' object passed to this method will be the same in all other
    callback methods. You can assign new properties (like bomb_history below)
    here or later on and they will be persistent even across multiple games.
    You can also use the self.logger object at any time to write to the log
    file for debugging (see https://docs.python.org/3.7/library/logging.html).
    """    
    np.random.seed()    

    # Reward for full game (10 rounds)
    self.accumulated_reward = 0    
    self.current_round = 1

    # Hyperparameters for learning methods.
    self.discount = 0.95
    self.alpha = 0.01

    # Values for diminished epsilon policy.
    self.accumulated_reward_generation = 0
    self.generation_current = 1
    self.generation_nrounds = 10
    self.generation_total = max(1, int(s.MAX_STEPS/self.generation_nrounds))
    self.game_ratio = self.generation_current/self.generation_total

    # Hyperparameters for (prioritized) experience replay.
    self.replay_buffer = []
    self.replay_buffer_max_steps = 200
    self.replay_buffer_update_after_nrounds = 10
    self.replay_buffer_sample_size = 50
    self.replay_buffer_every_ngenerations = 1

    # Load persistent data for exploitation.
    try:
        self.weights = np.load(weights_file)
        print("LOADED WEIGHTS", self.weights)
    except EnvironmentError:
        self.weights = initialize_weights(t_weight_begin)
        print("INITIALIZED WEIGHTS", self.weights)

    # List for storing y values (matplotlib)
    self.plot_rewards = []
    self.plot_weights = [self.weights]


def reward_update(self):
    """Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occured during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state. In
    contrast to act, this method has no time limit.

    @fvanmaele: We transform action consequences into something that Q-learning
    can use to learn the Q-function by giving in-game events a numerical reward.
    For learning the optimal behavior, the rewards of different objectives should
    be set carefully so that maximizing the obtained rewards results in the
    desired behavior.

    The following rewards have been chosen to clearly distinct between good and
    bad actions. Dying is represented by a very negative reward. The reward of
    killing a player is attributed to the player that actually placed the involved
    bomb. The rest of the rewards promote active behavior. No reward is given to
    finally winning the game (when all other players died).

    In order to maximize the total reward intake, the agent should learn not to
    die, and kill as many opponents and break most crates with its bombs.
    """
    # We need to perform at least one action before computing an
    # intermediary reward.
    if self.game_state['step'] == 1:
        return None

    reward = new_reward(self.events)
    self.logger.info('Given reward of %s', reward)
    self.accumulated_reward += reward

    # Extract features from the new ("next") game state.
    self.F = RLFeatureExtraction(self.game_state)

    # Get action of the previous step. The previous action was set in
    # the last act() call and is thus named 'next_action'.
    self.prev_action = self.next_action

    # Keep track of all experiences in the episode for later learning.
    if self.game_state['step'] <= self.replay_buffer_max_steps + 1:
        # The last tuple element defines if there was a transition to
        # a terminal state.
        self.replay_buffer.append((self.F_prev, self.prev_action, reward, self.F, False))


def act(self, game_state: dict):
    """Called each game step to determine the agent's next action.

    You can find out about the state of the game environment via self.game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    Set the action you wish to perform by assigning the relevant string to
    self.next_action. You can assign to this variable multiple times during
    your computations. If this method takes longer than the time limit specified
    in settings.py, execution is interrupted by the game and the current value
    of self.next_action will be used. The default value is 'WAIT'.
    """

    self.game_state = game_state
    # Update features to for the current ("previous") game state.
    self.F_prev = RLFeatureExtraction(self.game_state)

    # Multiple actions may give the same (optimal) reward. To avoid bias towards
    # a particular action, shuffle the greedy actions.
    Q_max, A_max = self.F_prev.max_q(self.weights)        
    random.shuffle(A_max)

    self.next_action = policy_select_action(A_max[0], t_policy, t_policy_eps,
                                            self.game_ratio, self.logger)

    return self.next_action


def end_of_episode(self):
    """Called at the end of each game to hand out final rewards and do training.

    This is similar to reward_update, except it is only called at the end of a
    game. self.events will contain all events that occured during your agent's
    final step. You should place your actual learning code in this method.
    """    
    # Add experience for transition to terminal state.
    self.F = RLFeatureExtraction(self.game_state)
    self.prev_action = self.next_action
    reward = new_reward(self.events)

    self.replay_buffer.append((self.F_prev, self.prev_action, reward, self.F, True))
    self.logger.info('Given end reward of %s', reward)
    self.accumulated_reward += reward
    self.accumulated_reward_generation += self.accumulated_reward
    self.accumulated_reward = 0
    self.current_round += 1

    # Begin experience replay
    if (self.current_round % self.generation_nrounds) == 0:
        experience_mini_batch = random.sample(self.replay_buffer, self.replay_buffer_sample_size)
        print("REPLAY BUFFER SIZE:", len(self.replay_buffer))
        print("MINI BATCH SIZE:", len(experience_mini_batch))
        self.replay_buffer_sample_size += 20 # set as variable

        # Reset replay buffer
        if self.generation_current % self.replay_buffer_every_ngenerations == 0:
            print("RESETTING REPLAY BUFFER")
            self.replay_buffer = []

        weights_batch_update = np.zeros(len(self.weights))
        for X, A, R, Y, terminal in experience_mini_batch:
            # # Compute maximum Q value and corresponding action.
            # X_A = X.state_action(A)
            # Q_max, A_max = Y.max_q(self.weights)

            # # Do not change the reward if the state is terminal.
            # if terminal == True:
            #     V = R
            # else:
            #     V = R + self.discount * Q_max
            if A != None:
                X_A = X.state_action(A)
                Q_max, A_max = Y.max_q(self.weights)
                TD_error = R + (self.discount * Q_max) - np.dot(X_A, self.weights)

                weights_batch_update = weights_batch_update + self.alpha/self.replay_buffer_sample_size * TD_error * X_A
                # incremental gradient descent
                #self.weights = self.weights + self.alpha / self.replay_buffer_sample_size * TD_error * X_A

        self.weights = self.weights + weights_batch_update
        self.plot_weights.append(self.weights)        
        #print("TOTAL REWARD FOR GENERATION {}: {}".format(self.generation_current,
                                                         # self.accumulated_reward_generation))
        #print("WEIGHTS FOR NEXT GENERATION:", self.weights, sep=" ")

        self.generation_current += 1
        self.game_ratio = self.generation_current/self.generation_total

        self.plot_rewards.append(self.accumulated_reward_generation)
        np.save(t_training_id, self.plot_rewards)
        self.accumulated_reward_generation = 0

    np.save(weights_file, self.weights)
    
