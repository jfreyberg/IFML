import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque
from .src.parameters import get_parameters, INVERSE_OPTIONS
import gc
from .src.helpers import manhatten_distance
from pprint import pprint

N = 4
MOVEMENT = ['steps', 'WAITED', 'MOVED_UP', 'MOVED_DOWN', 'MOVED_LEFT', 'MOVED_RIGHT', 'INVALID_ACTION']
EVENTS = ['CRATE_DESTROYED', 'COIN_COLLECTED', 'KILLED_SELF', 'KILLED_OPPONENT', 'OPPONENT_ELIMINATED',
          'BOMB_DROPPED', 'COIN_FOUND', 'SURVIVED_ROUND']
fig, axs = plt.subplots(4, figsize=(15, 15))
fig.tight_layout(pad=5)


def setup_training(self):

    np.random.seed()
    self.reward_history = []
    self.exploration_rate_history = []
    self.event_history = defaultdict(list)
    self.event_counter = defaultdict(int)
    self.move_history = deque(maxlen=2)

    _, self.REWARDS, _ = get_parameters()


def compute_reward(self, events, old_state):

    # pprint(old_state)

    total_reward = self.REWARDS.get('DEFAULT', 0)

    for event, reward in self.REWARDS.items():
        if event in events:
            total_reward += reward

    movement = None
    if 'MOVED_LEFT' in events:
        movement = (-1, 0)
    elif 'MOVED_RIGHT' in events:
        movement = (1, 0)
    elif 'MOVED_UP' in events:
        movement = (0, -1)
    elif 'MOVED_DOWN' in events:
        movement = (0, 1)

    enemy_pos = [x[3] for x in old_state['others']]
    self_pos = old_state['self'][3]
    if 'BOMB_DROPPED' in events:

        # rate bomb quality

        # when an enemy is next to us, this was a good bomb!

        for enemy_p in enemy_pos:
            if manhatten_distance(enemy_p, self_pos) < 2:
                reward += self.REWARDS.get('ENEMY_VERY_CLOSE_BOMB', 0)
            elif manhatten_distance(enemy_p, self_pos) < 3:
                reward += self.REWARDS.get('ENEMY_CLOSE_BOMB', 0)

    # when an enemy is near we give a reward to encourage hunting behaviour

    for enemy_p in enemy_pos:
        if manhatten_distance(enemy_p, self_pos) < 4:
            reward += self.REWARDS.get('ENEMY_CLOSE', 0)

    if movement is not None:
        self.move_history.append(movement)
        if len(self.move_history) > 1:
            distance = sum([sum(x) for x in zip(*list(self.move_history)[-2:])])
            total_reward += self.REWARDS.get('ACTIVITY_BONUS', 0) * distance
            if abs(distance) < 1:
                total_reward -= self.REWARDS.get('LAZYINESS_PENALTY', 0)

    return total_reward


def step_done(self, old_state, new_state, action, events, done):

    reward = compute_reward(self, events, old_state)

    old_state, old_extra_info = self.state_processor.get_feature_space(old_state)
    new_state, new_extra_info = self.state_processor.get_feature_space(new_state)

    self.agent.store_transition(old_state, old_extra_info, INVERSE_OPTIONS[action], reward, new_state, new_extra_info, done)

    self.agent.learn()

    self.reward_history.append(reward)
    self.exploration_rate_history.append(self.agent.epsilon)

    for event in events:
        self.event_counter[event] += 1
    self.event_counter['steps'] += 1

    if self.agent.learn_step_counter % 10000 == 9999:
        reward_colors = ['cornflowerblue', 'midnightblue', 'crimson']
        for i, n in enumerate([100, 1000, 100000]):

            axs[0].plot(running_mean(self.reward_history, n), label=f'running mean: {n}', color=reward_colors[i])

        axs[0].set(xlabel='step', ylabel='reward', title='Rewards')
        axs[0].legend(loc='upper left')
        axs[0].set_xlim(left=0)
        axs[0].grid()

        axs[1].set(xlabel='round', ylabel='count', title='Movement counts')
        for e in MOVEMENT:
            axs[1].plot(running_mean(self.event_history[e], 1000), label=e)
        axs[1].set_ylim(bottom=0)
        axs[1].set_xlim(left=0)
        axs[1].grid()
        axs[1].legend(ncol=len(MOVEMENT), loc='upper left')

        axs[2].set(xlabel='round', ylabel='count', title='Event counts')
        for e in EVENTS:
            axs[2].plot(running_mean(self.event_history[e], 1000), label=e)
        axs[2].set_ylim(bottom=0)
        axs[2].set_xlim(left=0)
        axs[2].grid()
        axs[2].legend(ncol=6, loc='upper left')

        axs[3].set(xlabel='step', ylabel='exploration rate', title='Exploration rate')
        axs[3].plot(self.exploration_rate_history)
        axs[3].set_ylim(0, 1)
        axs[3].set_xlim(left=0)
        axs[3].grid()

        fig.savefig(f"reward_{self.agent.checkpoint}.png")

        axs[0].clear()
        axs[1].clear()
        axs[2].clear()
        axs[3].clear()

        gc.collect()

        self.agent.save()


def get_game_ratios(history, e1, e2):
    return list(np.array(history[e1][1:])/np.array(history[e2][1:]))


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    done = False
    if old_game_state is None or new_game_state is None:
        return

    step_done(self, old_game_state, new_game_state, self_action, events, done)


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def end_of_round(self, last_game_state, last_action, events):
    done = True
    if last_game_state is None:
        return

    step_done(self, last_game_state, last_game_state, last_action, events, done)

    for e, r in self.REWARDS.items():
        self.event_history[e].append(self.event_counter[e])
    self.event_history['steps'].append(self.event_counter['steps'])

    # print(self.event_history)

    self.event_counter.clear()
