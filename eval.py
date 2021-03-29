from argparse import ArgumentParser
from collections import defaultdict, deque
import os
import json
import matplotlib.pyplot as plt
import matplotlib.ticker

import events as e
import settings as s
from environment import GenericWorld
from fallbacks import tqdm
from agents import Agent
from items import Coin
import numpy as np
import random
from collections import namedtuple
from items import Bomb
WorldArgs = namedtuple("WorldArgs",
                       ["no_gui", "fps", "turn_based", "update_interval", "save_replay", "replay", "make_video", "continue_without_training", "log_dir"])

seed = 0

# read rewards from JSON
CONF = os.environ.get('AGENT_CONF', 'agent_code/revised_dq_agent/confs/default.json')
conf = json.load(open(CONF, 'r'))
REWARDS = conf['rewards']


class EvalWorld(GenericWorld):
    def __init__(self, args: WorldArgs, agents):
        super().__init__(args)

        self.setup_agents(agents)
        self.new_round()
        self.eval_actions = []

    def setup_agents(self, agents):
        # Add specified agents and start their subprocesses
        self.agents = []
        for agent_dir, train in agents:
            if list([d for d, t in agents]).count(agent_dir) > 1:
                name = agent_dir + '_' + str(list([a.code_name for a in self.agents]).count(agent_dir))
            else:
                name = agent_dir
            self.colors.append('blue')
            self.add_agent(agent_dir, name, train=train)

    def new_round(self):
        global seed
        random.seed(seed)
        np.random.seed(seed)

        self.round += 1

        # Bookkeeping
        self.step = 0
        self.active_agents = []
        self.bombs = []
        self.explosions = []

        # Arena with wall and crate layout
        self.arena = (np.random.rand(s.COLS, s.ROWS) < s.CRATE_DENSITY).astype(int)
        self.arena[:1, :] = -1
        self.arena[-1:, :] = -1
        self.arena[:, :1] = -1
        self.arena[:, -1:] = -1
        for x in range(s.COLS):
            for y in range(s.ROWS):
                if (x + 1) * (y + 1) % 2 == 1:
                    self.arena[x, y] = -1

        # Starting positions
        start_positions = [(1, 1), (1, s.ROWS - 2), (s.COLS - 2, 1), (s.COLS - 2, s.ROWS - 2)]
        random.shuffle(start_positions)
        for (x, y) in start_positions:
            for (xx, yy) in [(x, y), (x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:
                if self.arena[xx, yy] == 1:
                    self.arena[xx, yy] = 0

        # Distribute coins evenly
        self.coins = []
        """coin_pattern = np.array([
            [1, 1, 1],
            [0, 0, 1],
        ])
        coins = np.zeros_like(self.arena)
        for x in range(1, s.COLS - 2, coin_pattern.shape[0]):
            for i in range(coin_pattern.shape[0]):
                for j in range(coin_pattern.shape[1]):
                    if coin_pattern[i, j] == 1:
                        self.coins.append(Coin((x + i, x + j), self.arena[x+i,x+j] == 0))
                        coins[x + i, x + j] += 1"""
        for i in range(3):
            for j in range(3):
                for _ in range(1):
                    n_crates = (self.arena[1 + 5 * i:6 + 5 * i, 1 + 5 * j:6 + 5 * j] == 1).sum()
                    while True:
                        x, y = np.random.randint(1 + 5 * i, 6 + 5 * i), np.random.randint(1 + 5 * j, 6 + 5 * j)
                        if n_crates == 0 and self.arena[x, y] == 0:
                            self.coins.append(Coin((x, y)))
                            self.coins[-1].collectable = True
                            break
                        elif self.arena[x, y] == 1:
                            self.coins.append(Coin((x, y)))
                            break

        # Reset agents and distribute starting positions
        for agent in self.agents:
            agent.start_round()
            self.active_agents.append(agent)
            agent.x, agent.y = start_positions.pop()

        self.replay = {
            'round': self.round,
            'arena': np.array(self.arena),
            'coins': [c.get_state() for c in self.coins],
            'agents': [a.get_state() for a in self.agents],
            'actions': dict([(a.name, []) for a in self.agents]),
            'permutations': []
        }

        self.running = True

    def get_state_for_agent(self, agent: Agent):
        state = {
            'round': self.round,
            'step': self.step,
            'field': np.array(self.arena),
            'self': agent.get_state(),
            'others': [other.get_state() for other in self.active_agents if other is not agent],
            'bombs': [bomb.get_state() for bomb in self.bombs],
            'coins': [coin.get_state() for coin in self.coins if coin.collectable],
            'user_input': self.user_input,
        }

        explosion_map = np.zeros(self.arena.shape)
        for exp in self.explosions:
            for (x, y) in exp.blast_coords:
                explosion_map[x, y] = max(explosion_map[x, y], exp.timer)
        state['explosion_map'] = explosion_map

        return state

    def send_training_events(self):
        # Send events to all agents that expect them, then reset and wait for them
        for a in self.agents:
            if a.train:
                if not a.dead:
                    a.process_game_events(self.get_state_for_agent(a))
                for enemy in self.active_agents:
                    if enemy is not a:
                        pass
                        # a.process_enemy_game_events(self.get_state_for_agent(enemy), enemy)
        for a in self.agents:
            if a.train:
                if not a.dead:
                    a.wait_for_game_event_processing()
                for enemy in self.active_agents:
                    if enemy is not a:
                        pass
                        # a.wait_for_enemy_game_event_processing()
        for a in self.active_agents:
            a.store_game_state(self.get_state_for_agent(a))
            a.reset_game_events()

    def perform_agent_action(self, agent: Agent, action: str):
        # Perform the specified action if possible, wait otherwise
        if action == 'UP' and self.tile_is_free(agent.x, agent.y - 1):
            agent.y -= 1
            agent.add_event(e.MOVED_UP)
        elif action == 'DOWN' and self.tile_is_free(agent.x, agent.y + 1):
            agent.y += 1
            agent.add_event(e.MOVED_DOWN)
        elif action == 'LEFT' and self.tile_is_free(agent.x - 1, agent.y):
            agent.x -= 1
            agent.add_event(e.MOVED_LEFT)
        elif action == 'RIGHT' and self.tile_is_free(agent.x + 1, agent.y):
            agent.x += 1
            agent.add_event(e.MOVED_RIGHT)
        elif action == 'BOMB' and agent.bombs_left:
            self.logger.info(f'Agent <{agent.name}> drops bomb at {(agent.x, agent.y)}')
            self.bombs.append(Bomb((agent.x, agent.y), agent, s.BOMB_TIMER, s.BOMB_POWER, agent.color, custom_sprite=agent.bomb_sprite))
            agent.bombs_left = False
            agent.add_event(e.BOMB_DROPPED)
        elif action == 'WAIT':
            agent.add_event(e.WAITED)
        else:
            agent.add_event(e.INVALID_ACTION)

    def poll_and_run_agents(self):
        self.send_training_events()

        # Tell agents to act
        for a in self.active_agents:
            a.act(self.get_state_for_agent(a))

        # Give agents time to decide
        perm = np.random.permutation(len(self.active_agents))
        for i in perm:
            a = self.active_agents[i]
            action, think_time = a.wait_for_act()
            self.perform_agent_action(a, action)

    def end_round(self):
        assert self.running, "End of round requested while not running"
        super().end_round()

        # Clean up survivors
        for a in self.active_agents:
            a.add_event(e.SURVIVED_ROUND)

        # Send final event to agents that expect them
        for a in self.agents:
            if a.train:
                a.round_ended()

        # Mark round as ended
        self.running = False

        self.ready_for_restart_flag.set()

    def end(self):
        if self.running:
            self.end_round()


def compute_reward(move_history, evs):
    total_reward = REWARDS.get('DEFAULT', 0)

    for event, reward in REWARDS.items():
        if event in evs:
            total_reward += reward

    movement = None
    if 'MOVED_LEFT' in evs:
        movement = (-1, 0)
    elif 'MOVED_RIGHT' in evs:
        movement = (1, 0)
    elif 'MOVED_UP' in evs:
        movement = (0, -1)
    elif 'MOVED_DOWN' in evs:
        movement = (0, 1)

    if movement is not None:
        move_history.append(movement)
        if len(move_history) > 1:
            distance = sum([sum(x) for x in zip(*list(move_history)[-2:])])
            total_reward += REWARDS.get('ACTIVITY_BONUS', 50) * distance
            if abs(distance) < 1:
                total_reward -= REWARDS.get('LAZYINESS_PENALTY', 50)

    return move_history, total_reward


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def main(argv=None):

    # valid events
    EVENTS = ['MOVED_LEFT', 'MOVED_RIGHT', 'MOVED_UP', 'MOVED_DOWN', 'WAITED',
              'INVALID_ACTION', 'BOMB_DROPPED', 'BOMB_EXPLODED',
              'CRATE_DESTROYED', 'COIN_FOUND', 'COIN_COLLECTED',
              'KILLED_OPPONENT', 'KILLED_SELF', 'GOT_KILLED',
              'OPPONENT_ELIMINATED', 'SURVIVED_ROUND']
    MOVEMENT = ['MOVED_LEFT', 'MOVED_RIGHT', 'MOVED_UP', 'MOVED_DOWN', 'WAITED',
                'INVALID_ACTION', 'BOMB_DROPPED']
    EVENTS_NO_MVNT = ['CRATE_DESTROYED', 'COIN_COLLECTED', 'KILLED_SELF', 'KILLED_OPPONENT', 'OPPONENT_ELIMINATED',
                      'BOMB_DROPPED', 'COIN_FOUND', 'SURVIVED_ROUND']

    # interesting stuff for plotting
    all_game_rewards_mean = []  # has shape (#loaded checkpoints, #games per checkpoint)
    all_scores = []  # has shape (#loaded checkpoints, #games per checkpoint)
    all_steps_alive = []  # has shape (#loaded checkpoints, #games per checkpoint)
    all_rewards = []  # has shape (#loaded checkpoints, #games per checkpoint, #steps per game)
    all_rewards_steps = []  # has shape (#checkpoints*games*steps)

    all_events = defaultdict(list)  # dict containing eventcounts in shape (#loaded checkpoints, #games per checkpoint)
    all_ratios = defaultdict(list)  # dict containing ratios computed per chechkpoint, i.e. (#loaded checkpoints, )

    epsilons = []

    play_parser = ArgumentParser()

    # Run arguments
    agent_group = play_parser.add_mutually_exclusive_group()
    agent_group.add_argument("--my-agent", type=str, help="Play agent of name ... against three rule_based_agents")
    agent_group.add_argument("--agents", type=str, nargs="+", default=["rule_based_agent"] * s.MAX_AGENTS, help="Explicitly set the agent names in the game")
    play_parser.add_argument("--save-steps", type=int, nargs="+", default=[0] * s.MAX_AGENTS, help="Explicitly set the save point for the agent")
    play_parser.add_argument("--train", default=0, type=int, choices=[0, 1, 2, 3, 4], help="First … agents should be set to training mode")
    play_parser.add_argument("--continue-without-training", default=False, action="store_true")
    play_parser.add_argument("--eval-start", default=0, type=int, help="first eval step")
    play_parser.add_argument("--eval-stop", default=0, type=int, help="last eval step")
    play_parser.add_argument("--eval-step", default=1, type=int, help="eval step")
    play_parser.add_argument("--games", default=10, type=int, help="number of games to evaluate per checkpoint")
    play_parser.add_argument("--name", default='', type=str, help="name of eval plots")
    # play_parser.add_argument("--single-process", default=False, action="store_true")

    args = play_parser.parse_args(argv)
    args.no_gui = True
    args.make_video = False
    args.log_dir = '/tmp'

    # Initialize environment and agents
    agents = []
    if args.train == 0 and not args.continue_without_training:
        args.continue_without_training = True
    if args.my_agent:
        agents.append((args.my_agent, len(agents) < args.train))
        args.agents = ["rule_based_agent"] * (s.MAX_AGENTS - 1)
    for agent_name in args.agents:
        agents.append((agent_name, len(agents) < args.train))

    fig, axs = plt.subplots(4, figsize=(15, 15))
    fig.tight_layout(pad=5)

    fig2, ax = plt.subplots(4, figsize=(15, 15))
    fig2.tight_layout(pad=6)

    ax_1_2 = ax[1].twinx()

    eval_name = ''

    for save_step_iter in tqdm(list(range(args.eval_start, args.eval_stop+1, args.eval_step))):
        global seed
        seed = 0

        world = EvalWorld(args, agents)

        for i, a in enumerate(world.agents):
            args.save_steps[i] = save_step_iter
            if args.save_steps[i] >= 0:
                prev_cwd = os.getcwd()
                try:
                    if a.backend.runner.fake_self.agent.save_step < save_step_iter:
                        print("last checkpoint reached -> done")
                        return
                    os.chdir(f'./agent_code/{world.agents[0].backend.code_name}/')
                    a.backend.runner.fake_self.agent.load(args.save_steps[i])
                    a.backend.runner.fake_self.agent.evaluate_model = True
                except Exception as e:
                    print(f'{a.name} does not support loading!')
                    print(e)
                finally:
                    os.chdir(prev_cwd)

        try:
            if not args.name and not eval_name:
                eval_name = '_' + world.agents[0].backend.runner.fake_self.agent.checkpoint
                print(f'using the name {eval_name[1:]}')
            elif not eval_name:
                eval_name = '_' + args.name
                print(f'using the name {eval_name[1:]}')
            epsilons.append(world.agents[0].backend.runner.fake_self.agent.epsilon)
        except:
            epsilons.append(0)

        score = []

        event_counter = defaultdict(list)
        step_counter = []
        reward_history = []
        move_history = deque(maxlen=2)

        for round_cnt in range(args.games):
            seed = round_cnt+1

            score.append(0)
            step_counter.append(0)
            reward_history.append([0])
            for ev in EVENTS:
                event_counter[ev].append(0)

            if not world.running:
                world.ready_for_restart_flag.wait()
                world.ready_for_restart_flag.clear()
                world.new_round()

            # Main game loop
            round_finished = False
            dead = False
            while not round_finished:
                if world.running:
                    world.do_step('WAIT')
                    if not dead:
                        step_counter[-1] += 1
                        for ev in world.agents[0].events:
                            if ev == 'COIN_COLLECTED':
                                score[-1] += s.REWARD_COIN
                            elif ev == 'KILLED_OPPONENT':
                                score[-1] += s.REWARD_KILL
                            event_counter[ev][-1] += 1
                        move_history, reward = compute_reward(move_history, world.agents[0].events)
                        reward_history[-1].append(reward)
                        all_rewards_steps.append(reward)
                    dead = world.agents[0].dead
                    # if not world.agents[0].dead:
                    #    print(1, world.agents[0].events)
                    # if not world.agents[1].dead:
                    #    print(2, world.agents[1].events)
                    # if not world.agents[2].dead:
                    #    print(3, world.agents[2].events)
                    # if not world.agents[3].dead:
                    #    print(4, world.agents[3].events)
                else:
                    round_finished = True

            world.end()

        # general plotting values

        #print(f'score: {sum(score)}')
        all_scores.append(score)
        #print(f'steps alive: {sum(step_counter)}')
        all_steps_alive.append(step_counter)
        for ev in EVENTS:
            #print(f'{e}: {sum(event_counter[e])}')
            all_events[ev].append(event_counter[ev])

        try:
            crate_bomb_ratio = sum(event_counter["CRATE_DESTROYED"])/sum(event_counter["BOMB_DROPPED"])
        except ZeroDivisionError:
            crate_bomb_ratio = 0

        all_ratios['crate-bomb-ratio'].append(crate_bomb_ratio)
        #print(f'crate-bomb-ratio: {round(crate_bomb_ratio, 2)}')

        game_rewards_mean = [np.mean(x) for x in reward_history]
        all_rewards.append(reward_history)

        reward_colors = ['cornflowerblue', 'midnightblue', 'crimson']
        all_game_rewards_mean.append([np.mean(x) for x in reward_history])

        if len(all_steps_alive) > 1:
            #############################
            #######     plots   #########
            #############################
            '''
            for i, n in enumerate([1, 5, 50]):
                if i == 0:
                    axs[0].plot(all_rewards_steps, label=f'reward per step', color=reward_colors[i])
                else:
                    axs[0].plot(running_mean(all_rewards_steps, n), label=f'running mean: {n}', color=reward_colors[i])


            axs[0].set(xlabel='steps', ylabel='reward', title='Rewards')
            axs[0].legend(loc='upper left')
            axs[0].set_xlim(left=0)
            axs[0].grid()

            axs[1].set(xlabel='checkpoint', ylabel='mean reward', title='Mean reward and movements per checkpoint')
            axs[1].plot(game_rewards_mean[1:-1], label='reward', linewidth=2.0)
            for e in MOVEMENT:   
                axs[1].plot(running_mean(all_events[e][1:], 2), label=e)
            axs[1].set_xlim(left=0)
            axs[1].grid()
            axs[1].legend(ncol=len(MOVEMENT), loc='upper left')

            axs[2].set(xlabel='checkpoint', ylabel='count', title='Mean reward and event counts per checkpoint')
            axs[2].plot(game_rewards_mean[1:-1], label='reward', linewidth=2.0)
            for e in EVENTS_NO_MVNT:
                axs[2].plot(running_mean(all_events[e][1:], 2), label=e)
            axs[2].set_xlim(left=0)
            axs[2].grid()
            axs[2].legend(ncol=6, loc='upper left')
            fig.savefig(f"agent_code/revised_dq_agent/eval/{world.agents[0].name}_general.png")
            axs[0].clear()
            axs[1].clear()
            axs[2].clear()
            '''

            # same per checkpoint

            ax[0].set(xlabel='checkpoint', ylabel='steps', title="Steps survived per checkpoint")
            ax[0].plot(np.mean(np.array(all_steps_alive), axis=1), label='mean steps', color='dimgrey', alpha=0.6)
            ax[0].plot(running_mean(np.mean(np.array(all_steps_alive), axis=1), 10), label='running mean (10)', color='dimgrey')
            #ax[0].plot(np.array(all_ratios['crate-bomb-ratio']*args.games), label='crate/bomb')
            # print(np.array(all_ratios['crate-bomb-ratio']))
            ax[0].legend(loc='upper left')
            ax[0].set_xlim(0, len(all_steps_alive)-1)
            ax[0].grid()

            ax[1].set(xlabel='checkpoint', title="Score and reward per checkpoint")
            ax[1].plot(np.mean(np.array(all_game_rewards_mean), axis=1), label='rewards', color='navy', alpha=0.6)
            ax[1].plot(running_mean(np.mean(np.array(all_game_rewards_mean), axis=1), 10), label='running mean (10)', color='navy')
            ax[1].set_ylabel('reward', color='navy')
            ax[1].set_xlim(0, len(all_steps_alive)-1)
            ax[1].grid()
            ax_1_2.plot(np.mean(np.array(all_scores), axis=1), color='crimson', alpha=0.6)
            ax_1_2.set_ylabel('score', color='crimson')
            ax_1_2.plot(running_mean(np.mean(np.array(all_scores), axis=1), 10), color='crimson')
            ax_1_2.set_xlim(0, len(all_steps_alive)-1)

            # adjust left scale to grid
            l = ax[1].get_ylim()
            l2 = ax_1_2.get_ylim()
            def f(x): return l2[0]+(x-l[0])/(l[1]-l[0])*(l2[1]-l2[0])
            ticks = f(ax[1].get_yticks())
            ax_1_2.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))

            #align.yaxes(ax[1], 0, ax_1_2, 0, 0.5)

            y = []
            #['MOVED_LEFT', 'MOVED_RIGHT', 'MOVED_UP', 'MOVED_DOWN', 'WAITED','INVALID_ACTION', 'BOMB_DROPPED']
            color_moves = ['dodgerblue', 'deepskyblue', 'limegreen', 'seagreen', 'slategrey', 'coral', 'orangered']
            for ev in MOVEMENT:
                y.append(np.mean(np.array(all_events[ev]), axis=1)/np.mean(np.array(all_steps_alive), axis=1))
            ax[2].stackplot(range(len(y[0])), *y, labels=MOVEMENT, colors=color_moves)
            ax[2].legend(bbox_to_anchor=(0.5, 1.15), loc='upper center', ncol=len(MOVEMENT))
            ax[2].grid()
            ax[2].set_xlim(0, len(all_steps_alive)-1)
            ax[2].set_ylim(0, 1)
            '''

            ax[2].set(xlabel='checkpoint', ylabel='mean reward', title='Mean events per game')
            for e in EVENTS_NO_MVNT:   
                ax[2].plot(running_mean(np.mean(np.array(all_events[e]), axis=0),2), label=e)
            ax[2].set_xlim(left=0)
            ax[2].grid()
            ax[2].legend(ncol=7, loc='upper left')

            ax[3].set(xlabel='checkpoint', ylabel='mean reward', title='Mean movements per game')
            for e in MOVEMENT:  
                ax[1].plot(running_mean(np.mean(np.array(all_events[e]), axis=0), 2), label=e)
            ax[3].set_xlim(left=0)
            ax[3].grid()
            ax[3].legend(ncol=len(MOVEMENT), loc='upper left')
            '''

            ax[3].set(xlabel='checkpoint', title="Epsilon per checkpoint", ylabel='epsilon')
            ax[3].plot(epsilons, color='cornflowerblue', linewidth=2.0)
            ax[3].set_xlim(0, len(all_steps_alive)-1)
            ax[3].set_ylim(0, 1)
            ax[3].grid()

            fig2.savefig(f"agent_code/revised_dq_agent/eval/{world.agents[0].name}{eval_name}_checkpoint.png")
            ax[0].clear()
            ax[1].clear()
            ax[2].clear()
            ax[3].clear()
            ax_1_2.clear()


if __name__ == '__main__':
    main()
