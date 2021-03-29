from argparse import ArgumentParser
import os

import events as e
import settings as s
from environment import GenericWorld
from fallbacks import tqdm
from agents import Agent
from items import Coin
import numpy as np
import random
from collections import namedtuple
WorldArgs = namedtuple("WorldArgs",
                       ["no_gui", "fps", "turn_based", "update_interval", "save_replay", "replay", "make_video", "continue_without_training", "log_dir"])


class TrainWorld(GenericWorld):
    def __init__(self, args: WorldArgs, agents):
        super().__init__(args)

        self.setup_agents(agents)
        self.new_round()

    def setup_agents(self, agents):
        # Add specified agents and start their subprocesses
        self.agents = []
        for agent_dir, train in agents:
            if list([d for d, t in agents]).count(agent_dir) > 1:
                name = agent_dir + '_' + str(list([a.code_name for a in self.agents]).count(agent_dir))
            else:
                name = agent_dir
            self.add_agent(agent_dir, name, train=train)

    def new_round(self):
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


def main(argv=None):
    play_parser = ArgumentParser()

    # Run arguments
    agent_group = play_parser.add_mutually_exclusive_group()
    agent_group.add_argument("--my-agent", type=str, help="Play agent of name ... against three rule_based_agents")
    agent_group.add_argument("--agents", type=str, nargs="+", default=["rule_based_agent"] * s.MAX_AGENTS, help="Explicitly set the agent names in the game")
    play_parser.add_argument("--train", default=0, type=int, choices=[0, 1, 2, 3, 4],
                             help="First … agents should be set to training mode")
    play_parser.add_argument("--continue-without-training", default=False, action="store_true")
    # play_parser.add_argument("--single-process", default=False, action="store_true")

    play_parser.add_argument("--n-rounds", type=int, default=-1, help="How many rounds to play")
    play_parser.add_argument("--n-steps", type=int, default=-1, help="How many steps to play")
    play_parser.add_argument("--reload-steps", type=int, default=10000, help="How many steps until reload")

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

    world = TrainWorld(args, agents)

    step_counter = 0
    round_counter = 0
    prev_load_counter = 0

    tqdm_iter_count = args.n_rounds if args.n_rounds != -1 else args.n_steps
    pbar = tqdm(total=tqdm_iter_count)
    # Run one or more games
    done = False
    while not done:
        if prev_load_counter + args.reload_steps <= step_counter:
            prev_load_counter = step_counter
            print('trying to update agents')
            try:
                save_step = world.agents[0].backend.runner.fake_self.agent.save_step - 1
                prev_cwd = os.getcwd()
                if save_step >= 0:
                    os.chdir(f'./agent_code/{world.agents[0].backend.code_name}/')
                    for a in world.agents[1:]:
                        try:
                            a.backend.runner.fake_self.agent.load(save_step)
                            print(f'reloaded agent {a.name} for step {save_step}')
                        except Exception as e:
                            print(f'{a.name} does not support loading!')
                            print(e)
            except Exception as e:
                print('first agent is not one of us!')
                print(e)
            finally:
                os.chdir(prev_cwd)
        if not world.running:
            world.ready_for_restart_flag.wait()
            world.ready_for_restart_flag.clear()
            world.new_round()

        # Main game loop
        round_finished = False
        while not round_finished:
            if world.running:
                if step_counter >= args.n_steps and args.n_steps != -1:
                    world.end_round()
                world.do_step('WAIT')
                step_counter += 1
                if args.n_rounds == -1:
                    pbar.update(1)
            else:
                round_finished = True
                round_counter += 1
                if args.n_steps == -1:
                    pbar.update(1)

        if step_counter >= args.n_steps and args.n_steps != -1:
            done = True
        if  round_counter >= args.n_rounds and args.n_rounds != -1:
            done = True

    world.end()

    print(f'steps trained: {step_counter}')
    print(f'rounds trained: {round_counter}')


if __name__ == '__main__':
    main()
