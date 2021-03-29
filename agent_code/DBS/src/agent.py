import numpy as np
import torch
import torch.nn as nn
import pickle
import os

from .NN import DQNetwork
from .Memory import Memory


class OwnAgent:
    def __init__(self, input_dim, extra_dim, gamma, burnin, epsilon, learning_rate,
                 action_dim, batch_size, eps_min, eps_dec, replace,
                 save_dir, memory_size, random_chance, rotation_loss_factor,
                 priority_replay,  checkpoint, load_model, training,
                 not_load_eps, evaluate_model, debug_mode, use_checkpoint,
                 network_arch=None):
        self.input_dim = input_dim
        self.extra_dim = extra_dim
        self.gamma = gamma
        self.burnin = burnin
        self.learning_rate = learning_rate
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace = replace
        self.save_dir = save_dir
        self.save_step = 0
        self.memory_size = memory_size
        self.random_chance = random_chance
        self.rotation_loss_factor = rotation_loss_factor
        self.priority_replay = priority_replay
        self.checkpoint = checkpoint
        self.load_model = load_model
        self.is_training = training
        self.epsilon = epsilon if training else 0  # must be down here to overwrite epilon from loaded models
        self.not_load_eps = not_load_eps
        self.evaluate_model = evaluate_model

        self.action_space = list(range(self.action_dim))
        self.learn_step_counter = 0
        self.evaluate_step_counter = 0
        self.memory = Memory(self.memory_size, input_dim, extra_dim)

        self.q_eval = DQNetwork(self.input_dim, self.extra_dim, self.action_dim,
                                self.learning_rate, is_training=self.is_training,
                                network_arch=network_arch)
        self.q_next = DQNetwork(self.input_dim, self.extra_dim, self.action_dim,
                                self.learning_rate, network_arch=network_arch)

        self.rotation_loss_function = nn.MSELoss()

        self.epsilon = epsilon if training else 0  # must be down here to overwrite epilon from loaded models

        if self.load_model:
            self.load()

    @torch.no_grad()
    def choose_action(self, observation, extra_information):
        state = torch.tensor([observation], dtype=torch.float32, requires_grad=False)
        extra_information = torch.tensor([extra_information], dtype=torch.float32, requires_grad=False)
        q_values = self.q_eval(state, extra_information)
        actions = torch.nn.functional.softmax(q_values, dim=1).detach().numpy().flatten()

        prop = 0
        while prop <= 0.1:
            action = np.random.choice(np.arange(6), p=actions)
            prop = actions[action]

        return action

    @torch.no_grad()
    def store_transition(self, state, extra_info, action, reward, new_state,
                         new_extra_info, done):
        self.memory.store(state, extra_info, action, reward, new_state,
                          new_extra_info, done)

    @torch.no_grad()
    def sample_memory(self, batch_size):
        old_states, old_extra_infos, actions, rewards, new_states, new_extra_infos, dones = self.memory.sample(batch_size, self.priority_replay)

        return old_states, old_extra_infos, actions, rewards, new_states, new_extra_infos, dones

    @torch.no_grad()
    def replace_target_network(self):
        if self.learn_step_counter % self.replace == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())
            self.q_next.zero_grad()
            for p in self.q_next.parameters():
                p.requires_grad = False

    @torch.no_grad()
    def decrement_epsilon(self):
        self.epsilon = max(self.epsilon - self.eps_dec, self.eps_min)

    @torch.no_grad()
    def save(self):
        directory = f'{self.save_dir}/{self.checkpoint}'
        save_state = {
            'q_eval': self.q_eval.state_dict(),
            'q_next': self.q_next.state_dict(),
            'epsilon': self.epsilon,
            'save_step': self.save_step
        }
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(f'{directory}/{self.save_step}.ckpt', 'wb') as f:
            pickle.dump(save_state, f)

        print(f'saved: {directory}/{self.save_step}.ckpt')
        self.save_step += 1

    @torch.no_grad()
    def load(self, save_step=False, save_file=None):
        directory = f'{self.save_dir}/{self.checkpoint}'
        if type(save_step) == int:
            self.save_step = save_step
        else:
            for f in os.listdir(directory):
                name, ext = os.path.splitext(f)
                if ext == '.ckpt':
                    if int(name) > self.save_step:
                        self.save_step = int(name)

        if save_file is None:
            with open(f'{directory}/{self.save_step}.ckpt', 'rb') as f:
                save_state = pickle.load(f)
        else:
            with open(save_file, 'rb') as f:
                save_state = pickle.load(f)

        self.q_eval.load_state_dict(save_state['q_eval'])
        self.q_next.load_state_dict(save_state['q_next'])
        if not self.not_load_eps:
            self.epsilon = save_state['epsilon']
        self.save_step = save_state['save_step']

    @torch.no_grad()
    def act(self, state, extra):
        action = self.choose_action(state, extra)

        return action
