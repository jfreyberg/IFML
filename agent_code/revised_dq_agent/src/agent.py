import numpy as np
import torch
import torch.nn as nn
import pickle
import os

from .NN import DQNetwork
from .Memory import Memory
from .StateProcessor import rotate_action, rotate_state


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
    def choose_action(self, observation, extra_information, rule_based_action=None):
        if np.random.random() > self.epsilon or not self.is_training:
            state = torch.tensor([observation], dtype=torch.float32, requires_grad=False)
            extra_information = torch.tensor([extra_information], dtype=torch.float32, requires_grad=False)
            q_values = self.q_eval(state, extra_information)
            actions = torch.nn.functional.softmax(q_values, dim=1).detach().numpy().flatten()
            # action = np.argmax(actions)

            prop = 0
            while prop <= 0.1:
                action = np.random.choice(np.arange(6), p=actions)
                prop = actions[action]
                if self.is_training:
                    break

            if not self.is_training and not self.evaluate_model:
                print(['RI', 'LE', 'BO', 'UP', 'DO', 'WA'])
                print([round(x) for x in q_values.detach().numpy().flatten()])
                print([round(x, 2) for x in actions])
                print(round(actions[action], 2))
        else:
            if np.random.random() > self.random_chance:
                action = rule_based_action
            else:
                action = np.random.choice(self.action_space)

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
            #print(f'loaded: {directory}/{self.save_step}.ckpt')
        else:
            with open(save_file, 'rb') as f:
                save_state = pickle.load(f)

        self.q_eval.load_state_dict(save_state['q_eval'])
        self.q_next.load_state_dict(save_state['q_next'])
        if not self.not_load_eps:
            self.epsilon = save_state['epsilon']
        self.save_step = save_state['save_step']

    @torch.no_grad()
    def act(self, state, extra, rule_based_action=None):
        action = self.choose_action(state, extra, rule_based_action)

        return action

    def learn(self):
        if len(self.memory) < max(self.batch_size, self.burnin):
            return

        self.q_eval.optimizer.zero_grad()
        self.replace_target_network()

        old_states, old_extra_infos, actions, rewards, new_states, new_extra_infos, dones = self.sample_memory(self.batch_size)

        indices = torch.arange(self.batch_size, dtype=torch.int64)

        # TODO: IMPORTANT: I changed new_states and old_states below.... AND NOW IT WORKS!
        q_pred = self.q_eval(old_states, old_extra_infos)[indices, actions]
        q_next = self.q_next(new_states, new_extra_infos).max(dim=1)[0]

        q_next[dones] = 0.0
        q_target = rewards + self.gamma*q_next

        loss = self.q_eval.loss(q_target, q_pred)
        if self.rotation_loss_factor != 0:
            rloss = self.rotation_loss_factor * self.compute_rotation_loss(old_states, old_extra_infos, indices, actions)
            loss += 0.1 * rloss

        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()

    def compute_rotation_loss(self, state, extra, indices, actions):

        q_pred_1 = self.q_eval(state, extra)[indices, actions]

        state, extra = rotate_state(state, extra)
        actions = rotate_action(actions)
        q_pred_2 = self.q_eval(state, extra)[indices, actions]

        state, extra = rotate_state(state, extra)
        actions = rotate_action(actions)
        q_pred_3 = self.q_eval(state, extra)[indices, actions]

        state, extra = rotate_state(state, extra)
        actions = rotate_action(actions)
        q_pred_4 = self.q_eval(state, extra)[indices, actions]

        loss = self.rotation_loss_function(q_pred_1, q_pred_2)
        loss += self.rotation_loss_function(q_pred_1, q_pred_3)
        loss += self.rotation_loss_function(q_pred_1, q_pred_4)

        return loss
