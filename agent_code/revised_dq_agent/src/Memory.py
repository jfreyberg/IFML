import numpy as np
import torch

class Memory():
    @torch.no_grad()
    def __init__(self, max_size, input_dim, extra_dim):
        self.size = max_size
        self.index = 0
        self.length = 0

        c, h, w = input_dim


        self.old_state_memory = torch.zeros((self.size, c, h, w), dtype=torch.float32, requires_grad=False)
        self.old_extra_info_memory = torch.zeros((self.size, extra_dim), dtype=torch.float32, requires_grad=False)
        self.new_state_memory = torch.zeros((self.size, c, h, w), dtype=torch.float32, requires_grad=False)
        self.new_extra_info_memory = torch.zeros((self.size, extra_dim), dtype=torch.float32, requires_grad=False)

        self.action_memory = torch.zeros(self.size, dtype=torch.int64, requires_grad=False)
        self.reward_memory = torch.zeros(self.size, dtype=torch.float32, requires_grad=False)
        self.done_memory = torch.zeros(self.size, dtype=torch.bool, requires_grad=False)

    @torch.no_grad()
    def __len__(self):
        return self.length

    @torch.no_grad()
    def store(self, old_state, old_extra_info, action, reward, new_state,
              new_extra_info, done):

        self.index = self.index % self.size

        self.old_state_memory[self.index] = torch.tensor(old_state, dtype=torch.float32, requires_grad=False)
        self.old_extra_info_memory[self.index] = torch.tensor(old_extra_info, dtype=torch.float32, requires_grad=False)
        self.action_memory[self.index] = torch.tensor(action, dtype=torch.int64, requires_grad=False)
        self.reward_memory[self.index] = torch.tensor(reward, dtype=torch.float32, requires_grad=False)
        self.done_memory[self.index] = torch.tensor(done, dtype=torch.bool, requires_grad=False)
        self.new_state_memory[self.index] = torch.tensor(new_state, dtype=torch.float32, requires_grad=False)
        self.new_extra_info_memory[self.index] = torch.tensor(new_extra_info, dtype=torch.float32, requires_grad=False)

        self.index += 1
        if self.length != self.size:
            self.length = max(self.index, self.length)

    @torch.no_grad()
    def sample(self, batch_size, weighted=False):
        selection = np.random.choice(self.length, batch_size, replace=False)

        old_states = self.old_state_memory[selection]
        old_extra_infos = self.old_extra_info_memory[selection]
        new_states = self.new_state_memory[selection]
        new_extra_infos = self.new_extra_info_memory[selection]
        actions = self.action_memory[selection]
        rewards = self.reward_memory[selection]
        dones = self.done_memory[selection]

        return old_states, old_extra_infos, actions, rewards, new_states, new_extra_infos, dones
