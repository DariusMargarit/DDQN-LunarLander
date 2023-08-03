import torch
import random
import numpy as np
from collections import deque, namedtuple


class MemoryBuffer(object):
    def __init__(self, max_size, seed=True):
        self.memory_size = max_size
        self.trans_counter = 0
        self.index = 0
        self.buffer = deque(maxlen=self.memory_size)
        self.transition = namedtuple("Transition", field_names=["state", "action", "reward", "new_state", "terminal"])
        if seed:
            random.seed(12)

    def save(self, state, action, reward, new_state, terminal):
        t = self.transition(state, action, reward, new_state, terminal)
        self.buffer.append(t)
        self.trans_counter = (self.trans_counter + 1) % self.memory_size

    def random_sample(self, batch_size):
        assert len(self.buffer) >= batch_size
        transitions = random.sample(self.buffer, k=batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in transitions if e is not None])).float()
        actions = torch.from_numpy(np.vstack([e.action for e in transitions if e is not None])).long()
        rewards = torch.from_numpy(np.vstack([e.reward for e in transitions if e is not None])).float()
        new_states = torch.from_numpy(np.vstack([e.new_state for e in transitions if e is not None])).float()
        terminals = torch.from_numpy(
            np.vstack([e.terminal for e in transitions if e is not None]).astype(np.uint8)).float()

        return states, actions, rewards, new_states, terminals
