import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.parameters import parameters


class DQN(nn.Module):
    """Deep Q Network."""

    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.seed = torch.manual_seed(parameters["seed"])

        self.fc1 = nn.Linear(state_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 32)
        self.fc5 = nn.Linear(32, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return self.fc5(x)