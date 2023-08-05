import gymnasium as gym
import random
import numpy as np

from utils.plots import *
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

parameters = {
    "episodes": 600,
    "max_steps": 1000,
    "epsilon": 1,
    "epsilon_decay": 0.99,
    "epsilon_min": 0.01,
    "update_freq": 4,
    "gamma": 0.999,
    "seed": 12,
    "lr": 0.0005,
    "tau": 0.001,
    "buffer_size": 100_000,
    "batch_size": 64,
    "reward_target_mean": 2000,
    "render": False
}

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        random.seed(parameters["seed"])

    def save(self, *args):
        """Saves a transition"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


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


class Agent():

    def __init__(self, env_name):
        if parameters["render"]:
            self.env = gym.make(env_name, render_mode="human")
        else:
            self.env = gym.make(env_name)
        random.seed(parameters["seed"])
        state_size = self.env.observation_space.shape[0]
        action_size = self.env.action_space.n

        self.policy_net = DQN(state_size, action_size)
        self.target_net = DQN(state_size, action_size)

        # self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=parameters["lr"])
        # or self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.memory = ReplayMemory(parameters["buffer_size"])
        self.steps = 0

    def act(self, state, epsilon):

        state = torch.from_numpy(state).float().unsqueeze(0)
        self.policy_net.eval()
        self.policy_net.train()

        # epsilon-greedy
        if random.random() > epsilon:
            with torch.no_grad():
                # return self.policy_net(state).max(1)[1].view(1, 1)
                return np.argmax(self.policy_net(state).cpu().data.numpy())
        else:
            return self.env.action_space.sample()

    def learn(self, transitions, gamma):
        batch = Transition(*zip(*transitions))
        state_batch = torch.from_numpy(
            np.vstack([x for x in batch.state if x is not None])).float()
        action_batch = torch.from_numpy(
            np.vstack([x for x in batch.action if x is not None])).long()
        reward_batch = torch.from_numpy(
            np.vstack([x for x in batch.reward if x is not None])).float()
        next_state_batch = torch.from_numpy(
            np.vstack([x for x in batch.next_state if x is not None])).float()
        done_batch = torch.from_numpy(np.vstack(
            [x for x in batch.done if x is not None]).astype(np.uint8)).float()

        Q_argmax = self.policy_net(next_state_batch).detach()
        _, a_max = Q_argmax.max(1)

        Q_target_next = self.target_net(
            next_state_batch).detach().gather(1, a_max.unsqueeze(1))

        Q_target = reward_batch + \
                   (gamma * Q_target_next * (1 - done_batch))

        Q_expected = self.policy_net(state_batch).gather(1, action_batch)

        # loss = F.mse_loss(Q_expected, Q_target)

        # or use Huber loss
        loss = F.smooth_l1_loss(Q_expected, Q_target)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(
                parameters["tau"] * policy_param.data + (1. - parameters["tau"]) * target_param.data)

    def train(self):
        reward_history = []
        steps_per_episode = []
        epsilons = []
        fuel_consumption = []
        rolling_reward_history = deque(maxlen=100)
        epsilon = parameters["epsilon"]
        for i_episode in range(parameters["episodes"]):
            state = self.env.reset()[0]
            reward, fuel = 0, 0
            steps = parameters["max_steps"]
            for t in range(parameters["max_steps"]):
                action = self.act(state, epsilon)
                if action == 2:
                    fuel += 1
                elif action != 0:
                    fuel += 0.1
                next_state, r, done, _, _ = self.env.step(action)
                self.memory.save(state, action, next_state, r, done)

                self.steps = (self.steps + 1) % parameters["update_freq"]
                if self.steps == 0:
                    if len(self.memory) >= parameters["batch_size"]:
                        transitions = self.memory.sample(parameters["batch_size"])
                        self.learn(transitions, parameters["gamma"])
                state = next_state
                reward += r
                if done:
                    steps = t
                    break

            reward_history.append(reward)
            steps_per_episode.append(steps)
            rolling_reward_history.append(reward)
            epsilons.append(epsilon)
            fuel_consumption.append(fuel)
            epsilon = max(parameters["epsilon_decay"] * epsilon, parameters["epsilon_min"])

            print('\rEpisode {}\tAverage Reward: {:.2f}'.format(
                i_episode + 1, np.mean(rolling_reward_history)), end="")
            if (i_episode + 1) % 100 == 0:
                print('\rEpisode {}\tAverage Reward: {:.2f}'.format(
                    i_episode + 1, np.mean(rolling_reward_history)))
                torch.save(self.policy_net.state_dict(), 'model.pth')
            if np.mean(rolling_reward_history) >= parameters["reward_target_mean"]:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(
                    i_episode + 1 - 100, np.mean(rolling_reward_history)))
                torch.save(self.policy_net.state_dict(), 'model.pth')
                break

        with open("data/rewards.txt", "w") as f:
            for elem in reward_history:
                f.write(str(elem) + '\n')
        with open("data/epsilons.txt", "w") as f:
            for elem in epsilons:
                f.write(str(elem) + '\n')
        with open("data/steps_per_episode.txt", "w") as f:
            for elem in steps_per_episode:
                f.write(str(elem) + '\n')
        with open("data/fuel_consumption.txt", "w") as f:
            for elem in fuel_consumption:
                f.write(str(elem) + '\n')

    def test(self):
        self.policy_net.load_state_dict(torch.load(
            'data/model.pth', map_location=lambda storage, loc: storage))

        test_scores = []
        steps_per_episode = []
        fuel_consumption = []
        for j in range(100):
            state = self.env.reset()[0]
            reward, fuel = 0, 0
            steps = parameters["max_steps"]
            for k in range(parameters["max_steps"]):
                action = self.act(state, epsilon=0)
                if action == 2:
                    fuel += 1
                elif action != 0:
                    fuel += 0.1
                state, r, done, _, _ = self.env.step(action)
                reward += r
                if done:
                    steps = k
                    break

            print('Episode {}: {}'.format(j + 1, reward))
            test_scores.append(reward)
            steps_per_episode.append(steps)
            fuel_consumption.append(fuel)

        avg_score = sum(test_scores) / len(test_scores)

        print('\rAverage reward: {:.2f}'.format(avg_score))

        with open("eval_data/rewards.txt", "w") as f:
            for elem in test_scores:
                f.write(str(elem) + '\n')
        with open("eval_data/steps_per_episode.txt", "w") as f:
            for elem in steps_per_episode:
                f.write(str(elem) + '\n')
        with open("eval_data/fuel_consumption.txt", "w") as f:
            for elem in fuel_consumption:
                f.write(str(elem) + '\n')


if __name__ == '__main__':
    agent = Agent('RewardCustomLunarLander')
    # agent.train()
    agent.test()
