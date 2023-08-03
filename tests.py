# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import numpy as np
# import collections
# import random
# import gymnasium as gym
# from utils.plots import *
#
#
# class MemoryBuffer(object):
#     def __init__(self, max_size):
#         self.memory_size = max_size
#         self.trans_counter = 0
#         self.index = 0
#         self.buffer = collections.deque(maxlen=self.memory_size)
#         self.transition = collections.namedtuple("Transition",
#                                                  field_names=["state", "action", "reward", "new_state", "terminal"])
#         random.seed(12)
#
#     def save(self, state, action, reward, new_state, terminal):
#         t = self.transition(state, action, reward, new_state, terminal)
#         self.buffer.append(t)
#         self.trans_counter = (self.trans_counter + 1) % self.memory_size
#
#     def random_sample(self, batch_size):
#         assert len(self.buffer) >= batch_size
#         transitions = random.sample(self.buffer, k=batch_size)
#         states = torch.from_numpy(np.vstack([e.state for e in transitions if e is not None])).float()
#         actions = torch.from_numpy(np.vstack([e.action for e in transitions if e is not None])).long()
#         rewards = torch.from_numpy(np.vstack([e.reward for e in transitions if e is not None])).float()
#         new_states = torch.from_numpy(np.vstack([e.new_state for e in transitions if e is not None])).float()
#         terminals = torch.from_numpy(
#             np.vstack([e.terminal for e in transitions if e is not None]).astype(np.uint8)).float()
#
#         return states, actions, rewards, new_states, terminals
#
#
# class QNN(nn.Module):
#     def __init__(self, state_size, action_size, seed):
#         super(QNN, self).__init__()
#         self.seed = torch.manual_seed(seed)
#         self.fc1 = nn.Linear(state_size, 256)
#         self.fc2 = nn.Linear(256, 256)
#         self.fc3 = nn.Linear(256, action_size)
#
#     def forward(self, state):
#         x = self.fc1(state)
#         x = F.relu(x)
#         x = self.fc2(x)
#         x = F.relu(x)
#         return self.fc3(x)
#
#
# class Agent(object):
#     def __init__(self, gamma=0.99, epsilon=1.0, batch_size=128, lr=0.001,
#                  epsilon_dec=0.996, epsilon_end=0.01,
#                  mem_size=1000000):
#         self.gamma = gamma  # alpha = learn rate, gamma = discount
#         self.epsilon = epsilon
#         self.epsilon_dec = epsilon_dec
#         self.epsilon_min = epsilon_end
#         self.batch_size = batch_size
#         self.memory = MemoryBuffer(mem_size)
#         np.random.seed(20)
#
#     def save(self, state, action, reward, new_state, done):
#         self.memory.save(state, action, reward, new_state, done)
#
#     def choose_action(self, state):
#         rand = np.random.random()
#         state = torch.from_numpy(state).float().unsqueeze(0)
#         self.q_func.eval()
#         with torch.no_grad():
#             action_values = self.q_func(state)
#         self.q_func.train()
#         if rand > self.epsilon:
#             return np.argmax(action_values.cpu().data.numpy())
#         else:
#             return np.random.choice([i for i in range(4)])
#
#     def reduce_epsilon(self):
#         self.epsilon = self.epsilon * self.epsilon_dec if self.epsilon > \
#                                                           self.epsilon_min else self.epsilon_min
#
#     def learn(self):
#         raise Exception("Not implemented")
#
#     def save_model(self, path):
#         torch.save(self.q_func.state_dict(), path)
#
#     def load_saved_model(self, path):
#         self.q_func = QNN(8, 4, 42)
#         self.q_func.load_state_dict(torch.load(path))
#         self.q_func.eval()
#
#
# class DoubleQAgent(Agent):
#     def __init__(self, gamma=0.99, epsilon=1.0, batch_size=128, lr=0.001,
#                  epsilon_dec=0.996, epsilon_end=0.01,
#                  mem_size=1000000, replace_q_target=100):
#
#         super().__init__(lr=lr, gamma=gamma, epsilon=epsilon, batch_size=batch_size,
#                          epsilon_dec=epsilon_dec, epsilon_end=epsilon_end,
#                          mem_size=mem_size)
#
#         self.replace_q_target = replace_q_target
#         self.q_func = QNN(8, 4, 42)
#         self.q_func_target = QNN(8, 4, 42)
#         self.optimizer = optim.Adam(self.q_func.parameters(), lr=lr)
#
#     def learn(self):
#         if self.memory.trans_counter < self.batch_size:
#             return
#
#         # 1. Choose a sample from past transitions:
#         states, actions, rewards, new_states, terminals = self.memory.random_sample(self.batch_size)
#
#         # 2. Update the target values
#         q_next = self.q_func_target(new_states).detach().max(1)[0].unsqueeze(1)
#         q_updated = rewards + self.gamma * q_next * (1 - terminals)
#         q = self.q_func(states).gather(1, actions)
#
#         # 3. Update the main NN
#         loss = F.mse_loss(q, q_updated)
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()
#
#         # 4. Update the target NN (every N-th step)
#         if self.memory.trans_counter % self.replace_q_target == 0:  # wait before you start learning
#             for target_param, local_param in zip(self.q_func_target.parameters(), self.q_func.parameters()):
#                 target_param.data.copy_(local_param.data)
#
#         # 5. Reduce the exploration rate
#         self.reduce_epsilon()
#
#     def save_model(self, path):
#         super().save_model(path)
#         torch.save(self.q_func.state_dict(), path + '.target')
#
#     def load_saved_model(self, path):
#         super().load_saved_model(path)
#         self.q_func_target = QNN(8, 4, 42)
#         self.q_func_target.load_state_dict(torch.load(path + '.target'))
#         self.q_func_target.eval()
#
#
# LEARN_EVERY = 4
#
#
# def train_agent(n_episodes=2000, load_latest_model=False):
#     print("Training a DDQN agent on {} episodes. Pretrained model = {}".format(n_episodes, load_latest_model))
#     env = gym.make("LunarLander-v2")
#     env.reset(seed=10)
#     agent = DoubleQAgent(gamma=0.99, epsilon=1.0, epsilon_dec=0.995, lr=0.001, mem_size=200000, batch_size=128,
#                          epsilon_end=0.01)
#     if load_latest_model:
#         agent.load_saved_model('ddqn_torch_model.h5')
#         print('Loaded most recent: ddqn_torch_model.h5')
#
#     rewards = []
#     epsilons = []
#     steps_per_episode = []
#     fuel_consumption = []
#     for i in range(n_episodes):
#         terminated, truncated = False, False
#         score, steps, fuel = 0, 0, 0
#         state = env.reset()[0]
#         while not (terminated or truncated):
#             action = agent.choose_action(state)
#             if action == 2:
#                 fuel += 1
#             elif action != 0:
#                 fuel += 0.1
#             new_state, reward, terminated, truncated, info = env.step(action)
#             agent.save(state, action, reward, new_state, terminated)
#             state = new_state
#             if steps > 0 and steps % LEARN_EVERY == 0:
#                 agent.learn()
#             steps += 1
#             score += reward
#
#         epsilons.append(agent.epsilon)
#         rewards.append(score)
#         steps_per_episode.append(steps)
#         fuel_consumption.append(fuel)
#
#         print(f"Episode {i}, reward: {score}, eps: {agent.epsilon}")
#
#         if (i + 1) % 100 == 0 and i > 0:
#             agent.save_model('ddqn_torch_model.h5')
#
#
#     with open("rewards.txt", "w") as f:
#         for elem in rewards:
#             f.write(str(elem) + '\n')
#     with open("epsilons.txt", "w") as f:
#         for elem in epsilons:
#             f.write(str(elem) + '\n')
#     with open("steps_per_episode.txt", "w") as f:
#         for elem in steps_per_episode:
#             f.write(str(elem) + '\n')
#     with open("fuel_consumption.txt", "w") as f:
#         for elem in fuel_consumption:
#             f.write(str(elem) + '\n')
#
#     plot_rewards(rewards, "rewards.png")
#     return agent
#
#
# agent = train_agent(n_episodes=1500, load_latest_model=False)
