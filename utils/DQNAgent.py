import torch
import torch.nn.functional as F
import numpy as np

from utils.QNN import QNN
from utils.ReplayMemory import MemoryBuffer


class Agent(object):
    def __init__(self, gamma=0.99, epsilon=1.0, batch_size=128,
                 epsilon_dec=0.996, epsilon_end=0.01,
                 mem_size=1000000, seed=True):
        self.gamma = gamma  # alpha = learn rate, gamma = discount
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_end
        self.batch_size = batch_size
        self.memory = MemoryBuffer(mem_size)
        if seed:
            np.random.seed(20)

    def save(self, state, action, reward, new_state, done):
        self.memory.save(state, action, reward, new_state, done)

    def choose_action(self, state):
        rand = np.random.random()
        state = torch.from_numpy(state).float().unsqueeze(0)
        self.q_func.eval()
        with torch.no_grad():
            action_values = self.q_func(state)
        self.q_func.train()
        if rand > self.epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return np.random.choice([i for i in range(4)])

    def reduce_epsilon(self):
        self.epsilon = self.epsilon * self.epsilon_dec if self.epsilon > \
                                                          self.epsilon_min else self.epsilon_min

    def learn(self):
        raise Exception("Not implemented")

    def save_model(self, path):
        torch.save(self.q_func.state_dict(), path)

    def load_saved_model(self, path):
        self.q_func = QNN(8, 4, 42)
        self.q_func.load_state_dict(torch.load(path))
        self.q_func.eval()


class DoubleQAgent(Agent):
    def __init__(self, gamma=0.99, epsilon=1.0, batch_size=128, lr=0.001,
                 epsilon_dec=0.996, epsilon_end=0.01,
                 mem_size=1000000, replace_q_target=100, seed=True):

        super().__init__(gamma=gamma, epsilon=epsilon, batch_size=batch_size,
                         epsilon_dec=epsilon_dec, epsilon_end=epsilon_end,
                         mem_size=mem_size, seed=seed)

        self.replace_q_target = replace_q_target
        self.q_func = QNN(8, 4, 42)
        self.q_func_target = QNN(8, 4, 42)
        self.optimizer = torch.optim.Adam(self.q_func.parameters(), lr=lr)

    def learn(self):
        if self.memory.trans_counter < self.batch_size:
            return

        # 1. Choose a sample from past transitions:
        states, actions, rewards, new_states, terminals = self.memory.random_sample(self.batch_size)

        # 2. Update the target values
        q_next = self.q_func_target(new_states).detach().max(1)[0].unsqueeze(1)
        q_updated = rewards + self.gamma * q_next * (1 - terminals)
        q = self.q_func(states).gather(1, actions)

        # 3. Update the main NN
        loss = F.mse_loss(q, q_updated)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 4. Update the target NN (every N-th step)
        if self.memory.trans_counter % self.replace_q_target == 0:  # wait before you start learning
            for target_param, local_param in zip(self.q_func_target.parameters(), self.q_func.parameters()):
                target_param.data.copy_(local_param.data)

        # 5. Reduce the exploration rate
        self.reduce_epsilon()

    def save_model(self, path):
        super().save_model(path)
        torch.save(self.q_func.state_dict(), path + '.target')

    def load_saved_model(self, path):
        super().load_saved_model(path)
        self.q_func_target = QNN(8, 4, 42)
        self.q_func_target.load_state_dict(torch.load(path + '.target'))
        self.q_func_target.eval()
