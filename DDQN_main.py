from utils.plots import *
from utils.DDQNAgent import Agent

agent = Agent('RewardCustomLunarLander')
# agent.train()
# plot_data()
agent.test()
# plot_eval_data()
