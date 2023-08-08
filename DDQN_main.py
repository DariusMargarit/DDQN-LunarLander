from utils.plots import *
from utils.DDQNAgent import Agent

agent = Agent('RewardCustomLunarLander')
# agent.train()
# plot_data("data")
agent.test()
# plot_data("eval_data")
