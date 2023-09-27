from utils.plots import *
from utils.DDQNAgent import Agent

episodes = 10

agent = Agent('RewardCustomLunarLander')
agent.test(episodes)