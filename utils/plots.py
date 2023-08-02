import pandas as pd
import matplotlib.pyplot as plt

def moving_average(data, window):
    series = pd.Series(data)
    return series.rolling(window).mean()


def plot_rewards(values, path):
    plt.figure(2)
    plt.clf()
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(values)
    plt.plot(moving_average(values, 100))
    plt.savefig(path)

def plot_epsilon(values, path):
    plt.figure(2)
    plt.clf()
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.plot(values)
    plt.savefig(path)

def plot_steps_per_episode(values, path):
    plt.figure(2)
    plt.clf()
    plt.xlabel('Episode')
    plt.ylabel('Steps per episode')
    plt.plot(values)
    plt.savefig(path)

def plot_fuel_consumption(values, path):
    plt.figure(2)
    plt.clf()
    plt.xlabel('Episode')
    plt.ylabel('Fuel consumption')
    plt.plot(values)
    plt.savefig(path)