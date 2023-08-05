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


def plot_data():
    with open("data/rewards.txt", "r") as f:
        lines = f.readlines()
        rewards = [float(line.strip()) for line in lines]
        plot_rewards(rewards, "data/rewards.png")

    with open("data/epsilons.txt", "r") as f:
        lines = f.readlines()
        epsilons = [float(line.strip()) for line in lines]
        plot_epsilon(epsilons, "data/epsilons.png")

    with open("data/steps_per_episode.txt", "r") as f:
        lines = f.readlines()
        steps_per_episode = [float(line.strip()) for line in lines]
        plot_steps_per_episode(steps_per_episode, "data/steps_per_episode.png")

    with open("data/fuel_consumption.txt", "r") as f:
        lines = f.readlines()
        fuel_consumption = [float(line.strip()) for line in lines]
        plot_fuel_consumption(fuel_consumption, "data/fuel_consumption.png")


def plot_eval_data():
    with open("eval_data/rewards.txt", "r") as f:
        lines = f.readlines()
        rewards = [float(line.strip()) for line in lines]
        plot_rewards(rewards, "eval_data/rewards.png")

    with open("eval_data/steps_per_episode.txt", "r") as f:
        lines = f.readlines()
        steps_per_episode = [float(line.strip()) for line in lines]
        plot_steps_per_episode(steps_per_episode, "eval_data/steps_per_episode.png")

    with open("eval_data/fuel_consumption.txt", "r") as f:
        lines = f.readlines()
        fuel_consumption = [float(line.strip()) for line in lines]
        plot_fuel_consumption(fuel_consumption, "eval_data/fuel_consumption.png")
