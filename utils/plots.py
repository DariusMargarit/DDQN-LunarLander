import pandas as pd
import matplotlib.pyplot as plt
import os


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


def plot_data(path):
    """
    Give a directory path.
    """
    if os.path.isfile(path + "/rewards.txt"):
        with open(path + "/rewards.txt", "r") as f:
            lines = f.readlines()
            rewards = [float(line.strip()) for line in lines]
            plot_rewards(rewards, path + "/rewards.png")

    if os.path.isfile(path + "/epsilons.txt"):
        with open(path + "/epsilons.txt", "r") as f:
            lines = f.readlines()
            epsilons = [float(line.strip()) for line in lines]
            plot_epsilon(epsilons, path + "/epsilons.png")

    if os.path.isfile(path + "/steps_per_episode.txt"):
        with open(path + "/steps_per_episode.txt", "r") as f:
            lines = f.readlines()
            steps_per_episode = [float(line.strip()) for line in lines]
            plot_steps_per_episode(steps_per_episode, path + "/steps_per_episode.png")

    if os.path.isfile(path + "/fuel_consumption.txt"):
        with open(path + "/fuel_consumption.txt", "r") as f:
            lines = f.readlines()
            fuel_consumption = [float(line.strip()) for line in lines]
            plot_fuel_consumption(fuel_consumption, path + "/fuel_consumption.png")


def plot_accuracy(path):
    """
    Give a .txt path.
    """
    with open(path, "r") as f:
        lines = f.readlines()
        landings = 0
        crashes = 0
        for line in lines:
            reward = float(line.strip())
            if reward < -500:
                crashes += 1
            else:
                landings += 1
        labels = "Landing\n" + str(landings * 100 / (landings + crashes)) + "%", \
            "Crash\n" + str(crashes * 100 / (landings + crashes)) + "%"
        sizes = [int(landings), int(crashes)]
        plt.pie(sizes, labels=labels)
        plt.savefig(path[:-3] + "png")
