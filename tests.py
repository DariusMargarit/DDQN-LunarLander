import matplotlib.pyplot as plt
def plot_steps_per_episode(values, path):
    plt.figure(2)
    plt.clf()
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.plot(values)
    plt.savefig(path)


with open("eval_data" + "/steps_per_episode.txt", "r") as f:
    lines = f.readlines()
    steps_per_episode = [float(line.strip()) for line in lines]
    plot_steps_per_episode(steps_per_episode, "eval_data" + "/steps_per_episode.png")