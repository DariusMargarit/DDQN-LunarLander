from utils.plots import *
import gymnasium as gym
from utils.DDQNAgent import DoubleQAgent


env = gym.make("LunarLander-v2")
agent = DoubleQAgent(gamma=0.99, epsilon=0, epsilon_dec=0, lr=0.001, mem_size=1, batch_size=128,
                     epsilon_end=0, seed=False)
agent.load_saved_model('data/ddqn_torch_model.h5')
print('Loaded most recent: ddqn_torch_model.h5')

rewards = []
steps_per_episode = []
fuel_consumption = []
for i in range(300):
    terminated, truncated = False, False
    score, steps, fuel = 0, 0, 0
    state = env.reset()[0]
    while not (terminated or truncated):
        action = agent.choose_action(state)
        if action == 2:
            fuel += 1
        elif action != 0:
            fuel += 0.1
        new_state, reward, terminated, truncated, info = env.step(action)
        state = new_state
        steps += 1
        score += reward

    rewards.append(score)
    steps_per_episode.append(steps)
    fuel_consumption.append(fuel)
    print(f"Episode {i}, reward: {score}, eps: {agent.epsilon}")

with open("eval_data/reward_plots/rewards.txt", "w") as f:
    for elem in rewards:
        f.write(str(elem) + '\n')

with open("eval_data/steps_per_episode.txt", "w") as f:
    for elem in steps_per_episode:
        f.write(str(elem) + '\n')

with open("eval_data/fuel_consumption.txt", "w") as f:
    for elem in fuel_consumption:
        f.write(str(elem) + '\n')



with open("eval_data/reward_plots/rewards.txt", "r") as f:
    lines = f.readlines()
    rewards = [float(line.strip()) for line in lines]
    plot_rewards(rewards, "eval_data/reward_plots/rewards.png")

with open("eval_data/steps_per_episode.txt", "r") as f:
    lines = f.readlines()
    steps_per_episode = [float(line.strip()) for line in lines]
    plot_steps_per_episode(steps_per_episode, "eval_data/steps_per_episode.png")

with open("eval_data/fuel_consumption.txt", "r") as f:
    lines = f.readlines()
    fuel_consumption = [float(line.strip()) for line in lines]
    plot_fuel_consumption(fuel_consumption, "eval_data/fuel_consumption.png")