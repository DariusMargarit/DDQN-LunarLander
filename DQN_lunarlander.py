import gymnasium as gym

from utils.plots import *
from utils.DQNAgent import DoubleQAgent

LEARN_EVERY = 4


def train_agent(n_episodes=2000, load_latest_model=False):
    print("Training a DDQN agent on {} episodes. Pretrained model = {}".format(n_episodes, load_latest_model))
    env = gym.make("LunarLander-v2")
    env.reset(seed=10)
    agent = DoubleQAgent(gamma=0.99, epsilon=1.0, epsilon_dec=0.995, lr=0.001, mem_size=200000, batch_size=128,
                         epsilon_end=0.01)
    if load_latest_model:
        agent.load_saved_model('ddqn_torch_model.h5')
        print('Loaded most recent: ddqn_torch_model.h5')

    rewards = []
    epsilons = []
    steps_per_episode = []
    fuel_consumption = []
    for i in range(n_episodes):
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
            agent.save(state, action, reward, new_state, terminated)
            state = new_state
            if steps > 0 and steps % LEARN_EVERY == 0:
                agent.learn()
            steps += 1
            score += reward

        epsilons.append(agent.epsilon)
        rewards.append(score)
        steps_per_episode.append(steps)
        fuel_consumption.append(fuel)

        print(f"Episode {i}, reward: {score}, eps: {agent.epsilon}")

        if (i + 1) % 100 == 0 and i > 0:
            agent.save_model('ddqn_torch_model.h5')


    with open("rewards.txt", "w") as f:
        for elem in rewards:
            f.write(str(elem) + '\n')
    with open("epsilons.txt", "w") as f:
        for elem in epsilons:
            f.write(str(elem) + '\n')
    with open("steps_per_episode.txt", "w") as f:
        for elem in steps_per_episode:
            f.write(str(elem) + '\n')
    with open("fuel_consumption.txt", "w") as f:
        for elem in fuel_consumption:
            f.write(str(elem) + '\n')

    plot_rewards(rewards, "rewards.png")
    return agent


agent = train_agent(n_episodes=1500, load_latest_model=False)
