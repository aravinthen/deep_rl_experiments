#
# Program name: reinforce_soln.py
# Description: Solving cartpole-v1 with reinforce.
#
from cProfile import label

from algorithms.reinforce.vanilla import REINFORCE
from algorithms.reinforce.baselines import REINFORCEWithBaseline
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time

training_eps = 500
agents = 10
reward_log = 1

# generate seeds only once
random_seeds = np.random.randint(0, 1000, size = agents)

def test_reinforce(size_of_layer: int,):
    print(f"Running experiment for layer size {size_of_layer}...")
    t0 = time.time()
    rewards = {agent: [] for agent in range(agents)}
    steps = [i * reward_log for i in range(training_eps // reward_log)]

    for agent in range(agents):
        print(f"Starting run for agent {agent}...")
        cartpole = gym.make('CartPole-v1')

        reinforce = REINFORCE(env=cartpole,
                              gamma=0.99,
                              lr=5e-3,
                              seed=int(random_seeds[agent]),
                              hidden_layer=size_of_layer,
                              norm_reward=True)

        for ep in range(training_eps):
            reward = reinforce.run_policy()
            reinforce.update()

            if ep % reward_log == 0:
                rewards[agent].append(reward)

        cartpole.close()

    t1 = time.time()
    print(f"Time taken: {t1 - t0}")

    # plot average reward during training
    all_rewards = np.array([rewards[agent] for agent in rewards])

    mean = np.mean(all_rewards, axis=0)
    std = np.std(all_rewards, axis=0)

    plt.plot(steps, mean, label=f'Hidden layer: {size_of_layer}, No Baseline')
    plt.fill_between(steps, mean - std, mean + std, alpha=0.2)

def test_baseline(size_of_layer: int,):
    print(f"Running experiment for layer size {size_of_layer}...")
    t0 = time.time()
    rewards = {agent: [] for agent in range(agents)}
    steps = [i * reward_log for i in range(training_eps // reward_log)]

    for agent in range(agents):
        print(f"Starting run for agent {agent}...")
        cartpole = gym.make('CartPole-v1')
        reinforce = REINFORCEWithBaseline(env=cartpole,
                                          gamma=0.99,
                                          lr=5e-3,
                                          v_lr=1e-3,
                                          value_scale=1.0,
                                          seed=int(random_seeds[agent]),
                                          hidden_layer=size_of_layer,
                                          value_layer=size_of_layer,
                                          norm_reward=True)

        for ep in range(training_eps):
            reward = reinforce.run_policy()
            reinforce.update()

            if ep % reward_log == 0:
                rewards[agent].append(reward)

        cartpole.close()

    t1 = time.time()
    print(f"Time taken: {t1 - t0}")

    # plot average reward during training
    all_rewards = np.array([rewards[agent] for agent in rewards])

    mean = np.mean(all_rewards, axis=0)
    std = np.std(all_rewards, axis=0)

    plt.plot(steps, mean, label=f'Hidden layer: {size_of_layer}, With Baseline')
    plt.fill_between(steps, mean - std, mean + std, alpha=0.2)


test_reinforce(64)
test_baseline(64)

plt.xlabel("Episodes")
plt.ylabel("Episodic reward")
plt.legend()
plt.show()