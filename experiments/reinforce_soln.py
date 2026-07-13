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

training_eps = 2000
agents = 10
reward_log = 1

# generate seeds only once
random_seeds = np.random.randint(0, 1000, size = agents)

def layer_change(size_of_layer: int, normalize=False):
    print(f"Running experiment for layer size {size_of_layer}...")
    t0 = time.time()
    rewards = {agent: [] for agent in range(agents)}
    steps = [i * reward_log for i in range(training_eps // reward_log)]

    for agent in range(agents):
        print(f"Starting run for agent {agent}...")
        cartpole = gym.make('CartPole-v1')

        if normalize:
            reinforce = REINFORCE(env=cartpole,
                                  gamma=0.999,
                                  lr=1e-2,
                                  seed=int(random_seeds[agent]),
                                  hidden_layer=size_of_layer,
                                  norm_reward=True)
        else:
            reinforce = REINFORCE(env=cartpole,
                                  seed=int(random_seeds[agent]),
                                  hidden_layer=size_of_layer)

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

    plt.plot(steps, mean, label=f'Hidden layer: {size_of_layer}, Norm={normalize}')
    plt.fill_between(steps, mean - std, mean + std, alpha=0.2)

# for layer in [256, 512, 1024]:
#    layer_change(layer)

layer_change(128, True)

plt.xlabel("Episodes")
plt.ylabel("Average episodic reward")
plt.legend()
plt.show()