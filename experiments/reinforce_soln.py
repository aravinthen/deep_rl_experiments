#
# Program name: reinforce_soln.py
# Description: Solving cartpole-v1 with reinforce.
#

from algorithms.reinforce.vanilla import REINFORCE
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

training_eps = 100
agents = 10
reward_log = 1

random_seeds = np.random.randint(0, 1000, size = agents)
rewards = {agent: [] for agent in range(agents)}
steps = [i*reward_log for i in range(training_eps//reward_log)]

for agent in range(agents):
    print(f"Starting run for agent {agent}...")
    cartpole = gym.make('CartPole-v1')
    reinforce = REINFORCE(env=cartpole, seed=int(random_seeds[agent]), hidden_layer=0)

    for ep in range(training_eps):
        reward = reinforce.run_policy()
        reinforce.update()

        if ep % reward_log == 0:
            rewards[agent].append(reward)

    cartpole.close()

# plot average reward during training
all_rewards = np.array([rewards[agent] for agent in rewards])

mean = np.mean(all_rewards, axis=0)
std = np.std(all_rewards, axis=0)

plt.plot(steps, mean)
plt.fill_between(steps, mean - std, mean + std, alpha=0.2)
plt.xlabel("Episodes")
plt.ylabel("Average episodic reward")

plt.show()