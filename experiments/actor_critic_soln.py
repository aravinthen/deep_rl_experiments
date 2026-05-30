#
# Program name: reinforce_soln.py
# Description: Solving cartpole-v1 with reinforce.
#
from cProfile import label

from algorithms.reinforce.vanilla import REINFORCE
from algorithms.reinforce.baselines import REINFORCEWithBaseline
from algorithms.reinforce.actor_critic import ActorCritic
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time

training_steps = 5000
agents = 10
reward_log = 1

# generate seeds only once
random_seeds = np.random.randint(0, 1000, size = agents)

def test_actor_critic(actor: int, critic: int):
    t0 = time.time()
    rewards = {agent: [] for agent in range(agents)}
    steps = [i * reward_log for i in range(training_steps // reward_log)]

    for agent in range(agents):
        print(f"Starting run for agent {agent}...")
        cartpole = gym.make('CartPole-v1')
        actor_critic = ActorCritic(env=cartpole,
                                   gamma=0.99,
                                   lr=1e-4,
                                   v_lr=1e-3,
                                   value_scale=0.5,
                                   seed=int(random_seeds[agent]),
                                   hidden_layer=actor,
                                   value_layer=critic,
                                   norm_reward=True)

        for ep in range(training_steps):
            reward = actor_critic.train_episode()

            if ep % reward_log == 0:
                rewards[agent].append(reward)

        cartpole.close()

    t1 = time.time()
    print(f"Time taken: {t1 - t0}")

    # plot average reward during training
    all_rewards = np.array([rewards[agent] for agent in rewards])

    mean = np.mean(all_rewards, axis=0)
    std = np.std(all_rewards, axis=0)

    plt.plot(steps, mean, label=f'Hidden layer: {actor}, {critic}, With Actor Critic')
    plt.fill_between(steps, mean - std, mean + std, alpha=0.2)

test_actor_critic(64, 64)

plt.xlabel("Training steps")
plt.ylabel("Average episodic reward")
plt.legend()
plt.show()