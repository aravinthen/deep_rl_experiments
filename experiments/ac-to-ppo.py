#
# Program name: ac-to-ppo.py
# Description:  Actor critic to PPO comparison experiments
#

import gymnasium as gym
import copy

import torch
import numpy as np

import matplotlib.pyplot as plt

from algorithms.ppo.a2c import AdvantageActorCritic, Actor, Critic
from algorithms.ppo.a2c_batched import BatchedActorCritic
from algorithms.ppo.ppo import PPO

def test_algo(algorithm: AdvantageActorCritic | BatchedActorCritic | PPO,
              **env_kwargs
              ):
    """
    Carry out a test using a generic environment.
    """

    num_ep = env_kwargs["num_eps"]
    n_envs = env_kwargs["num_envs"]
    global_s = env_kwargs["global_step"]
    total_t = env_kwargs["total_timesteps"]
    next_ev = env_kwargs["next_eval"]
    rollout_s = env_kwargs["rollout_steps"]
    eval_ever = env_kwargs["eval_every"]
    stoch = env_kwargs["stochastic"]
    seeds = env_kwargs["seeds_list"]
    label = env_kwargs["label"]

    steps = []
    rewards = []
    rewards_std = []

    while global_s < total_t:
        algorithm.train(1, rollout_steps=rollout_s)
        global_s += n_envs * rollout_s

        if global_s >= next_ev:
            reward, reward_std = algorithm.test(num_ep, seeds, stochastic=stoch)
            steps.append(global_s)
            rewards.append(reward)
            rewards_std.append(reward_std)
            next_ev += eval_ever

    steps = np.array(steps)
    rewards = np.array(rewards)
    rewards_std = np.array(rewards_std)

    plt.plot(steps, rewards, label=label)
    plt.fill_between(steps, rewards - rewards_std, rewards + rewards_std, alpha=0.2)


if __name__ == '__main__':
    # set seeds
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)

    # test parameters
    total_timesteps = 1000000
    global_step = 0
    eval_every = 10000
    next_eval = 0
    num_eps = 20
    rollout_steps = 4
    stochastic = False

    # batched parameters
    num_envs = 8

    # algorithm params
    actor_lr = 3e-4
    critic_lr = 5e-3
    gam = 0.99
    lam = 0.95

    env = gym.make("Pendulum-v1")
    envs = gym.vector.SyncVectorEnv(
        [lambda: gym.make("Pendulum-v1") for _ in range(num_envs)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)

    ppo_envs = gym.vector.SyncVectorEnv(
        [lambda: gym.make("Pendulum-v1") for _ in range(num_envs)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)

    test_env_b = gym.make("Pendulum-v1")
    test_env_s = gym.make("Pendulum-v1")
    test_env_p = gym.make("Pendulum-v1")

    obs_dim = envs.single_observation_space.shape[0]
    act_dim = envs.single_action_space.shape[0]

    # initialise agents with the same weights
    base_actor, base_critic = Actor(obs_dim, act_dim, hidden=64), Critic(obs_dim, hidden=64)
    s_a, s_c = Actor(obs_dim, act_dim, hidden=64), Critic(obs_dim, hidden=64)
    b_a, b_c = Actor(obs_dim, act_dim, hidden=64), Critic(obs_dim, hidden=64)
    p_a, p_c = Actor(obs_dim, act_dim, hidden=64), Critic(obs_dim, hidden=64)

    s_a.load_state_dict(copy.deepcopy(base_actor.state_dict()))
    b_a.load_state_dict(copy.deepcopy(base_actor.state_dict()))
    p_a.load_state_dict(copy.deepcopy(base_actor.state_dict()))

    s_c.load_state_dict(copy.deepcopy(base_critic.state_dict()))
    b_c.load_state_dict(copy.deepcopy(base_critic.state_dict()))
    p_c.load_state_dict(copy.deepcopy(base_critic.state_dict()))

    sequential = AdvantageActorCritic(s_a, s_c, env, test_env_s, actor_lr=3e-4, critic_lr=5e-3, gam=0.99,  lam=0.95)
    batched = BatchedActorCritic(b_a, b_c, envs, test_env_b, actor_lr=3e-4, critic_lr=5e-3, gam=0.99,  lam=0.95)
    ppo = PPO(p_a, p_c, ppo_envs, test_env_p, lr=3e-4, gam=0.99,  lam=0.95)

    seeds_list = eval_seeds = list(range(10000, 10000+num_eps))

    print("Starting batched experiment...")
    test_algo(batched,
              num_eps=num_eps,
              num_envs=num_envs,
              rollout_steps=rollout_steps,
              total_timesteps=total_timesteps,
              global_step=global_step,
              eval_every=eval_every,
              next_eval=next_eval,
              stochastic=stochastic,
              seeds_list=seeds_list,
              label="Batched run")

    print("Starting sequential experiment...")
    test_algo(sequential,
              num_eps=num_eps,
              num_envs=1,
              rollout_steps=rollout_steps*num_envs,
              total_timesteps = total_timesteps,
              global_step = global_step,
              eval_every = eval_every,
              next_eval = next_eval,
              stochastic=stochastic,
              seeds_list=seeds_list,
              label="Sequential run")

    print("Starting PPO experiment...")
    test_algo(ppo,
              num_eps=num_eps,
              num_envs=num_envs,
              rollout_steps=128,
              total_timesteps = total_timesteps,
              global_step = global_step,
              eval_every = eval_every,
              next_eval = next_eval,
              stochastic=stochastic,
              seeds_list=seeds_list,
              label="PPO run")

    plt.legend()
    plt.xlabel("Environment steps")
    plt.ylabel("Evaluation return")
    plt.title("Sequential A2C vs Batched A2C vs PPO")
    plt.show()

    envs.close()
    env.close()
    test_env_s.close()
    test_env_b.close()
    test_env_p.close()