#
# Program name: ppo.py
# Description: A full PPO implementation.
#

import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np

from algorithms.ppo.a2c import Actor, Critic
from algorithms.ppo.a2c_batched import BatchedRollout

class PPO:
    """
    A batched implementation of Proximal Policy Optimization.
    """
    def __init__(self, actor: Actor, critic: Critic, envs: gym.vector.VectorEnv, test_env: gym.Env,
                 lr:float = 3e-4,
                 gam:float = 0.99,
                 lam:float=0.95,
                 clip_eps: float = 0.2,
                 update_epochs: int = 10,
                 minibatch_size: int = 64,
                 vf_coef: float = 0.5,
                 ent_coef: float = 0.0,
                 max_grad_norm: float = 0.5):

        # networks
        self.actor, self.actor_lr = actor, lr
        self.critic, self.critic_lr = critic, lr
        self.rollout = BatchedRollout(envs, gamma=gam, lam=lam)

        # params
        self.clip_eps = clip_eps
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm

        self.test_env = test_env

        # PPO uses a shared optimizer for both the actor and the critic
        self.optimizer = torch.optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()),
                                          lr=self.actor_lr)
        self.value_loss = nn.MSELoss()

    def train(self, n_updates, rollout_steps=256):

        for update in range(n_updates):
            # reset rollout
            self.rollout.clear()

            # generate rollout
            self.rollout.run(self.actor, self.critic, rollout_steps)
            self.rollout.calculate_gae()

            # obtain tensorized rollout values
            batch = self.rollout.tensors()

            observations = batch["observations"]
            actions =  batch["actions"]
            advantages = batch["advantages"]
            returns = batch["returns"]

            # used in PPO update rule to compute the ratio
            old_log_probs = batch["log_probs"]
            old_values = batch["values"]

            T, N = advantages.shape
            batch_size = T * N

            # flatten all values into one update batch
            T, N = advantages.shape
            observations = observations.reshape(T * N, *observations.shape[2:])
            actions = actions.reshape(T * N,*actions.shape[2:])
            advantages = advantages.reshape(T * N).detach()
            returns = returns.reshape(T * N).detach()
            old_log_probs = old_log_probs.reshape(batch_size).detach()

            # advantage normalization
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

            # PPO epochs and minibatching
            for epoc in range(self.update_epochs):
                # generate a random selection of indices to control the minibatch.
                indices = torch.randperm(batch_size)

                for start in range(0, batch_size, self.minibatch_size):
                    end = start + self.minibatch_size
                    minibatch_ids = indices[start:end]

                    # obtain components of the batch
                    minibatch_obs = observations[minibatch_ids]
                    minibatch_actions = actions[minibatch_ids]
                    minibatch_old_log_probs = old_log_probs[minibatch_ids]
                    minibatch_advantages = advantages[minibatch_ids]
                    minibatch_returns = returns[minibatch_ids]

                    # obtain the new log probabilities
                    means, stds = self.actor(minibatch_obs)
                    dist = torch.distributions.Normal(means, stds)
                    new_log_probs = dist.log_prob(minibatch_actions).sum(dim=-1)
                    entropy = dist.entropy().sum(dim=-1).mean()
                    values = self.critic(minibatch_obs)

                    # ppo ratio
                    log_ratio = new_log_probs - minibatch_old_log_probs
                    ratio = torch.exp(log_ratio)

                    # surrogate objective
                    unclipped_loss = ratio * minibatch_advantages

                    # TODO add a clippng metric!
                    clipped_loss = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * minibatch_advantages

                    # actor loss
                    actor_loss = -torch.min(unclipped_loss, clipped_loss).mean()

                    # value loss
                    critic_loss = self.value_loss(values, minibatch_returns)

                    # full loss with entropy term
                    loss = actor_loss + self.vf_coef*critic_loss - self.ent_coef * entropy

                    self.optimizer.zero_grad()
                    loss.backward()

                    # gradient clipping following actor-critic
                    grad_norm = torch.nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()),
                                                               self.max_grad_norm)
                    self.optimizer.step()

                # TODO - add metrics


    def test(self, num_eps, seeds, stochastic=False):
        """
        Test the current actor and return the average reward it obtains.
        """
        rewards = []
        for s in range(num_eps):
            obs, _ = self.test_env.reset(seed=seeds[s])
            terminated = False
            truncated = False

            total_reward = 0

            # run until episide is not truncated or terminated
            while not terminated and not truncated:
                obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

                with torch.no_grad():
                    mean, std = self.actor(obs)
                    if not stochastic:
                        # test with just the mean action
                        action = mean
                    else:
                        dist = torch.distributions.Normal(mean, std)
                        action = dist.sample()


                env_action = action.squeeze(0).clamp(
                    torch.as_tensor(self.test_env.action_space.low, dtype=torch.float32),
                    torch.as_tensor(self.test_env.action_space.high, dtype=torch.float32)
                )

                new_obs, reward, terminated, truncated, _ = self.test_env.step(env_action.numpy())

                total_reward += reward
                obs = new_obs

            rewards.append(total_reward)

        return np.mean(rewards), np.std(rewards)
