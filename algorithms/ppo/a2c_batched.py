#
# Program name: a2c_batched.py
# Description: A batched version of advantage actor critic.
#

import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np

from algorithms.ppo.a2c import Actor, Critic

class BatchedRollout:
    """
    A store of information from the current policy that is used for the update rule, as well as a set of
    methods that allow us to actually generate that information in the first place.
    This is a technique employed in policy gradient methods.
    """

    def __init__(self, envs: gym.vector.VectorEnv, gamma:float = 0.99, lam:float = 0.5):
        # environment details
        self.envs = envs
        self.num_envs = envs.num_envs

        self.gamma = gamma # the discount factor
        self.lam = lam # lambda, the GAE advantage

        self.current_obs, _ = self.envs.reset()

        # rollout buffer
        self.observations =     []
        self.new_observations = []
        self.actions =          []
        self.rewards =          []
        self.terminations =     []
        self.truncations =      []
        self.log_probs =        []
        self.values =           []
        self.next_values =      []

        # required for GAE
        self.advantages =   []
        self.returns =      []

        # used for metrics
        self.completed_returns = []
        self.completed_lengths = []

        self.episode_returns = np.zeros(self.num_envs)
        self.episode_lengths = np.zeros(self.num_envs)

    def tensors(self):
        """
        Convert all rollout information into tensors
        """
        tensors = {"observations":  torch.stack(self.observations),
                   "actions":       torch.stack(self.actions),
                   "rewards":       torch.stack(self.rewards),
                   "terminations":  torch.stack(self.terminations),
                   "truncations":   torch.stack(self.truncations),
                   "log_probs":     torch.stack(self.log_probs),
                   "values":        torch.stack(self.values),
                   "next_values":   torch.stack(self.next_values),
                   "advantages":    self.advantages,
                   "returns":       self.returns}

        return tensors

    def clear(self):
        """
        Return the rollout to a clear state.
        """
        self.observations = []
        self.actions =      []
        self.rewards =      []
        self.terminations = []
        self.truncations =  []
        self.log_probs =    []
        self.values =       []
        self.next_values =  []

        self.advantages =   []
        self.returns =      []


    def run(self, actor: Actor, critic: Critic, rollout_steps: int):
        """
        Takes a actor network and uses it to generate actions to interact with the environment.
        """
        obs = self.current_obs
        for t in range(rollout_steps):
            # convert obs into a tensor
            obs = torch.tensor(obs, dtype=torch.float32)

            with torch.no_grad():
                # sample action and value
                mean, std = actor(obs)
                value = critic(obs)

                dist = torch.distributions.Normal(mean, std)
                actions = dist.sample()
                log_prob = dist.log_prob(actions).sum(dim=-1)

                # ensure the produced action is kept within the bounds of the continuous action space
                env_actions = actions.clamp(
                    torch.as_tensor(self.envs.action_space.low, dtype=torch.float32),
                    torch.as_tensor(self.envs.action_space.high, dtype=torch.float32)
                )

            # generate new observation
            new_obs, rewards, terminated, truncated, info = self.envs.step(env_actions.numpy())

            done = np.logical_or(terminated, truncated)

            # used to calculate the next value - not used in the rollout buffer
            # this prevents the truncated transition value from being the reset state over the actual final state,
            # which is stored in info when using the SAMESTEP environment
            bootstrap_obs = np.array(new_obs, copy=True)
            if "final_obs" in info:
                for env_index in np.flatnonzero(truncated):
                    bootstrap_obs[env_index] = info["final_obs"][env_index]

            with torch.no_grad():
                next_val = critic(torch.as_tensor(bootstrap_obs, dtype=torch.float32))

            # append all info into rollout buffer
            self.observations.append(obs)
            self.actions.append(actions)
            self.rewards.append(torch.as_tensor(rewards, dtype=torch.float32))
            self.terminations.append(torch.as_tensor(terminated, dtype=torch.bool))
            self.truncations.append(torch.as_tensor(truncated, dtype=torch.bool))
            self.log_probs.append(log_prob)
            self.values.append(value)
            self.next_values.append(next_val)

            self.episode_returns += rewards
            self.episode_lengths += 1

            for env_index in np.flatnonzero(done):
                self.completed_returns.append(self.episode_returns[env_index])
                self.completed_lengths.append(self.episode_lengths[env_index])
                self.episode_returns[env_index] = 0.0
                self.episode_lengths[env_index] = 0

            obs = new_obs

        # feed obs back into class
        self.current_obs = obs

    def calculate_gae(self):
        """
        Employs the generalised advantage estimate employed in the main paper.
        GAE is merely a weighted advantage estimator that is modulated by a parameter (lambda) to determine how many
        steps to generate an advantage estimate over..
        """

        # could use the "tensors" method, but I've opted not to - maintains continuity with previous implementation
        rewards = torch.stack(self.rewards)
        values = torch.stack(self.values)
        next_values = torch.stack(self.next_values)
        terminations = torch.stack(self.terminations)
        truncations = torch.stack(self.truncations)

        advantages = torch.zeros_like(rewards)

        gae = torch.zeros(self.num_envs, dtype=torch.float32)

        for t in reversed(range(len(self.rewards))):
            # truncation and terminations have to be handled differently
            #   * if truncation occurs, the episode could have continued and the next state shouldn't be defaulted to zero
            #   * however, termination should lead to a zero-value new stage.
            # we need to distinguish on these cases.

            termination_mask = (~terminations[t]).float()
            new_episode_mask = (~(terminations[t] | truncations[t])).float()

            # accumulate exponentially weighted TD residuals
            # make sure to remove the contributions of the next state if the current state is terminal
            delta = rewards[t] + self.gamma*next_values[t]*termination_mask - values[t]
            gae = delta + self.gamma * self.lam * gae * new_episode_mask

            advantages[t] = gae

        self.advantages = advantages
        self.returns = advantages + values

class BatchedActorCritic:
    """
    A batched implementation of advantage actor-critic.
    """
    def __init__(self, actor: Actor, critic: Critic, envs: gym.vector.VectorEnv, test_env: gym.Env,
                 actor_lr:float = 3e-4, critic_lr:float=3e-4,
                 gam:float = 0.99, lam:float=0.95):

        # networks
        self.actor, self.actor_lr = actor, actor_lr
        self.critic, self.critic_lr = critic, critic_lr
        self.rollout = BatchedRollout(envs, gamma=gam, lam=lam)

        self.test_env = test_env

        self.value_loss = nn.MSELoss()
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)


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

            # advantage normalization
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

            # flatten all values into one update batch
            T, N = advantages.shape
            observations = observations.reshape(T * N, *observations.shape[2:])
            actions = actions.reshape(T * N,*actions.shape[2:])
            advantages = advantages.reshape(T * N).detach()
            returns = returns.reshape(T * N).detach()

            # recompute policy quantities for loss update
            means, stds = self.actor(observations)
            distribution = torch.distributions.Normal(means, stds)
            log_probs = distribution.log_prob(actions).sum(dim=-1)
            predicted_values = self.critic(observations)

            # calculate loss
            actor_loss = - (log_probs * advantages).mean()
            critic_loss = self.value_loss(predicted_values, returns)

            # backprop
            self.actor_optim.zero_grad()
            actor_loss.backward()
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optim.step()

            self.critic_optim.zero_grad()
            critic_loss.backward()
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic_optim.step()

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