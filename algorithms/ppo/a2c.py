#
# Program name: a2c.py
# Description: The components and training loop necessary to carry out PPO.
#

import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np

class Actor(nn.Module):
    """
    Module for the Actor where weights are not shared with the critic.
    As specified by original paper, the actor is designed for continuous control. This means that there must also be
    an additional trainable parameter for the standard deviation.
    Hidden defaulted to 64 as per the paper.
    """
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, act_dim),
        )

        # add learnable noise to the actor output
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs: torch.Tensor):
        """
        The output of the model is considered the 'mean action', whilst the standard deviation is trained.
        """
        mean = self.model(obs)
        std = torch.exp(self.log_std)
        return mean, std


class Critic(nn.Module):
    """
    Module for the Critic. Not too different from the basic structure introduced in the REINFORCE code, although using
    tanh activation layers as discussed in the original paper
    """
    def __init__(self, obs_dim: int, hidden: int = 64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor):
        return self.model(obs).squeeze(-1)


class Rollout:
    """
    A store of information from the current policy that is used for the update rule, as well as a set of
    methods that allow us to actually generate that information in the first place.
    This is a technique employed in policy gradient methods.
    """

    def __init__(self, env: gym.Env, gamma:float = 0.99, lam:float = 0.5):
        # environment details
        self.env = env
        self.gamma = gamma # the discount factor
        self.lam = lam # lambda, the GAE advantage

        self.current_obs, _ = self.env.reset()

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
        self.final_value = None
        self.advantages =   []
        self.returns =      []

        # used for metrics
        self.completed_returns = []
        self.completed_lengths = []
        self.episode_return = 0
        self.episode_length = 0

    def tensors(self):
        """
        Convert all rollout information into tensors
        """
        tensors = {"observations":  torch.stack(self.observations),
                   "actions":       torch.stack(self.actions),
                   "rewards":       torch.as_tensor(self.rewards, dtype=torch.float32),
                   "terminations":  torch.as_tensor(self.terminations, dtype=torch.bool),
                   "truncations":   torch.as_tensor(self.truncations, dtype=torch.bool),
                   "log_probs":     torch.stack(self.log_probs),
                   "values":        torch.stack(self.values),
                   "next_values":   torch.stack(self.next_values),
                   "final_value":   torch.as_tensor(self.final_value, dtype=torch.float32),
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

        self.final_value =  None
        self.advantages =   []
        self.returns =      []


    def run(self, actor: Actor, critic: Critic, rollout_steps: int):
        """
        Takes a actor network and uses it to generate actions to interact with the environment.
        """
        obs = self.current_obs
        for t in range(rollout_steps):
            # convert obs into a tensor
            obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                # sample action and value
                mean, std = actor(obs)
                value = critic(obs)

                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)

            # ensure the produced action is kept within the bounds of the continuous action space
            env_action = action.squeeze(0).clamp(
                torch.as_tensor(self.env.action_space.low, dtype=torch.float32),
                torch.as_tensor(self.env.action_space.high, dtype=torch.float32)
            )

            # generate new observation
            new_obs, reward, terminated, truncated, info = self.env.step(env_action.numpy())

            # handling truncation for propagation
            new_obs_tensor = torch.as_tensor(new_obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                next_val = critic(new_obs_tensor).squeeze(0)

            done = terminated or truncated

            # append all info into rollout buffer
            self.observations.append(obs.squeeze(0))
            self.actions.append(action.squeeze(0))
            self.rewards.append(reward)
            self.terminations.append(terminated)
            self.truncations.append(truncated)
            self.log_probs.append(log_prob.squeeze(0))
            self.values.append(value.squeeze(0))
            self.next_values.append(next_val)

            self.episode_return += reward
            self.episode_length += 1

            # termination and update
            if done:
                # logging metrics
                self.completed_returns.append(self.episode_return)
                self.completed_lengths.append(self.episode_length)

                self.episode_return = 0.0
                self.episode_length = 0

                obs, info = self.env.reset()
            else:
                obs = new_obs

        # feed obs back into class
        self.current_obs = obs

        # final value estimate handling (for GAE)
        final_obs = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            final_value = critic(final_obs).squeeze(0)

        self.final_value = final_value.item()

    def calculate_gae(self):
        """
        Employs the generalised advantage estimate employed in the main paper.
        GAE is merely a weighted advantage estimator that is modulated by a parameter (lambda) to determine how many
        steps to generate an advantage estimate over..
        """

        advantages = torch.zeros(len(self.rewards), dtype=torch.float32)
        gae = 0.0

        for t in reversed(range(len(self.rewards))):
            # truncation and terminations have to be handled differently
            #   * if truncation occurs, the episode could have continued and the next state shouldn't be defaulted to zero
            #   * however, termination should lead to a zero-value new stage.
            # we need to distinguish on these cases.

            terminated = float(self.terminations[t])
            new_episode_event = float(self.truncations[t] or self.terminations[t])
            termination_mask = 1.0 - terminated
            new_episode_mask = 1.0 - new_episode_event

            # values
            value = self.values[t].item()
            next_value = self.next_values[t].item()

            # accumulate exponentially weighted TD residuals
            # make sure to remove the contributions of the next state if the current state is terminal
            delta = self.rewards[t] + self.gamma*next_value*termination_mask - self.values[t].item()
            gae = delta + self.gamma * self.lam * gae * new_episode_mask

            advantages[t] = gae

        values = torch.stack(self.values)
        self.advantages = advantages
        self.returns = advantages + values

class AdvantageActorCritic:
    """
    A batched implementation of advantage actor-critic.
    """
    def __init__(self, actor: Actor, critic: Critic, env: gym.Env, test_env: gym.Env,
                 actor_lr:float = 3e-4, critic_lr:float=3e-4,
                 gam:float = 0.99, lam:float=0.95):

        # networks
        self.actor, self.actor_lr = actor, actor_lr
        self.critic, self.critic_lr = critic, critic_lr
        self.rollout = Rollout(env, gamma=gam, lam=lam)

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

            # assemble rollout into tensors
            observations = batch["observations"]
            actions =  batch["actions"]

            # detach advantages and returns for loss calculation
            advantages = batch["advantages"].detach()

            # advantage normalization
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

            returns = batch["returns"].detach()

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

        # check if arguments are sustainable
        assert len(seeds) == num_eps

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