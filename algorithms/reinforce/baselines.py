#
# Program name: baselines.py
# Description: Enhances REINFORCE with a baseline calculation
#

import torch
import torch.nn as nn
import torch.optim as optim
import time

from algorithms.reinforce.vanilla import REINFORCE
import gymnasium as gym

class ValueNetwork(nn.Module):
    """
    A simple neural network for value prediction.
    """
    def __init__(self, obs: int, hidden: int):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(obs, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs):
        return self.model(obs).squeeze(-1)

class REINFORCEWithBaseline(REINFORCE):
    """
    An implementation of REINFORCE with a learned approximate baseline.
    """

    def __init__(self, env: gym.Env,
                 gamma:float = 0.99,
                 hidden_layer: int =0,
                 value_layer: int=128,
                 lr = 1e-3,
                 v_lr = 1e-3,
                 value_scale = 0.5,
                 norm_reward: bool = False,
                 seed: int = 42):

        super().__init__(env=env,
                         gamma=gamma,
                         hidden_layer=hidden_layer,
                         lr=lr,
                         norm_reward=norm_reward,
                         seed=seed)

        self.value = ValueNetwork(env.observation_space.shape[0],
                                  value_layer)

        self.value_optim = optim.Adam(self.value.parameters(), lr=v_lr)
        self.value_loss = nn.MSELoss()
        self.value_scale = value_scale

        # baseline is state dependent
        self.states = []

    def run_policy(self, render: bool = False):
        """
        Run a full trajectory of the environment with the provided policy.
        """
        self.obs, self.info = self.env.reset()
        self.rewards = []
        self.log_probs = []
        self.states = []

        done = False
        while not done:

            # used only for rendering environments
            if render:
                time.sleep(0.02)

            # obs from gym are provided as numpy arrays,
            # these have to be changed to have a batch-size of 1 ([1, obs_dim])
            obs = torch.tensor(self.obs, dtype=torch.float32)

            # store the state for the value function
            self.states.append(obs)

            policy_obs = obs.unsqueeze(0)

            # agent will pass logits
            logs = self.agent_policy(policy_obs)

            # sample action
            #   1. first generate a probability distribution over the actions
            #   2. sample an action over that distribution,
            action_prob = torch.distributions.Categorical(logits=logs)
            action = action_prob.sample()

            # append log probability for calculation
            self.log_probs.append(action_prob.log_prob(action))

            # gym does not take tensors - obtain action item
            self.obs, reward, terminated, truncated, info = self.env.step(action.item())
            done = terminated or truncated

            self.rewards.append(reward)

        return sum(self.rewards)


    def update(self,):
        """
        Calculate the returns from an episode and update the policy.
        This method overrides the original update method for vanilla REINFORCE.
        """
        if not self.rewards:
            print("Running episode")
            self.run_policy()

        # build total reward list starting from 0 discounted return G
        returns = []
        G = 0

        # loop through rewards backwards
        for r in reversed(self.rewards):
            G = r + self.gamma * G

            # build rewards relative to step
            returns.insert(0, G)

        returns = torch.tensor(returns)

        # incorporate value estimates
        states = torch.stack(self.states)
        values = self.value(states)

        advantages = returns - values.detach()

        if self.norm_reward:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # find the individual loss for each step
        policy_loss = []
        for step in range(len(self.log_probs)):
            log_prob = self.log_probs[step]

            # calculate advantage
            advantage = advantages[step]

            # maximise rewards (pytorch default minimization)
            policy_loss.append(-log_prob * advantage)

        # update policy and value networks
        self.optim.zero_grad()
        self.value_optim.zero_grad()

        # calculate losses
        policy_loss = torch.stack(policy_loss).sum()
        value_loss = self.value_loss(values, returns)

        (policy_loss + self.value_scale*value_loss).backward()

        self.optim.step()
        self.value_optim.step()

        # reset storage
        self.log_probs = []
        self.rewards = []
        self.states = []
        self.obs, self.info = self.env.reset()