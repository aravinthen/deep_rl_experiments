#
# Program name: vanilla.py
# Description: Implementation of REINFORCE (vanilla policy gradient)
#

import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import time

class Policy(nn.Module):
    """
    A neural network representing a simple generic policy.
    """

    def __init__(self, obs: int, action:int, hidden:int):
        super().__init__()

        # agent policy network
        if hidden == 0:
            self.model = nn.Sequential(
                nn.Linear(obs, action),
                nn.Softmax(dim=-1)
            )
        else:
            self.model = nn.Sequential(
                nn.Linear(obs, hidden),
                nn.ReLU(),
                nn.Linear(hidden, action),
                nn.Softmax(dim=-1)
            )

    def forward(self, obs: torch.Tensor):
        """
        Predicts actions when given an observation tensor.
        """
        return self.model(obs)

class REINFORCE:
    """
    A full implementation of the REINFORCE algorithm from Sutton and Barto.
    Built to work with a Gym interface.
    """

    def __init__(self, env: gym.Env, gamma:float = 0.99, hidden_layer=0, seed: int = 42):

        # seeds
        self.seed = seed
        torch.manual_seed(self.seed)

        # specify environment and agent
        self.env = env
        self.gamma = gamma
        self.agent_policy = Policy(env.observation_space.shape[0],
                                   env.action_space.n,
                                   hidden_layer)

        # optimizer for gradient generation
        self.optim = optim.Adam(self.agent_policy.parameters(), lr = 1e-3)

        # state information
        self.log_probs = []
        self.rewards = []
        self.obs, self.info = self.env.reset(seed=self.seed)

    def run_policy(self, render: bool = False):
        """
        Run a full trajectory of the environment with the provided policy.
        """
        self.obs, self.info = self.env.reset()
        self.rewards = []
        self.log_probs = []

        done = False
        while not done:

            # used only for rendering environments
            if render:
                time.sleep(0.02)

            # obs from gym are provided as numpy arrays,
            # these have to be changed to have a batch-size of 1 ([1, obs_dim])
            obs = torch.tensor(self.obs, dtype=torch.float32).unsqueeze(0)

            # agent will pass logits
            probs = self.agent_policy(obs)

            # sample action
            #   1. first generate a probability distribution over the actions
            #   2. sample an action over that distribution,
            action_prob = torch.distributions.Categorical(probs)
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

        # find the individual loss for each step
        policy_loss = []
        for step in range(len(self.log_probs)):
            log_prob = self.log_probs[step]
            G = returns[step]

            # maximise rewards (pytorch default minimization)
            policy_loss.append(-log_prob * G)

        # update network
        self.optim.zero_grad()

        # note to self: building a tensor from scratch breaks grad_fn
        torch.stack(policy_loss).sum().backward()
        self.optim.step()

        # reset storage
        self.log_probs = []
        self.rewards = []
        self.obs, self.info = self.env.reset()

if __name__ == '__main__':
    reinforce = REINFORCE(env=gym.make('CartPole-v1'),
                          seed = 42)

    for episode in range(10):
        reward = reinforce.run_policy()
        reinforce.update()

        print(f"Episode {episode} finished with reward {reward}.")
