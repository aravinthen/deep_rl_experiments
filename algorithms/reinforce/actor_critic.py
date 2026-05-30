#
# Program name: actor_critic.py
# Description: A temporal-difference version of REINFORCE with a baseline.
#

import torch
from algorithms.reinforce.baselines import REINFORCEWithBaseline
import gymnasium as gym

class ActorCritic(REINFORCEWithBaseline):
    """
    Implementation of a temporal-difference version of REINFORCE with a baseline,
    where the baseline itself is a Critic network.
    """
    def __init__(self,
                 env: gym.Env,
                 gamma:float = 0.99,
                 hidden_layer: int =0,
                 value_layer: int=128,
                 lr = 1e-3,
                 v_lr = 1e-3,
                 value_scale = 0.5,
                 norm_reward: bool = False,
                 seed: int = 42):

        # much of the basic functionality of reinforce with baselines remains
        super().__init__(env=env,
                         gamma=gamma,
                         hidden_layer=hidden_layer,
                         value_layer=value_layer,
                         lr=lr,
                         v_lr=v_lr,
                         value_scale=value_scale,
                         norm_reward=norm_reward,
                         seed=seed)

        self.episode_count = 0

    def update_step(self, obs, next_obs, reward, done, log_prob):
        """
        A temporal difference update for the original reinforce method. Performs a critic update
        and actor update for just one transition.
        """

        # tensorize relevant values
        reward = torch.tensor(reward, dtype=torch.float32)
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        next_obs = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0)

        value = self.value(obs).squeeze()
        next_value = self.value(next_obs).squeeze()

        if done:
            td_target = reward
            self.episode_count+=1
        else:
            td_target = reward + self.gamma * next_value.detach()

        # calculate td error
        td_error = td_target - value

        # calculate losses
        actor_loss = -log_prob * td_error.detach()
        critic_loss = self.value_loss(value, td_target.detach())

        loss = actor_loss + self.value_scale*critic_loss

        # backprop
        self.optim.zero_grad()
        self.value_optim.zero_grad()

        loss.backward()

        self.optim.step()
        self.value_optim.step()

    def train_episode(self):
        """
        Carry out a training episode. This is similar to the run_policy rollout method, but explicitly trains the
        networks using the update step rather than the normal step.
        """

        obs, info = self.env.reset()

        episode_reward = 0
        done = False
        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            logs = self.agent_policy(obs_tensor)

            action_prob = torch.distributions.Categorical(logits=logs)
            action = action_prob.sample()
            log_prob = action_prob.log_prob(action)

            next_obs, reward, terminated, truncated, info = self.env.step(action.item())
            done = terminated or truncated

            self.update_step(obs, next_obs, reward, done, log_prob)
            episode_reward += reward

            obs = next_obs

        return episode_reward

if __name__ == '__main__':

    algo = ActorCritic(env=gym.make('CartPole-v1'),
                       gamma=0.99,
                       lr=5e-3,
                       v_lr=1e-3,
                       value_scale=1.0,
                       hidden_layer=128,
                       value_layer=128,
                       norm_reward=True,
                       seed=100)

    # train for 100 episodes
    for ep in range(0,100):
        algo.train_episode()

