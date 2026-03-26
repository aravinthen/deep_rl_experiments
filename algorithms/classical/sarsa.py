#
# Program name: sarsa.py
# Description: SARSA implementation
#

import numpy as np

class SARSA:
    """
    A class implementing SARSA.
    """
    def __init__(self, mdp, alph=0.5, eps=0.1, gam=0.99):
        self.gamma = gam
        self.mdp = mdp

        # define policy
        self.epsilon = eps
        self.alpha = alph

        # value function
        self.Q = np.zeros((mdp.S, mdp.A))

    def sample_action(self, state):
        """
        Samples action with epsilon greedy policy
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.mdp.A)
        else:
            return np.argmax(self.Q[state])

    def run_steps(self, timesteps):
        """
        Carries out a run of temporal difference learning for the current policy.
        """

        state = self.mdp.reset()
        action = self.sample_action(state)

        for _ in range(timesteps):
            next_state, reward, done = self.mdp.step(action)
            next_action = self.sample_action(next_state)

            # required to stop bootstrapping from terminal states
            if done:
                td_error = reward - self.Q[state, action]
            else:
                td_error = reward + self.gamma*self.Q[next_state, next_action] - self.Q[state, action]

            self.Q[state, action] = (self.Q[state, action] + self.alpha*td_error)

            if done:
                state = self.mdp.reset()
                action = self.sample_action(state)
            else:
                state = next_state
                action = next_action

    def policy(self):
        """
        Return the policy after the learning updates
        """
        return np.argmax(self.Q, axis=1)