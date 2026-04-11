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

        # action, specific to sarsa (action has to be stored)
        self.action = None

    def sample_action(self, state):
        """
        Samples action with epsilon greedy policy
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.mdp.A)
        else:
            return np.argmax(self.Q[state])

    def update(self):
        """
        Update step for SARSA.
        """
        state = self.mdp.state

        if self.action is None:
            self.action = self.sample_action(state)

        next_state, reward, done = self.mdp.step(self.action)

        if self.mdp.noise is not None:
            reward += np.random.uniform(*self.mdp.noise)

        # required to stop bootstrapping from terminal states
        if done:
            td_error = reward - self.Q[state, self.action]
            self.Q[state, self.action] = (self.Q[state, self.action] + self.alpha*td_error)
            self.action = None

        else:
            # forward step
            next_action = self.sample_action(next_state)
            td_error = reward + self.gamma*self.Q[next_state, next_action] - self.Q[state, self.action]
            self.Q[state, self.action] = (self.Q[state, self.action] + self.alpha*td_error)

            # store next action
            self.action = next_action

        return done

    def run(self, num_steps):
        """
        Carries out a run of temporal difference learning for the current policy.
        """
        self.mdp.reset()
        for _ in range(num_steps):
            done = self.update()
            if done:
                self.mdp.reset()

    def policy(self):
        """
        Return the policy after the learning updates
        """
        return np.argmax(self.Q, axis=1)