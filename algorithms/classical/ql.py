#
# Program: ql.py
# Description: q-learning implementation
#
import random

import numpy as np

class QLearning:
    """
    A class implementing Q-learning.
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

    def update(self):
        """
        Carry out Q-learning update. This is a little bit more involved than the
        sarsa update in that it requires the q-value of the next state rather
        than the current.
        """
        state = self.mdp.state
        action = self.sample_action(state)
        next_state, reward, done = self.mdp.step(action)

        # this is the key distinguisher between sarsa and q-learning.
        # in q-learning, the max Q over the next state is taken for the update.
        if done:
            td_error = reward - self.Q[state, action]
        else:
            next_max = np.max(self.Q[next_state])
            td_error = reward + self.gamma * next_max - self.Q[state, action]


        self.Q[state, action] = self.Q[state, action] + self.alpha * td_error

        return done

    def run(self, num_steps):
        """
        Carries out a full run of Q-learning
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

class doubleQLearning(QLearning):
    """
    Implementation of double Q learning using previous Q-learning class.
    """
    def __init__(self, mdp, alph=0.5, eps=0.1, gam=0.99):
        super().__init__(mdp, alph=alph, eps=eps, gam=gam)

        # another Q function, used explicitly in the update
        self.Q2 = np.zeros((mdp.S, mdp.A))

    def sample_action(self, state):
        """
        Samples action with epsilon greedy policy
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.mdp.A)
        else:
            return np.argmax(self.Q[state]+ self.Q2[state])

    def update(self):
        """
        Carry out Q-learning update, but with a decoupled Q-function.
        """
        state = self.mdp.state
        action = self.sample_action(state)
        next_state, reward, done = self.mdp.step(action)

        # randomly choose which Q function to update
        if np.random.rand() < 0.5:
            # Q1 update
            if done:
                td_error = reward - self.Q[state, action]
            else:
                action_max = np.argmax(self.Q[next_state])
                target = reward + self.gamma * self.Q2[next_state, action_max]
                td_error = target - self.Q[state, action]

            self.Q[state, action] = self.Q[state, action] + self.alpha * td_error

        else:
            # Q2 update
            if done:
                td_error = reward - self.Q2[state, action]
            else:
                action_max = np.argmax(self.Q2[next_state])
                target = reward + self.gamma * self.Q[next_state, action_max]
                td_error = target - self.Q2[state, action]

            self.Q2[state, action] = self.Q2[state, action] + self.alpha * td_error

        return done