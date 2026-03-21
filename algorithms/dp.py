#
# Program name: dp.py
# Description:  Classical reinforcement learning algorithms.
#
from math import gamma

import numpy as np

class ValueIteration:
    """
    A class implementing value iteration.
    """

    def __init__(self, mdp, tolerance=1e-3, gam=0.99):
        # algorithm parameters
        self.tolerance = tolerance
        self.gamma = gam

        # model and value function for the environment
        self.mdp = mdp
        self.V = np.zeros(mdp.S)
        self.delta = np.inf

    def update(self,):
        # carries out a vectorized bellman update
        V = np.max(np.sum(self.mdp.P * (self.mdp.R + self.gamma*self.V), axis=2), axis=1)
        delta = np.max(np.abs(self.V - V))

        # update attributes
        self.V = V
        self.delta = delta

        # return convergence based on the class
        return delta < self.tolerance

    def policy(self,):
        """
        Extract a policy using the converged value function. The policy is greedy with respect to the provided
        value function.
        """
        policy = np.argmax(np.sum(self.mdp.P * (self.mdp.R + self.gamma*self.V), axis=2), axis=1)
        return policy

class PolicyIteration:
    """
    Carries out the full policy iteration process, a method that uses policy evaluation in tandem with policy
    improvement.
    The policy is a 1D vector that provides the action to take in state S.
    """
    def __init__(self, mdp, gam=0.99, tolerance=1e-3):
        # define environment and policy
        self.mdp = mdp
        self.gamma = gam

        self.tolerance = tolerance

        # the policy - the action that must be taken in a given state (1D vector)
        self.policy = np.random.randint(mdp.A, size=mdp.S)

        # value function - all set to zero (although non-terminal states need not
        # be set such.
        self.V = np.zeros(mdp.S)

        # data
        self.eval_steps = []

    def value_update(self):
        """
        Implement the value function update rule for a given policy as per Sutton and Barto.
        This is essentially the same thing as value iteration, but without carrying out updates over all actions.
        """
        # pair state with action prescribed by the policy - going from P(s, a, s') to P(s, s')
        P_policy = self.mdp.P[np.arange(self.mdp.S), self.policy]
        R_policy = self.mdp.R[np.arange(self.mdp.S), self.policy]

        # calculate V with P and R relating to the policy
        V = np.sum(P_policy * (R_policy + self.gamma * self.V[None, :]), axis=1)

        # enforce terminality as per Sutton and Barto
        V[self.mdp.terminal] = 0.0

        # check convergence
        delta = np.max(np.abs(self.V - V))
        self.V = V

        return delta < self.tolerance

    def policy_evaluation(self):
        """
        Carry out a full sweep of policy evaluation.
        """
        converged = False
        steps = 0
        while not converged:
            converged = self.value_update()
            steps +=1

        self.eval_steps.append(steps)

        return self.V

    def policy_update(self):
        """
        Generate a new policy using the current value function.
        If the policy is stable (that is, if updates cause no significant change in policy, return True.
        """
        new_policy = np.argmax(np.sum(self.mdp.P * (self.mdp.R + self.gamma * self.V[None, None, :]), axis=2), axis=1)
        stable = np.all(new_policy == self.policy)

        self.policy = new_policy
        return stable

    def run(self, max_steps=1000):
        """
        Carry out the full policy iteration loop
        """
        stable = False
        step = 0
        while not stable and step < max_steps:
            self.policy_evaluation()
            stable = self.policy_update()
            step += 1

        return self.policy, self.V, step



