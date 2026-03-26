#
# Program name: mc.py
# Description:  Monte Carlo reinforcement learning algorithms.
#

import numpy as np

class MonteCarlo:
    """
    A class implementing Monte Carlo simulation
    """
    def __init__(self, mdp, gam=0.99, alpha=0.0, eps=0.1):
        # algorithm parameters
        self.gamma = gam
        self.mdp = mdp

        # generate a random policy to start
        self.policy = np.random.randint(mdp.A, size=mdp.S)
        self.epsilon = eps
        self.alpha = alpha

        # value function
        self.returns = np.zeros((mdp.S, mdp.A))
        self.visits = np.zeros((mdp.S, mdp.A))

        # value function, calculated afterward
        self.Q = np.zeros((mdp.S, mdp.A))
        self.Q[self.mdp.terminal] = 0.0

    def sample_action(self, state):
        """
        Samples an action with epsilon greedy policy.
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.mdp.A)
        else:
            return np.argmax(self.policy[state])

    def generate_trajectory(self):
        """
        Generate a trajectory with the policy defined.
        """
        trajectory = []

        # initialise sampling variables
        state = self.mdp.reset()

        done = False
        while not done:
            action = self.sample_action(state)
            next_state, reward, done = self.mdp.step(action)
            trajectory.append((state, action, reward))

            # continue the process
            state = next_state

        return trajectory

    def backward_pass(self,):
        """
        Carry out a backward pass on a single trajectory.
        """
        G = 0
        visited = set()

        # generate trajectory
        traj = self.generate_trajectory()

        # reversing trajectory for backwards pass
        for t in reversed(range(len(traj))):
            state, action, reward = traj[t]

            # discount backwards
            G = reward + self.gamma * G

            if (state, action) not in visited:
                visited.add((state, action))
                self.returns[state, action] += G
                self.visits[state, action] += 1

                # update Q function
                if self.alpha == 0:
                    alpha =  1 / self.visits[state, action]
                else:
                    alpha = self.alpha

                self.Q[state, action] += alpha*(G - self.Q[state, action])

    def improve_policy(self):
        """
        improve policy by being greedy with respect to Q
        """
        self.policy = np.argmax(self.Q, axis=1)

    def run(self, num_eps, epsilon_decay):
        """
        Run <num> MC trajectory counts and update the value function
        """
        for _ in range(num_eps):
            self.backward_pass()
            self.improve_policy()
            self.epsilon = self.epsilon*epsilon_decay

        return self.policy, self.Q