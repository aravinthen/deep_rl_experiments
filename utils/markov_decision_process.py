#
# Program name: markov_decision_process.py
# Description:  A customizable Markov decision process class. 
#
import numpy as np

class MDP:
    """
    This is a testbed class for trying out classical reinforcement learning
    algorithms. 
    """
    def __init__(self,
                 num_states: int,
                 num_actions: int,
                 num_terminal: int = 0,
                 reward_range = (-0.1,0.1),
                 seed = 42):

        # set seed - should only happen once!
        np.random.seed(seed)

        self.S = num_states + num_terminal
        self.A = num_actions

        # just for the
        self.num_states = num_states
        self.num_terminal = num_terminal

        # terminal states mask
        self.terminal = np.zeros(self.S, dtype=bool)
        self.terminal[num_states:] = True

        # set an initial_states as being equally likely
        self.initial_dist = np.zeros(self.S)
        self.initial_dist[:num_states] = 1/num_states
        
        # transitions look-up tensor
        self.P = self._set_transitions()
        
        # reward lookup tensor
        self.R = self._set_reward(reward_range)

        # assign a current state
        self.state = self.reset()

    def _set_transitions(self,):
        """
        Generate a transition matrix.
        """

        # generate exactly as according to MDP definition
        # first index: current state
        # next index: action
        # last index: next state
        P = np.zeros((self.S, self.A, self.S))

        for s in range(self.S):
            # handle terminal states: self-loop onto same state
            if self.terminal[s]:
                P[s, :, s] = 1.0
                continue

            for a in range(self.A):
                # generate random probabilities for next state
                probs = np.random.random(self.S)
                probs /= probs.sum()

                # probability assigned for each next state
                P[s, a] = probs

        return P

    def _set_reward(self, reward_range):
        """
        Fully define a reward tensor.
        """
        # generate reward assignment in bulk
        R = np.random.uniform(reward_range[0],
                              reward_range[1],
                              size=(self.S, self.A, self.S))

        # assign terminal rewards as per definition in Sutton and Barto,
        # that is, zero reward on terminal state
        for s in range(self.S):
            if self.terminal[s]:
                R[s, :, :] = 0

        return R

    def reset(self):
        self.state = np.random.choice(self.S, p=self.initial_dist)
        return self.state

    def sample_action(self):
        return np.random.randint(self.A)

    def expected_reward(self, s, a):
        return np.dot(self.P[s, a], self.R[s, a])

    def step(self, action):
        """
        Following the standard Gym style output.
        """

        current_state = self.state

        if self.terminal[current_state]:
            return current_state, 0, True

        # generate observation+reward output
        probs = self.P[current_state, action]
        next_state = np.random.choice(self.S, p=probs)
        step_reward = self.R[current_state, action, next_state]
        is_done = self.terminal[next_state]

        self.state = next_state

        return next_state, step_reward, is_done

if __name__=='__main__':

    # basic variables required to specify an MDP
    states = 5
    terminal_states = 1
    actions = 4

    # note that the probabilities will be generated automatically within the class.
    m = MDP(states,
            actions,
            num_terminal=terminal_states,
            seed=1)

    total_reward = 0
    s = m.reset()

    done = False
    while not done:
        action = m.sample_action()
        state, reward, done = m.step(action)
        total_reward += reward

        print(action, state, done)

    print(total_reward)
