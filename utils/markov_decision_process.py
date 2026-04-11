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
                 terminal_list = None,
                 forbidden_list = None,
                 reward_range = (-0.1,0.1),
                 noise = None,
                 seed = 42):

        # set seed - should only happen once!
        np.random.seed(seed)

        self.noise = noise # used in algorithms to inject noise

        self.S = num_states
        self.A = num_actions

        # just for the
        self.num_states = num_states

        # terminal states mask
        self.terminal = np.zeros(self.S, dtype=bool)
        self.forbidden = forbidden_list if forbidden_list is not None else []

        # explicitly set terminal states only if defined
        if terminal_list is not None:
            self.terminal[terminal_list] = True

        # set an initial_states as being equally likely
        self.initial_dist = np.zeros(self.S)

        # only consider non-terminal states as starting points
        non_terminal = np.where(~self.terminal)[0]
        self.initial_dist[non_terminal] = 1/len(non_terminal)
        
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

        if next_state in self.forbidden:
            return next_state, -10, True

        is_done = self.terminal[next_state]

        self.state = next_state

        return next_state, step_reward, is_done

class GridWorld(MDP):
    """
    A subclass that overrides the transitions and reward function.
    In order to fit with the basic convention described in the MDP class, we keep the state enumeration
    and use conversion functions instead.
    """
    def __init__(self, shape=(5,5), forbidden_list = None, slip_prob=0.01, gamma=0.99, noise=None, seed=42):
        self.shape = shape # shape[0] = row, shape[1] = column
        self.cell_count = shape[0]*shape[1]
        self.gamma = gamma

        # additional actions that determine perpendicular slip chances
        self.slip_prob = slip_prob
        self.actual_step_prob = 1 - slip_prob
        self.slips = {
            0: [2,3],
            1: [2,3],
            2: [0,1],
            3: [0,1]
        }

        # terminal states are hard-coded as (0,0) and (N, N)
        super().__init__(num_states=self.cell_count,
                         num_actions=4,
                         forbidden_list=forbidden_list,
                         terminal_list = [0, self.cell_count - 1],
                         noise = noise,
                         seed = seed)

    def state_to_coord(self, state):
        return divmod(state, self.shape[1])

    def coord_to_state(self, coord):
        return coord[0]*self.shape[1] + coord[1]

    def move(self, state, action):
        """
        Carry out a move within the space and automatically convert to state enums
        """
        row, column = self.state_to_coord(state)
        new_row, new_column = row, column

        # determine action with respect to grid shape
        # 0 - right
        # 1 - left
        # 2 - up
        # 3 - down
        match action:
            case 0:
                new_row = max(row - 1, 0)
            case 1:
                new_row = min(row + 1, self.shape[0] - 1)
            case 2:
                new_column = max(column - 1, 0)
            case 3:
                new_column = min(column + 1, self.shape[1] - 1)

        return self.coord_to_state((new_row, new_column))


    def _set_transitions(self):
        """
        Explicitly set P to represent grid transitions.
        """
        P = np.zeros((self.S, self.A, self.S))

        # for every state,
        for state in range(self.S):
            if self.terminal[state]:
                P[state,:,state] = 1.0
                continue

            # consider the effect of an action on that state
            for action in range(self.A):
                s_actual = self.move(state, action)
                P[state, action, s_actual] += self.actual_step_prob

                # add slip probabilities to perpendicular moves
                for slip in self.slips[action]:
                    slip_state = self.move(state, slip)
                    P[state, action, slip_state] += self.slip_prob / 2

        return P

    def _set_reward(self, reward_range=None):
        """
        Set the exact same reward structure as Sutton and Barto example
        """

        R = np.zeros((self.S, self.A, self.S))

        for state in range(self.S):
            if self.terminal[state]:
                continue

            # set a negative reward for every state but a terminal state
            # (follows Sutton and Barto)
            for action in range(self.A):
                for next_state in range(self.S):
                    if self.P[state, action, next_state] > 0:
                        if next_state in self.forbidden:
                            R[state, action, next_state] = -10
                        else:
                            R[state, action, next_state] = -1

        return R


if __name__=='__main__':

    # basic variables required to specify an MDP
    states = 5
    terminal_states = 1
    actions = 4

    # note that the probabilities will be generated automatically within the class.
    m = MDP(states,
            actions,
            terminal_list=[states-1],
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
