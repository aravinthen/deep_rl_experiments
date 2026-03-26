from algorithms.classical.sarsa import SARSA
from utils.markov_decision_process import MDP

seed = 42
n_s = 10
n_a = 5
n_t = 1

# value iteration
problem = MDP(num_states=n_s,
              num_actions=n_a,
              num_terminal=n_t,
              reward_range=(-1,1),
              seed=seed)

s = SARSA(mdp=problem)
print(s.policy())
s.run_steps(200000)
print(s.policy())

# mc: [2 1 1 0 3 0 3 3 2 4 0]
# dp: [2 4 4 0 3 0 3 1 2 0 0]