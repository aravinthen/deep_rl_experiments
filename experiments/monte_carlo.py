from algorithms.mc import MonteCarlo
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

first_visit_mc = MonteCarlo(mdp=problem)
first_visit_mc.run(200000, 0.999)
print(first_visit_mc.policy)
# [2 4 4 0 3 0 3 1 2 0 0]