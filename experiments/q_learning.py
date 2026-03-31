from algorithms.classical.ql import QLearning, doubleQLearning
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

q = QLearning(mdp=problem)
print(q.policy())
q.run(10000)
print(q.policy())

problem.reset()

dq = doubleQLearning(mdp=problem)
print(dq.policy())
dq.run(10000)
print(dq.policy())
