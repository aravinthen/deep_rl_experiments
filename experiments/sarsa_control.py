from algorithms.classical.sarsa import SARSA
from utils.markov_decision_process import GridWorld

seed = 42
n_s = 10
n_a = 5
n_t = 1

# value iteration
problem = GridWorld(shape=[5,5],
                    forbidden_list = [1, 2, 3, 4, 6, 7, 8, 13, 14],
                    noise=[0,0],
                    seed=seed)

s = SARSA(mdp=problem)

s.run(100000)
print(s.policy().reshape((5,5)))

# mc: [2 1 1 0 3 0 3 3 2 4 0]
# dp: [2 4 4 0 3 0 3 1 2 0 0]