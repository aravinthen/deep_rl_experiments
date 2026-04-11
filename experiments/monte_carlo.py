from algorithms.classical.mc import MonteCarlo
from utils.markov_decision_process import MDP, GridWorld

seed = 42
n_s = 10
n_a = 5
n_t = 1

# value iteration
problem = GridWorld(shape=[5,5],
                    forbidden_list = [1, 2, 3, 4, 6, 7, 8, 13, 14],
                    noise=[0,0],
                    seed=seed)

first_visit_mc = MonteCarlo(mdp=problem)
first_visit_mc.run(100000)
print(first_visit_mc.policy.reshape([5,5]))