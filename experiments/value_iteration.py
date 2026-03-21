from algorithms.classical import ValueIteration
from utils.markov_decision_process import MDP

problem = MDP(num_states=10,
              num_actions=5,
              reward_range=(-5,1),
              seed=10)

value_iteration = ValueIteration(mdp=problem,
                                 convergence=1e-5)

max_iterations = 1000
for i in range(max_iterations):
    value_iteration.delta = 0
    for state in range(problem.num_states):
        old_v = value_iteration.value[state]
        value_iteration.update(state)

        value_iteration.delta = max(value_iteration.delta,
                                    abs(value_iteration.value[state] - old_v))

    print(f"Step {i} - Convergence: {value_iteration.delta}")
    if value_iteration.delta < value_iteration.convergence:
        break

print(value_iteration.value)