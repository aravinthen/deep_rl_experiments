from algorithms.dp import (ValueIteration,
                           PolicyIteration)
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

value_iteration = ValueIteration(mdp=problem,
                                 tolerance=1e-5)
vi_steps = 0
converged = False
while not converged:
    converged = value_iteration.update()
    vi_steps += 1

print("Value iteration policy: ", value_iteration.policy())
print("Value iteration value: ", value_iteration.V)
print("Value iteration steps: ", vi_steps)

print(" ")

# policy iteration
problem = MDP(num_states=n_s,
              num_actions=n_a,
              num_terminal=n_t,
              reward_range=(-1,1),
              seed=seed)

policy_iteration = PolicyIteration(mdp=problem,
                                   tolerance=1e-5)
pi_policy, pi_value, steps = policy_iteration.run()

print("Policy iteration policy: ", pi_policy)
print("Policy iteration value: ", pi_value)
print("Policy iteration steps: ", steps, sum(policy_iteration.eval_steps))