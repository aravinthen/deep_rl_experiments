from algorithms.classical.sarsa import SARSA
from algorithms.classical.ql import QLearning, DoubleQLearning
from utils.markov_decision_process import GridWorld
from utils.experiments import Experiment

# config for gridworld
config = {
    "test_steps": 12500,
    "test_interval": 500,
    "eval_episodes": 100,
    "eval_steps": 200,
    "mdp_kwargs": {
        "shape": (5,5),
        "forbidden_list": [1,2,3,4,6,7,8,13,14],
        "noise": (-5, 5),
        "seed": 0,
    },
    "agent_kwargs": {
        "alph": 0.1,
        "eps": 0.1,
        "gam": 0.99,
    }
}

seeds = [i for i in range(10)]

experiment = Experiment(config)

results = {}
# SARSA
x, mean, std = experiment.multi_seed_experiment(SARSA, GridWorld, seeds)
results["SARSA"] = (x, mean, std)

# Q-Learning
x, mean, std = experiment.multi_seed_experiment(QLearning, GridWorld, seeds)
results["Q-Learning"] = (x, mean, std)

# Double Q-Learning
x, mean, std = experiment.multi_seed_experiment(DoubleQLearning, GridWorld, seeds)
results["Double Q-Learning"] = (x, mean, std)

Experiment.plot_experiments(results)