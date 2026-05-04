#
# Program name: experiments.py
# Description: Experiment harness for agents and algorithms
#

import numpy as np
import matplotlib.pyplot as plt

class Experiment:
    def __init__(self, config):
        # parse config dictionary
        self.steps = config["test_steps"]
        self.test_interval = config["test_interval"]

        self.eval_episodes = config["eval_episodes"]
        self.eval_max_steps = config["eval_steps"]

        self.mdp_kwargs = config["mdp_kwargs"]
        self.agent_kwargs = config["agent_kwargs"]

    @staticmethod
    def evaluate_policy(algo, mdp, episodes, max_steps):
        """
        Carries out a policy evaluation on the given mdp.
        """
        policy = algo.policy()

        returns = []
        for _ in range(episodes):
            state = mdp.reset()
            total_reward = 0

            for _ in range(max_steps):
                 action = policy[state]
                 state, reward, done = mdp.step(action)
                 total_reward += reward

                 if done:
                     break

            returns.append(total_reward)

        return np.mean(returns)

    def run_experiment(self, algo_class, mdp_class):
        """
        Run experiments - use standardized interface from before.
        """
        # load from
        mdp = mdp_class(**self.mdp_kwargs)
        algo = algo_class(mdp, **self.agent_kwargs)

        results = []
        steps = []

        # set state to default
        mdp.reset()

        for t in range(1, self.steps+1):
            done = algo.update()

            if done:
                mdp.reset()

            if t % self.test_interval == 0:
                # new initialisation of mdp required for evaluation
                avg_return = self.evaluate_policy(algo,
                                                  mdp.__class__(**self.mdp_kwargs),
                                                  self.eval_episodes,
                                                  self.eval_max_steps)
                results.append(avg_return)
                steps.append(t)

        return np.array(steps), np.array(results)

    def multi_seed_experiment(self, algo_class, mdp_class, seeds):
        """
        Carry out experiments over a given number of seeds
        """
        all_runs = []
        steps = []
        for seed in seeds:
            np.random.seed(seed)
            self.mdp_kwargs["seed"] = seed
            steps, trajectory = self.run_experiment(algo_class, mdp_class)

            all_runs.append(trajectory)

        all_runs = np.array(all_runs)

        # average over the seeds
        mean = np.mean(all_runs, axis=0)
        std = np.std(all_runs, axis=0)

        return steps, mean, std

    @staticmethod
    def plot_experiments(results):
        for label, (x, mean, std) in results.items():
            plt.plot(x, mean, label=label)
            plt.fill_between(x, mean - std, mean + std, alpha=0.2)

        plt.xlabel("Environment Steps")
        plt.ylabel("Average Return")
        plt.title("Learning Curves")
        plt.legend()
        plt.grid()
        plt.show()

