import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

from models import QLearning, ExpectedSARSA
from tilecoding import TileCoder
from itertools import product

np.random.seed(33)


def setup():
    # Setup the environment for Mountain-Car-v0
    env = gym.make("MountainCar-v0")
    env.reset()
    print("Action space:", env.action_space.n)
    print("State space:", env.observation_space.low)

    # value limits per dimension
    value_limits = np.array([env.observation_space.low, env.observation_space.high]).T
    value_limits = list([tuple(value_limits[i]) for i in range(value_limits.shape[0])])

    # Setup the tilecoder
    coder = TileCoder(
        tiles_per_dim=[2] * env.observation_space.shape[0],
        value_limits=value_limits,
        tilings=5,
    )
    return env, coder


def main(env, coder):
    # Setup plots
    fig, ax = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle("Mountain Car Performance", fontsize=16)
    ax = ax.flatten()

    for idx, (eps, alpha) in enumerate(product(epsilons, alphas)):
        ax[idx].set_title(f"eps={eps}, alpha={alpha}")

        print(f"Running Q-Learning with epsilon={eps} and alpha={alpha}")
        qlearner = QLearning(
            env=env,
            coder=coder,
            eps=eps,
            alpha=alpha,
            gamma=1.0,
        )
        returns = qlearner.train(
            num_trials=num_trials, num_episodes_per_trial=episodes_per_trial
        )
        qlearner.plot_performance(label=f"Q-Learning", ax=ax[idx], returns=returns)

        print(f"Running Expected SARSA with epsilon={eps} and alpha={alpha}")
        esarsa = ExpectedSARSA(
            env=env,
            coder=coder,
            eps=eps,
            alpha=alpha,
            gamma=1.0,
        )
        returns = esarsa.train(
            num_trials=num_trials, num_episodes_per_trial=episodes_per_trial
        )
        esarsa.plot_performance(label=f"Expected SARSA", ax=ax[idx], returns=returns)


if __name__ == "__main__":
    epsilons = [0.1, 0.01, 0.001]
    alphas = [1 / 4, 1 / 8, 1 / 16]
    num_trials = 1
    episodes_per_trial = 250

    env, coder = setup()
    main(env, coder)

    plt.savefig("mountain_car_performance.png")
