import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

from models import QLearning, ExpectedSARSA, Agent
from tilecoding import TileCoder
from itertools import product
from concurrent.futures import ProcessPoolExecutor, wait, ALL_COMPLETED

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


def run(idx, eps, alpha, env, coder, Algorithm, num_trials, episodes_per_trial):
    try:
        print(f"Running {Algorithm.__name__} with epsilon={eps} and alpha={alpha}")
        qlearner = Algorithm(
            env=env,
            coder=coder,
            eps=eps,
            alpha=alpha,
            gamma=1.0,
        )
        returns = qlearner.train(num_trials, episodes_per_trial)
        return (returns, Algorithm.__name__, idx)

    except Exception as e:
        print(e)


def main(env, coder):
    # Setup plots
    fig, ax = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle("Mountain Car Performance", fontsize=16)
    ax = ax.flatten()

    executor = ProcessPoolExecutor(18)
    futures = []

    for idx, (eps, alpha) in enumerate(product(epsilons, alphas)):
        ax[idx].set_title(f"eps={eps}, alpha={alpha}")
        for Algorithm in [QLearning, ExpectedSARSA]:
            futures.append(
                executor.submit(
                    run,
                    idx,
                    eps,
                    alpha,
                    env,
                    coder,
                    Algorithm,
                    num_trials,
                    episodes_per_trial,
                )
            )

    wait(futures, return_when=ALL_COMPLETED)
    res = [f.result() for f in futures]

    for returns, name, idx in res:
        Agent.plot_performance(label=name, ax=ax[idx], returns=returns)


if __name__ == "__main__":
    epsilons = [0.1, 0.01, 0.001]
    alphas = [1 / 4, 1 / 8, 1 / 16]
    num_trials = 50
    episodes_per_trial = 1000

    env, coder = setup()
    main(env, coder)

    plt.savefig("mountain_car_performance.png")
