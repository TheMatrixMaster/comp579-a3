import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

from models import QLearning, ExpectedSARSA, Agent, REINFORCE, ActorCritic
from tilecoding import TileCoder
from itertools import product
from concurrent.futures import ProcessPoolExecutor, wait, ALL_COMPLETED

np.random.seed(33)


def setup():
    # Setup the environment
    env = gym.make(env_name)
    env.reset()
    print("Action space:", env.action_space.n)
    print("State space:", env.observation_space.low)

    # value limits per dimension
    value_limits = np.array([env.observation_space.low, env.observation_space.high]).T

    # override the values in limits if infinity is present
    value_limits[value_limits <= -3.4e38] = -10
    value_limits[value_limits >= 3.4e38] = 10

    value_limits = list([tuple(value_limits[i]) for i in range(value_limits.shape[0])])

    # Setup the tilecoder
    coder = TileCoder(
        tiles_per_dim=[6] * env.observation_space.shape[0],
        value_limits=value_limits,
        tilings=8,
    )
    return env, coder


def run(
    idx,
    env,
    eps=0.01,
    alpha=0.001,
    coder=None,
    Algorithm=None,
    num_trials=50,
    episodes_per_trial=1000,
    temp=1.0,
    **kwargs,
):
    try:
        print(
            f"Running {Algorithm.__name__} with epsilon={eps}, alpha={alpha}, temp={temp}"
        )
        algo = Algorithm(
            env=env,
            coder=coder,
            eps=eps,
            alpha=alpha,
            gamma=1.0,
            temp=temp,
            **kwargs,
        )
        returns = algo.train(num_trials, episodes_per_trial)
        return (returns, Algorithm.__name__, idx)

    except Exception as e:
        print(e.with_traceback())


def main_p2(env, coder, num_processes=1):
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle(f"{env_name} Performance", fontsize=16)
    ax = ax.flatten()

    futures = []
    executor = ProcessPoolExecutor(num_processes)

    for idx, Algorithm in enumerate([REINFORCE, ActorCritic]):
        ax[0].set_title(f"fixed temp={temp}")
        ax[1].set_title(f"decreasing temp (initial={initial_temp})")

        futures.extend(
            [
                executor.submit(
                    run,
                    idx=0,
                    env=env,
                    coder=coder,
                    Algorithm=Algorithm,
                    num_trials=num_trials,
                    temp=temp,
                ),
                executor.submit(
                    run,
                    idx=1,
                    env=env,
                    coder=coder,
                    Algorithm=Algorithm,
                    num_trials=num_trials,
                    temp=initial_temp,
                    decay_temp=True,
                    temp_tau=0.999,
                    min_temp=0.1,
                ),
            ]
        )

    wait(futures, return_when=ALL_COMPLETED)
    res = [f.result() for f in futures]

    for returns, name, idx in res:
        Agent.plot_performance(label=name, ax=ax[idx], returns=returns)

    return "policy_based"


def main_p1(env, coder, num_processes=1):
    # Setup plots
    fig, ax = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle(f"{env_name} Performance", fontsize=16)
    ax = ax.flatten()

    executor = ProcessPoolExecutor(num_processes)
    futures = []

    for idx, (eps, alpha) in enumerate(product(epsilons, alphas)):
        ax[idx].set_title(f"eps={eps}, alpha={alpha}")
        for Algorithm in [QLearning, ExpectedSARSA]:
            futures.append(
                executor.submit(
                    run,
                    idx=idx,
                    eps=eps,
                    alpha=alpha,
                    env=env,
                    coder=coder,
                    Algorithm=Algorithm,
                    num_trials=num_trials,
                )
            )

    wait(futures, return_when=ALL_COMPLETED)
    res = [f.result() for f in futures]

    for returns, name, idx in res:
        Agent.plot_performance(label=name, ax=ax[idx], returns=returns)

    return "value_based"


if __name__ == "__main__":
    env_name = "CartPole-v1"
    epsilons = [0.1, 0.01, 0.001]
    alphas = [1 / 4, 1 / 8, 1 / 16]

    temp = 0.15
    initial_temp = 1.0

    num_trials = 1
    episodes_per_trial = 1000

    env, coder = setup()
    # _type = main_p1(env, coder, num_processes=18)
    _type = main_p2(env, coder, num_processes=4)
    plt.savefig(f"{env_name}_performance_{_type}.png")
