"""
This file contains the code for Q-learning, Exptected SARSA, REINFORCE, and Actor-Critic.
We use a linear function approximator to estimate Q-values for each state-action pair
such that Q(s,a) = w[a]^T * x[s] where w[a] is the weight vector for action a and 
x[s] is the feature vector for state s.
"""

import numpy as np
from typing import List
from gymnasium import Env
from tilecoding import TileCoder
from tqdm import tqdm


def softmax(x, temp=1.0, mask=None) -> np.ndarray:
    if np.isnan(x).any():
        raise ValueError("x contains NaN")

    # only consider actions where mask is 1
    if mask is not None:
        x = x * mask

    # subtract max for numerical stability
    x = x - np.max(x)
    exp_x = np.exp(x / temp)
    return exp_x / np.sum(exp_x)


class Agent:
    env: Env
    coder: TileCoder
    alpha: float
    gamma: float
    eps: float

    def __init__(
        self, env: Env, coder: TileCoder, alpha: float, gamma: float, eps: float
    ):
        self.env = env
        self.coder = coder
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

        # initialize parameters of weight matrix for Q-value function
        # to values uniformly between −0.001 and 0.001
        self.W = np.random.uniform(-0.001, 0.001, (coder.n_tiles, env.action_space.n))

    def select_action(self, state, mask, greedy=False) -> int:
        """Selects an action using the epsilon greedy policy.

        First computes the action value estimates for the current state using the
        weight matrix. With probability epsilon, selects a random action. Otherwise,
        selects the action with the highest action value estimate. greedy flag can be
        used to override the epsilon behavior and always select the greedy action.

        Args:
            state (np.ndarray): the current state
            mask (np.ndarray): a binary mask for the actions
            greedy (bool, optional): whether to select the greedy action. Defaults to False.

        Returns:
            int: the selected action
        """
        Q_sa = np.matmul(self.W.T, state) * mask
        if greedy:
            return np.argmax(Q_sa)

        if np.random.rand() < self.eps:
            return np.random.choice(np.where(mask == 1)[0])
        else:
            return np.argmax(Q_sa)

    def update(self, s, a, r, s_prime, a_prime, done, mask=None) -> None:
        raise NotImplementedError

    def train(self, num_trials, num_episodes_per_trial) -> List[List[float]]:
        """Trains the agent using the given number of trials and episodes per trial.

        Args:
            num_trials (int): the number of trials
            num_episodes_per_trial (int): the number of episodes per trial

        Returns:
            List[List[float]]: the returns for each episode in each trial
        """
        returns = []
        for i in range(num_trials):
            if i % 10 == 0:
                print(
                    f"Trial {i+1}/{num_trials} for {self.__name__} with eps={self.eps} and alpha={self.alpha}"
                )
            self.reset()
            trial_returns = []
            for _ in range(num_episodes_per_trial):
                total_return = self.train_episode()
                trial_returns.append(total_return)
            returns.append(trial_returns)
        return returns

    def train_episode(self, use_mask=False, is_train=True) -> float:
        """Trains the agent for a single episode."""
        state, info = self.env.reset()
        dummy_mask = np.ones(self.env.action_space.n)
        done = False
        total_return = 0

        while not done:
            mask = info["action_mask"] if use_mask else dummy_mask
            action = self.select_action(self.coder[state], mask)

            next_state, reward, done, truncated, info = self.env.step(action)

            mask = info["action_mask"] if use_mask else dummy_mask
            next_action = self.select_action(self.coder[next_state], mask)

            if is_train:
                self.update(
                    self.coder[state],
                    action,
                    reward,
                    self.coder[next_state],
                    next_action,
                    done,
                    mask,
                )

            state = next_state
            total_return += reward * self.gamma**self.env._elapsed_steps

            if truncated:
                break

        return total_return

    def plot_performance(label: str, ax: int, returns: List[List[float]]) -> None:
        """Plots the performance of the agent over the trials.

        Plots the average return on the Y-axis and the episode number on the X-axis
        as well as the interquartile range of the returns over the trials.

        Args:
            label (str): the label for the plot
            ax (int): the axis to plot on
            returns (List[List[float]]): the returns for each episode in each trial
        """
        returns = np.array(returns)
        mean_returns = np.mean(returns, axis=0)
        q1_returns = np.percentile(returns, 25, axis=0)
        q3_returns = np.percentile(returns, 75, axis=0)

        ax.plot(mean_returns, label=f"Mean ({label})")
        ax.fill_between(
            range(len(mean_returns)),
            q1_returns,
            q3_returns,
            alpha=0.3,
            label=f"IQR ({label})",
        )
        ax.set_xlabel("Episode")
        ax.set_ylabel("Return")
        ax.legend()

    def reset(self):
        # Resets the weight matrix to random values between −0.001 and 0.001
        self.W = np.random.uniform(
            -0.001, 0.001, (self.coder.n_tiles, self.env.action_space.n)
        )


class QLearning(Agent):
    __name__ = "Q-Learning"

    def update(self, s, a, r, s_prime, a_prime, done, mask=None) -> None:
        """Updates the weight matrix using the Q-learning update rule:

        w[a] = w[a] + alpha * (r + gamma * max_a' Q(s', a') - Q(s, a)) * x[s]

        Args:
            s (np.ndarray): the current state
            a (int): the current action
            r (float): the reward
            s_prime (np.ndarray): the next state
            a_prime (int): the next action
            done (bool): whether the episode is done
            mask (np.ndarray, optional): a binary mask for the actions. Defaults to None.
        """
        Q_sa = self.W.T.dot(s)
        Q_sa_prime = self.W.T.dot(s_prime)

        if done:
            target = r
        else:
            target = r + self.gamma * np.max(Q_sa_prime)

        delta = target - Q_sa[a]
        self.W[:, a] += self.alpha * delta * s


class ExpectedSARSA(Agent):
    __name__ = "Expected SARSA"

    def update(self, s, a, r, s_prime, a_prime, done, mask=None) -> None:
        """Updates the weight matrix using the Expected SARSA update rule:

        w[a] = w[a] + alpha * (r + gamma * sum_a' pi(a'|s') * Q(s', a') - Q(s, a)) * x[s]

        Args:
            s (np.ndarray): the current state
            a (int): the current action
            r (float): the reward
            s_prime (np.ndarray): the next state
            a_prime (int): the next action
            done (bool): whether the episode is done
            mask (np.ndarray, optional): a binary mask for the actions. Defaults to None.
        """
        Q_sa = self.W.T.dot(s)
        Q_sa_prime = self.W.T.dot(s_prime)

        if done:
            target = r
        else:
            target = r + self.gamma * softmax(Q_sa_prime, temp=self.eps).dot(Q_sa_prime)

        delta = target - Q_sa[a]
        self.W[:, a] += self.alpha * delta * s
