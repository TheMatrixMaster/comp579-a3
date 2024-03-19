"""
This file contains the code for Q-learning, Exptected SARSA, Monte-Carlo REINFORCE, 
and Actor-Critic. We use a linear function approximator to estimate Q-values for each 
state-action pair such that Q(s,a) = w[a]^T * x[s] where w[a] is the weight vector for 
action a and x[s] is the feature vector for state s.
"""

import pdb
import numpy as np
from typing import List
from gymnasium import Env
from tilecoding import TileCoder


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
    temperature: float
    decay_temperature: bool
    temperature_tau: float
    min_temperature: float

    def __init__(
        self,
        env: Env,
        coder: TileCoder,
        alpha: float,
        gamma: float,
        eps: float,
        temp: float = 1.0,
        decay_temp: bool = False,
        temp_tau: float = 0.995,
        min_temp: float = 0.1,
    ):
        self.env = env
        self.coder = coder
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.temperature = temp
        self.decay_temperature = decay_temp
        self.temperature_tau = temp_tau
        self.min_temperature = min_temp

        # initialize parameters of weight matrix for Q-value function
        # to values uniformly between −0.001 and 0.001
        self.W = np.random.uniform(-0.001, 0.001, (coder.n_tiles, env.action_space.n))

    def select_action(self, state, mask) -> int:
        """Selects an action using the epsilon greedy policy with self.eps

        Args:
            state (np.ndarray): the current state
            mask (np.ndarray): a binary mask for the actions

        Returns:
            int: the selected action
        """
        Q_sa = np.matmul(self.W.T, state) * mask
        if np.random.rand() < self.eps:
            return np.random.choice(np.where(mask == 1)[0])
        else:
            return np.argmax(Q_sa)

    def update(self, s, a, r, s_prime, a_prime, done, mask=None) -> None:
        raise NotImplementedError

    def update_temperature(self) -> None:
        """Updates the temperature using the decay schedule."""
        self.temperature = max(
            self.min_temperature, self.temperature * self.temperature_tau
        )

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
                    f"Trial {i+1}/{num_trials} for {self.__name__} with eps={self.eps}, alpha={self.alpha}, temp={self.temperature}"
                )
            self.reset()
            trial_returns = []
            for _ in range(num_episodes_per_trial):
                total_return = self.train_episode()
                trial_returns.append(total_return)

                if self.decay_temperature:
                    self.update_temperature()

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

    def reset(self) -> None:
        """Resets the agent's state."""
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
        """Updates the weight matrix using the Expected SARSA update rule where the
        policy used is the epsilon-greedy policy:

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
            pi = np.ones(self.env.action_space.n) * self.eps / self.env.action_space.n
            pi[np.argmax(Q_sa_prime)] += 1 - self.eps
            target = r + self.gamma * np.sum(pi * Q_sa_prime)

        delta = target - Q_sa[a]
        self.W[:, a] += self.alpha * delta * s


class REINFORCE(Agent):
    __name__ = "REINFORCE"
    with_baseline: bool

    def __init__(
        self, with_baseline: bool = False, beta: float = None, gamma=0.99, **kwargs
    ):
        super().__init__(gamma=gamma, **kwargs)
        self.with_baseline = with_baseline

        if with_baseline:
            self.V = np.random.uniform(-0.001, 0.001, (self.coder.n_tiles, 1))
            self.beta = beta if beta is not None else self.alpha

    def select_action(self, state, mask=None) -> int:
        """Selects an action using the softmax policy with self.temperature

        Args:
            state (np.ndarray): the current state
            mask (np.ndarray): a binary mask for the actions

        Returns:
            int: the selected action
        """
        Q_sa = np.matmul(self.W.T, state)
        num_actions = self.env.action_space.n
        return np.random.choice(
            num_actions,
            p=softmax(Q_sa, temp=self.temperature, mask=mask),
        )

    def update(
        self, states: List[np.ndarray], actions: List[int], rewards: List[float]
    ) -> None:
        """Updates the weights using the REINFORCE update rule:

        w[a] = w[a] + alpha * G_t * grad_log_pi(a|s)
        G_t = sum_{t'=t}^T gamma^(t'-t) * r_{t'}

        Args:
            states (List[np.ndarray]): the states
            actions (List[int]): the actions
            rewards (List[float]): the rewards
        """
        assert len(states) == len(actions) == len(rewards)

        G = 0
        for t in range(len(states) - 1, -1, -1):
            G = self.gamma * G + rewards[t]
            target = G
            grad_log_pi = self.grad_log_pi(states[t], actions[t])

            if self.with_baseline:
                delta = target - self.V.T.dot(states[t])
                self.V += (self.beta * delta * states[t])[:, None]
                self.W += self.alpha * delta * grad_log_pi
            else:
                self.W += self.alpha * target * grad_log_pi

    def grad_log_pi(self, state, action) -> np.ndarray:
        """Computes the gradient of the log policy for a given state and action
        with respect to the weights. The gradient is given by:

        grad_log_pi = -x * pi + x if a == a' else -x * pi

        Args:
            state (np.ndarray): the state
            action (int): the action

        Returns:
            np.ndarray: the gradient of the log policy
        """
        pi = softmax(self.W.T.dot(state), temp=self.temperature)
        grad_log_pi = -state[:, None] * pi
        grad_log_pi[:, action] += state
        return grad_log_pi

    def train_episode(self, use_mask=False, is_train=True) -> float:
        """Trains the agent for a single episode."""
        state, info = self.env.reset()
        done = False
        total_return = 0

        states = []
        actions = []
        rewards = []

        while not done:
            action = self.select_action(self.coder[state])
            next_state, reward, done, truncated, info = self.env.step(action)

            states.append(self.coder[state])
            actions.append(action)
            rewards.append(reward)

            state = next_state
            total_return += reward * self.gamma**self.env._elapsed_steps

            if truncated:
                break

        if is_train:
            self.update(states, actions, rewards)

        return total_return

    def reset(self) -> None:
        """Resets the agent's state."""
        super().reset()
        if self.with_baseline:
            self.V = np.random.uniform(-0.001, 0.001, (self.coder.n_tiles, 1))


class ActorCritic(Agent):
    __name__ = "Actor-Critic"
    beta: float

    def __init__(self, beta: float = None, **kwargs):
        super().__init__(**kwargs)

        # learning rate for the critic
        self.beta = beta if beta is not None else self.alpha
        # initialize parameters of the critic (state value function)
        self.V = np.random.uniform(-0.001, 0.001, (self.coder.n_tiles, 1))

    def select_action(self, state, mask=None) -> int:
        """Selects an action using the softmax policy with self.temperature

        Args:
            state (np.ndarray): the current state
            mask (np.ndarray): a binary mask for the actions

        Returns:
            int: the selected action
        """
        Q_sa = np.matmul(self.W.T, state)
        num_actions = self.env.action_space.n
        return np.random.choice(
            num_actions,
            p=softmax(Q_sa, temp=self.temperature, mask=mask),
        )

    def update(self, s, a, r, s_prime, a_prime, done, mask=None) -> None:
        """Updates the weights of the actor and critic using the Actor-Critic update rule:

        V[s] = V[s] + beta * (r + gamma * V[s'] - V[s]) * x[s]
        w[a] = w[a] + alpha * (r + gamma * V[s'] - V[s]) * grad_log_pi(a|s)

        Args:
            s (np.ndarray): the current state
            a (int): the current action
            r (float): the reward
            s_prime (np.ndarray): the next state
            a_prime (int): the next action
            done (bool): whether the episode is done
            mask (np.ndarray, optional): a binary mask for the actions. Defaults to None.
        """
        if done:
            target = r
        else:
            target = r + self.gamma * self.V.T.dot(s_prime)

        delta = target - self.V.T.dot(s)
        self.V += (self.beta * delta * s)[:, None]

        grad_log_pi = self.grad_log_pi(s, a)
        self.W += self.alpha * delta * grad_log_pi

    def grad_log_pi(self, state, action) -> np.ndarray:
        """Computes the gradient of the log policy for a given state and action
        with respect to the weights. The gradient is given by:

        grad_log_pi = -x * pi + x if a == a' else -x * pi

        Args:
            state (np.ndarray): the state
            action (int): the action

        Returns:
            np.ndarray: the gradient of the log policy
        """
        pi = softmax(self.W.T.dot(state), temp=self.temperature)
        grad_log_pi = -state[:, None] * pi
        grad_log_pi[:, action] += state
        return grad_log_pi

    def reset(self):
        """Resets the agent's state."""
        super().reset()
        self.V = np.random.uniform(-0.001, 0.001, (self.coder.n_tiles, 1))
