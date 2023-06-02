import gc
import os
import pickle
import warnings
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from IPython.display import clear_output
from scipy.stats import norm

from rl.env import env
from rl.network.small_net2 import build_model

warnings.filterwarnings('ignore')

tau_dict = {"none": 0, "cpw": 1, "pow": 2, "cvar": 3}


def default_tau(risk=0, train=False):
    risk = risk if not train else np.random.randint(5)
    tau = np.arange(1, 33, dtype=np.float32) / 33
    if risk == 0:
        tau = (tau ** 0.71) / ((tau ** 0.71 + (1 - tau) ** 0.71) ** (1 / 0.71))
    elif risk == 1:
        tau = norm.cdf(norm.ppf(tau) - 0.75)
    elif risk == 2:  # CVaR 10%
        tau *= 0.1
    elif risk == 3:  # CVaR 25%
        tau *= 0.25
    elif risk == 4:  # PoW -2
        tau = 1 - (1 - tau) ** (1 / 3)

    return tau.reshape((1, 32))


class Memory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = np.empty((capacity, 7), dtype=object)
        self.idx = 0
        self.full = False

    @classmethod
    def cast(cls, x: np.ndarray, _dtype) -> np.ndarray:
        return x.astype(_dtype)

    def add(self, state1, state2, action, reward, new_state1, new_state2, done):
        add_list = [state1, state2, action, reward, new_state1, new_state2, done]
        self.memory[self.idx] = add_list
        if self.full is False and self.idx == self.capacity - 1:
            self.full = True
        self.idx = (self.idx + 1) if self.idx + 1 != self.capacity else 0

    def sample(self, batch_size):
        # indexes = np.random.choice(self.capacity if self.full else self.idx, batch_size, replace=False)
        indexes = np.unique(np.random.randint(self.capacity if self.full else self.idx,
                                              size=int(batch_size * 1.5)))[:batch_size]

        states1, states2, actions, rewards, new_states1, new_states2, dones = np.transpose(self.memory[indexes])

        actions = self.cast(actions, np.int32)
        rewards = self.cast(rewards, np.float32)
        dones = self.cast(dones, np.float32)
        states1 = np.concatenate(states1, dtype=np.float32)
        states2 = np.concatenate(states2, dtype=np.float32)
        new_states1 = np.concatenate(new_states1, dtype=np.float32)
        new_states2 = np.concatenate(new_states2, dtype=np.float32)

        states = [states1, states2]
        new_states = [new_states1, new_states2]

        return states, actions, rewards, new_states, dones


class DqnLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        k = 1

        error = y_true - y_pred
        loss_ = tf.where(tf.abs(error) <= k, error ** 2 * 0.5, 0.5 * k ** 2 + k * (tf.abs(error) - k))
        loss_ = tf.reduce_mean(loss_)

        return loss_


class Agent:
    name = "dqn"
    loss = DqnLoss()

    epsilon = 1.
    i = 0

    def __init__(self, model_name, symbol, action_size=2, n=3, lr=1e-7, gamma=0.99,
                 test=8, valid=2, replay_size=1e6, replay_ratio=4, batch_size=128, env_name="discrete_env0"):
        self.model_name = model_name
        self.s = symbol
        self.action_size = action_size
        self.n = n
        self.lr = lr
        self.gamma = gamma
        self.test = test
        self.valid = valid
        self.replay_size = replay_size
        self.replay_ratio = replay_ratio
        self.batch_size = batch_size
        self.env_name = env_name

        self.action_dict = {0: 1, 1: -1, 2: 0}
        self.reverse_action_dict = {1: 0, -1: 1, 0: 2}

        self.gammas = np.array([self.gamma ** i for i in range(n)])

        self.x, self.y, self.low, self.high, self.atr, _ = self.create_env()

        self.model, self.target_model = self.build_model()
        self.target_model.set_weights(self.model.get_weights())

        self.w, self.tw = None, None

        self.memory = Memory(capacity=replay_size)
        self.trade = env[env_name](self.s, self.x, self.y, self.high, self.low, self.atr, self.get_action)
        self.test_trade = env[env_name](self.s, self.x, self.y, self.high, self.low, self.atr, self.get_action)

        self.train_step = np.arange(0, self.x.shape[1] - int(960 * (test + valid)))
        self.test_step = np.arange(self.train_step[-1], self.x.shape[1] - int(960 * valid))
        self.valid_step = np.arange(self.test_step[-1], self.x.shape[1])

        self.total_pips = []
        self.losses = []

    def create_env(self):
        """
        This function load the environment and returns the data
        :return: x, y, low, high, atr, spread - data
        """
        x = np.load("rl/data/x.npy")
        y = np.load("rl/data/target.npy")
        atr = np.load("rl/data/atr.npy").reshape((x.shape[0], -1)).astype(np.int32)

        x = np.round(x, 2)

        spread = y[:, :, 3].reshape((x.shape[0], -1))
        low = y[:, :, 2].reshape((x.shape[0], -1))
        high = y[:, :, 1].reshape((x.shape[0], -1))
        y = y[:, :, 0].reshape((x.shape[0], -1))

        s_list = [self.s] if isinstance(self.s, int) else self.s

        x = x[s_list, :, :, :]
        atr = atr[s_list, :]
        y = y[s_list, :]
        low = low[s_list, :]
        high = high[s_list, :]
        spread = spread[s_list, :]
        print(s_list)

        self.s = np.arange(x.shape[0])

        return x, y, low, high, atr, spread

    def build_model(self):
        state_shape = self.x.shape[-2:]
        model = build_model(self.model_name, state_shape, self.action_size, self.name, False)
        target_model = build_model(self.model_name, state_shape, self.action_size, self.name, False)

        model.compile(optimizer=tf.keras.optimizers.Adam(self.lr, clipvalue=1.))

        return model, target_model

    @tf.function(jit_compile=True)
    def get_q(self, state):
        return self.model(state)

    def get_action(self, state, train):
        if train and np.random.random() < self.epsilon:
            action = np.random.randint(0, self.action_size)
        else:
            q = self.get_q(state)
            action = np.argmax(q, axis=-1)[0]

        return self.action_dict[action]

    @tf.function
    def model_update(self, state, action, reward, new_state, done):
        action = tf.one_hot(action, self.action_size)

        target_a = self.model(new_state)
        target_a = tf.argmax(target_a, axis=-1)
        target_q = self.target_model(new_state)
        target_q = tf.reduce_sum(target_q * tf.one_hot(target_a, self.action_size), axis=-1)
        target_q = reward + self.gamma ** self.n * target_q * done

        with tf.GradientTape() as tape:
            q = self.model(state)
            q = tf.reduce_sum(q * action, axis=-1)
            loss = tf.reduce_mean(self.loss(target_q, q))

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return loss

    def update_target(self, tau=0.01):
        """
        This function updates the target network
        """

        w = np.array(self.model.get_weights())
        tw = np.array(self.target_model.get_weights())

        self.target_model.set_weights(tau * w + (1 - tau) * tw)

    def update_lr(self):
        lr = np.clip(1e-7 + (1e-3 - 1e-7) * (self.i / 10000), 0, 1e-3)
        self.model.optimizer.lr.assign(lr)

    def update(self, batch_size):
        self.update_lr()
        self.i += 1

        state, action, reward, new_state, done = self.memory.sample(batch_size)
        loss = self.model_update(state, action, reward, new_state, done).numpy()
        self.losses.append(loss)

        if self.i % 100 == 0:
            self.update_target()

        return loss

    def evolution(self, loss):
        if self.i % 10000 == 0:
            print(f"train loss = {loss}, now train total pip = {self.trade.total_pip}")
            self.test_trade.trade(self.test_step[0], self.valid_step[-1], 0, 1, 10)
            total_pips = np.sum(self.test_trade.pips, axis=-1)
            print(f"test total pip = {total_pips}")

            self.total_pips.append(np.sum(self.test_trade.pips))
            _ = plt.figure(figsize=(15, 5))
            plt.plot(self.total_pips)
            plt.show()

    def save(self):
        if self.i % 100000 == 0:
            gc.collect()

            if not os.path.exists(f"/traderl/save_agent/{self.name}"):
                os.makedirs(f"/traderl/save_agent/{self.name}")

            model, target_model = self.model, self.target_model
            self.model, self.target_model = None, None
            self.w, self.tw = model.get_weights(), target_model.get_weights()

            with open(f"/traderl/save_agent/{self.name}/agent.pickle", "wb") as f:
                pickle.dump(self, f)
            clear_output()

            self.model, self.target_model = model, target_model

    def start_end(self):
        start = np.random.randint(0, self.train_step[-1] - 12000)
        end = np.random.randint(start + 12000 - 1, self.train_step[-1])

        return start, end, end - start

    def train(self, batch_size=32, spread_scale=0.1):
        spread = np.mean(self.atr[self.trade.s if self.trade.s != -1 else 0]) * np.clip(spread_scale, 0.1, 10)

        states1, states2, actions, rewords, dones = [deque(maxlen=self.n + 1) for _ in range(5)]

        start, end, trade_length = self.start_end()
        step = self.trade.step(start, end, 1, 1, spread, True)

        for _ in range(100000000):
            returns = next(step, None)
            if returns is None:
                print(
                    f"symbol = {self.trade.s}, total_pip = {self.trade.total_pip}, "
                    f"asset = {self.trade.asset}, start = {start}, length = {trade_length}")

                start, end, trade_length = self.start_end()
                del step, states1, states2, actions, rewords
                step = self.trade.step(start, end, 1, 1, spread, True)
                states1, states2, actions, rewords, dones = [deque(maxlen=self.n + 1) for _ in range(5)]
            else:
                state1, state2, action, reward, done = returns
                states1.append(state1)
                states2.append(state2)
                actions.append(self.reverse_action_dict[action])
                rewords.append(reward)
                dones.append(done)

            if len(states1) == self.n + 1:
                if dones[-1] == 0:
                    self.memory.add(states1[-1], states2[-1], actions[-1], rewords[-1], states1[-1], states2[-1], 0)
                    states1, states2, actions, rewords, dones = [deque(maxlen=self.n + 1) for _ in range(5)]
                else:
                    reward = np.sum(np.array(rewords)[:-1] * self.gammas)
                    self.memory.add(states1[0], states2[0], actions[0], reward, states1[-1], states2[-1], dones[0])

                if (self.memory.idx >= 100000 or self.memory.full) and self.memory.idx % self.replay_ratio == 0:
                    self.update(batch_size)
                    loss = self.losses[-1] if self.losses else 0
                    self.evolution(loss)
                    self.save()

                    if self.i > 0:
                        self.epsilon = self.epsilon * 0.999995 if self.epsilon > 0.025 else 0.025
