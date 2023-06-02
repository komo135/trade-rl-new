import numpy as np


class Trade:
    """

    action -> -1: shot, 0: no position, 1:long
    """
    init_asset = 100000
    risk = 1

    def __init__(self, symbol, x, price, high, low, atr, get_action):
        self.symbol = symbol
        self.s = -1
        self.x = x
        self.price = price
        self.high = high
        self.low = low
        self.atr = atr
        self.get_action = get_action

        self.asset = self.init_asset

        self.growth = []
        self.pip = []
        self.total_pip = 0

        self.pips = np.zeros((len(self.symbol), 1))
        self.losscuts = []

        self.epsilon = 1.

    def init_val(self, length=None, train=False):
        self.s = (self.s + 1) if self.s + 1 < len(self.symbol) else 0
        self.growth = []
        self.pip = []
        self.total_pip = 0
        self.asset = self.init_asset
        self.losscuts = []

        if (length and self.s == 0) or train:
            self.pips = np.zeros((len(self.symbol), length))

    def calc_pip(self, price, i, old_i, old_a, min_pip, stop_loss, spread, position_size):
        if old_a != 0:
            pip = int(((price[i] - price[old_i]) * old_a - spread) if min_pip > -self.risk else -stop_loss)
            profit = pip * position_size
            asset = self.asset + profit

            self.asset = asset
            self.total_pip += pip
            self.pip.append(pip)
            self.pips[self.s, i] += pip

    def calc_position_size(self, atr, i, losscut, risk):
        """

        :param atr: average true range
        :param i: now index
        :param losscut: losscut
        :param risk: risk
        :return: position_size, stop_loss
        """
        stop_loss = np.clip(atr[i] * 2, losscut // 2, losscut * 3)
        self.risk = stop_loss / (losscut * 5)
        position_size = int((self.asset * risk) / stop_loss)
        position_size = np.minimum(position_size, 500 * 200 * 100)
        position_size = float(np.maximum(position_size, 0))

        return position_size, stop_loss

    def step(self, start, end, sticky_action=5, spread=10, train=False):
        position_state = np.zeros((1, 30, 2), dtype=np.float32)
        reward_state = np.zeros((1, 30, 8), dtype=np.float32)

        action, policy, state = 0, np.zeros((1, 1)), [0, 0]

        skip = sticky_action

        self.init_val(end - start, train)
        x, price, high, low, atr = (self.x[self.s, start:end], self.price[self.s, start:end],
                                    self.high[self.s, start:end], self.low[self.s, start:end],
                                    self.atr[self.s, start:end])
        x_shape = x.shape
        add_x_shape = (*x_shape[:-1], position_state.shape[-1])
        add_x = np.zeros(add_x_shape, dtype=np.float32)
        x = np.concatenate([x, add_x], axis=-1)
        del add_x

        losscut_base = np.mean(self.atr[self.s]) * 2
        pip_scaler = losscut_base * 5
        account_risk = 0.05

        scale_spread = spread / pip_scaler
        scale_pips = ((np.roll(price, -1) - price) / pip_scaler)[:-1]

        # initial position state
        pips, win_pips, lose_pips = [], [], []
        sum_reward, max_reward, drawdown, acc, win, lose, sr, er, ev, pf = [0] * 10
        win_len, lose_len, pip_len = 0, 0, 0
        total_win, total_lose = 0, 0
        is_target, is_stop = True, True
        limit = 0

        # initial trade state
        old_i, old_a, trade_len = 0, 0, 0
        now_reward, old_now_reward, now_drawdown, min_pip = [0] * 4
        now_action_drawdown = 0
        position_size, stop_loss = self.calc_position_size(atr, 0, losscut_base, account_risk)

        for i in range(0, end - start - 1):
            done, reward = 1, 0

            position_state[0, :-1] = position_state[0, 1:]
            position_state[0, -1] = np.round([old_a, now_drawdown], 3)

            if is_target or is_stop:
                if action != 0:
                    reward_state[0, -1] = np.round([
                        old_a, drawdown, sum_reward, ev, er, pf, acc, float(now_reward > 0)], 3)

                x[i, :, -add_x_shape[-1]:] = position_state.copy()
                state = [x[[i]], reward_state.copy()]

                action = self.get_action(state, train)
                skip = 5
                now_action_drawdown = now_drawdown

            if old_a != action or is_stop:
                if old_a != 0:
                    self.calc_pip(price, i, old_i, old_a, min_pip, stop_loss, spread, position_size)
                    pips.append(self.pip[-1] / pip_scaler)
                    pip_len += 1
                    if pips[-1] > 0:
                        win_pips.append(pips[-1])
                        total_win += pips[-1]
                        win_len += 1
                    elif pips[-1] < 0:
                        lose_pips.append(pips[-1])
                        total_lose += pips[-1]
                        lose_len += 1

                    if pip_len >= 30 and win_len and lose_len:
                        acc = np.mean(np.array(pips) > 0)
                        win = np.mean(win_pips) if len(win_pips) != 0 else 0
                        lose = np.mean(lose_pips) if len(lose_pips) != 0 else 0

                        win, lose = (win * acc), (lose * (1 - acc))
                        ev_l = win, lose
                        ev = ev_l[0] + ev_l[1]
                        er = ev / np.abs(ev_l[1])
                        ev *= 10
                        pf = np.sum(win_pips) / np.abs(np.sum(lose_pips))

                    sum_reward = np.clip(np.sum(pips) / 4, -1, 2)
                    max_reward = np.maximum(sum_reward, max_reward)
                    drawdown = np.clip(sum_reward - max_reward, -1, 0)

                    reward_state[0, -1] = np.round([
                        old_a, drawdown, sum_reward, ev, er, pf, acc, float(now_reward > 0)], 3)
                    reward_state[0, :-1] = reward_state[0, 1:]
                    reward_state[0, -1] *= 0

                # initial trade state
                position_state *= 0
                now_reward, old_now_reward, now_drawdown, min_pip = [0] * 4
                now_action_drawdown = 0
                old_i, old_a, trade_len = i, action, 0
                position_size, stop_loss = self.calc_position_size(atr, i, losscut_base, account_risk)

            if action == 0:
                is_target = True if skip == 0 else False
                is_stop = False
                skip -= 1
            elif action != 0:
                trade_len += 1
                scale_pips[i] = scale_pips[i] * action - (scale_spread if trade_len == 0 else 0)

                lp = (low[i] - spread - price[old_i] if action == 1 else
                      price[old_i] - (high[i] + spread) if action == -1 else [0])
                lp = np.round(lp / pip_scaler, 4)
                min_pip = np.minimum(lp, min_pip)

                now_reward += scale_pips[i]
                sum_reward += scale_pips[i] / 4
                now_drawdown = np.clip(now_drawdown + scale_pips[i] / self.risk, -1, 0)

                is_target = (now_drawdown - now_action_drawdown) <= -0.1
                is_stop = min_pip <= -self.risk or now_drawdown <= -1

            if is_target or is_stop:
                limit += 1
                if action != 0 and (pip_len >= 30 and win and lose):
                    acc = (win_len + (1 if now_reward > 0 else 0)) / (pip_len + 1)
                    now_r = now_reward
                    if now_reward > 0:
                        now_win = win + (now_r - win) / (win_len + 1)
                        now_lose = lose

                        now_total_win = total_win + now_r
                        now_total_lose = total_lose
                    else:
                        now_lose = lose + (now_r - lose) / (lose_len + 1)
                        now_win = win

                        now_total_win = total_win
                        now_total_lose = total_lose + now_r

                    now_win, now_lose = now_win * acc, now_lose * (1 - acc)
                    ev_l = now_win, now_lose
                    ev = ev_l[0] + ev_l[1]
                    er = ev / np.abs(ev_l[1])
                    ev *= 10
                    pf = now_total_win / np.abs(now_total_lose)

                reward = 0.1 if now_reward > 0 else 0

                if drawdown <= -0.25 or limit == 1000:
                    done = 0
                    if drawdown <= -0.25:
                        reward = -1

                    reward_state *= 0
                    pips, win_pips, lose_pips = [], [], []
                    sum_reward, max_reward, drawdown, acc, win, lose, sr, er, ev, pf = [0] * 10
                    win_len, lose_len, pip_len = 0, 0, 0
                    total_win, total_lose = 0, 0
                    limit = 0

                    sum_reward = now_reward / 4
                    drawdown = np.clip(sum_reward - max_reward, -0.24, 0)
                else:
                    sum_reward = np.clip(sum_reward, -1, 2)
                    drawdown = np.clip(sum_reward - max_reward, -0.24, 0)

                yield state[0], state[1], action, reward, done

    def trade(self, start: int, end: int, sticky_action=1, spread=10):
        self.s = -1

        asset = []
        for _ in range(len(self.symbol)):
            step = self.step(start, end, sticky_action, spread)
            _ = [_ for _ in step]
            asset.append(self.asset / self.init_asset)
        self.asset = np.mean(asset)
