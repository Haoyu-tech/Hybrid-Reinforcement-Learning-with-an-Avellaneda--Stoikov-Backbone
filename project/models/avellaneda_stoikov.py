"""
Avellaneda-Stoikov (AS) Market Making Backbone
参考: Avellaneda & Stoikov (2008), Quantitative Finance 8(3):217-224
"""

import numpy as np


class AvellanedaStoikov:
    """
    AS 做市商骨干模型。

    参数
    ----
    gamma     : 风险厌恶系数
    sigma     : 中间价波动率估计
    k         : 订单流深度敏感度（成交强度衰减）
    A         : 基准成交强度
    horizon   : 报价时间窗口 T
    tick_size : 最小价格变动单位
    """

    def __init__(
        self,
        gamma: float = 0.1,
        sigma: float = 1.0,
        k: float = 1.5,
        A: float = 140.0,
        horizon: float = 1.0,
        tick_size: float = 0.01,
    ):
        self.gamma = gamma
        self.sigma = sigma
        self.k = k
        self.A = A
        self.horizon = horizon
        self.tick_size = tick_size

    def reservation_price(self, mid, inventory, time_remaining, sigma=None):
        """
        库存偏斜后的预留价格:
            r_t = S_t - I_t * gamma * sigma^2 * (T - t)
        """
        s = sigma if sigma is not None else self.sigma
        tau = max(float(time_remaining), 0.0) * self.horizon
        return mid - inventory * self.gamma * (s ** 2) * tau

    def optimal_spread(self, time_remaining, sigma=None):
        """
        Guéant 近似最优价差:
            Delta* = gamma*sigma^2*(T-t) + (2/gamma)*ln(1 + gamma/k)
        """
        s = sigma if sigma is not None else self.sigma
        tau = max(float(time_remaining), 0.0) * self.horizon
        spread = (self.gamma * s ** 2 * tau
                  + (2.0 / self.gamma) * np.log(1.0 + self.gamma / self.k))
        return max(spread, self.tick_size)

    def fill_intensity(self, offset):
        """成交强度: lambda(delta) = A * exp(-k * delta)"""
        return self.A * np.exp(-self.k * offset)

    def quotes(self, mid, inventory, time_remaining, sigma=None):
        """
        生成 AS 买卖报价。
        返回: (bid, ask, reservation_price, spread)
        """
        s = sigma if sigma is not None else self.sigma
        r = self.reservation_price(mid, inventory, time_remaining, s)
        delta = self.optimal_spread(time_remaining, s)
        half = delta / 2.0
        bid = self._tick(r - half)
        ask = self._tick(r + half)
        if bid >= ask:
            bid = self._tick(r - self.tick_size)
            ask = self._tick(r + self.tick_size)
        return bid, ask, r, delta

    def update_sigma(self, new_sigma, alpha=0.1):
        """指数移动平均更新波动率"""
        self.sigma = (1 - alpha) * self.sigma + alpha * new_sigma

    def _tick(self, price):
        return round(price / self.tick_size) * self.tick_size

    def __repr__(self):
        return (f"AvellanedaStoikov(gamma={self.gamma}, sigma={self.sigma:.4f}, "
                f"k={self.k}, A={self.A}, horizon={self.horizon})")
