# --- 不要移除这些库 ---
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter)

# --- 在这里添加你的库文件 ---
import talib.abstract as ta # talib 仍然是 freqtrade 的核心依赖，这里保留
# 移除了 pandas_ta 和 qtpylib 的导入

# --- 策略特定导入 ---
# (如果需要，在此处添加其他导入)

class VmaCrossBbStrategyZhNoLib(IStrategy):
    """
    Freqtrade 策略，基于 TradingView 脚本:
    "VMA双均线交叉 + VMA布林带策略"
    **此版本不依赖外部库 pandas_ta 和 qtpylib**

    策略逻辑:
    - 入场: 快速 VMA 上穿 慢速 VMA (金叉)。
    - 出场: 快速 VMA 下穿 慢速 VMA (死叉) 或 价格下穿基于慢速 VMA 的布林带上轨。
    - VMA: 使用内部实现的 CMO (钱德动量摆动指标) 来调整波动率。
    - 布林带: 中轨使用慢速 VMA，但标准差基于原始价格计算。
    """

    # 策略接口版本 - 必需
    INTERFACE_VERSION = 3

    # --- 超参数 ---
    # == VMA 设置 ==
    fast_vma_len = IntParameter(5, 50, default=12, space="buy", optimize=True, load=True, description="快速 VMA 周期")
    slow_vma_len = IntParameter(10, 100, default=26, space="buy", optimize=True, load=True, description="慢速 VMA 周期")
    volatility_len = IntParameter(5, 30, default=9, space="buy", optimize=True, load=True, description="波动率周期 (用于 CMO)")
    price_src = CategoricalParameter(['open', 'high', 'low', 'close', 'hl2', 'hlc3', 'ohlc4'], default='close', space="buy", optimize=False, load=True, description="价格来源")

    # == 布林带 设置 (基于慢速VMA) ==
    BBlength = IntParameter(5, 50, default=20, space="buy", optimize=True, load=True, description="布林带标准差周期")
    BBmult = DecimalParameter(1.0, 10.0, default=5.0, decimals=1, space="buy", optimize=True, load=True, description="布林带标准差倍数")

    # == 出场信号参数 ==
    exit_fast_vma_len = IntParameter(5, 50, default=12, space="sell", optimize=True, load=True, description="[出场] 快速 VMA 周期")
    exit_slow_vma_len = IntParameter(10, 100, default=26, space="sell", optimize=True, load=True, description="[出场] 慢速 VMA 周期")
    exit_volatility_len = IntParameter(5, 30, default=9, space="sell", optimize=True, load=True, description="[出场] 波动率周期 (用于 CMO)")
    exit_price_src = CategoricalParameter(['open', 'high', 'low', 'close', 'hl2', 'hlc3', 'ohlc4'], default='close', space="sell", optimize=False, load=True, description="[出场] 价格来源")
    exit_BBlength = IntParameter(5, 50, default=20, space="sell", optimize=True, load=True, description="[出场] 布林带标准差周期")
    exit_BBmult = DecimalParameter(1.0, 10.0, default=50.0, decimals=1, space="sell", optimize=True, load=True, description="[出场] 布林带标准差倍数")

    # --- 策略设置 ---
    minimal_roi = {"0": 10.0}
    stoploss = -0.99
    timeframe = '5m'
    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    startup_candle_count: int = 30 # 确保此值大于或等于 max(slow_vma_len, volatility_len, BBlength)

    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }
    order_time_in_force = {'entry': 'gtc', 'exit': 'gtc'}

    # --- 辅助函数：计算 CMO (不使用 pandas_ta) ---
    def calculate_cmo(self, series: pd.Series, length: int) -> pd.Series:
        """
        使用 Pandas 计算钱德动量摆动指标 (CMO)。

        参数:
            series (pd.Series): 输入的价格序列。
            length (int): CMO 的计算周期。

        返回:
            pd.Series: 计算得到的 CMO 序列。
        """
        if length <= 0:
            return pd.Series(np.nan, index=series.index)
        # 计算价格变化
        delta = series.diff()
        # 计算向上移动 (大于0的变化)
        delta_up = delta.where(delta > 0, 0)
        # 计算向下移动 (小于0的变化的绝对值)
        delta_down = -delta.where(delta < 0, 0) # 注意取反得到正值

        # 计算指定周期内的向上移动之和
        sum_up = delta_up.rolling(window=length, min_periods=1).sum()
        # 计算指定周期内的向下移动之和
        sum_down = delta_down.rolling(window=length, min_periods=1).sum()

        # 计算总移动之和
        sum_total = sum_up + sum_down

        # 计算 CMO
        # 使用 np.where 避免除以零的错误，当 sum_total 为 0 时，CMO 也设为 0 或 NaN
        cmo = np.where(
            sum_total != 0,
            100 * (sum_up - sum_down) / sum_total,
            0 # 或者使用 np.nan 如果你倾向于在无波动时显示 NaN
        )

        return pd.Series(cmo, index=series.index).fillna(0) # 初始 NaN 用 0 填充

    # --- 自定义 VMA 函数 (调用内部 CMO 计算) ---
    def calculate_vma(self, series: pd.Series, length: int, vol_len: int) -> pd.Series:
        """
        计算波动率调整移动平均线 (VMA)。
        使用内部实现的 CMO 计算波动率因子 'k'。
        (此函数逻辑与之前版本基本相同，只是调用了 self.calculate_cmo)
        """
        # 计算 CMO (使用内部实现)
        cmo_val = self.calculate_cmo(series, vol_len)

        # 计算 'k' (波动率因子)
        k = np.abs(cmo_val) / 100.0

        # 计算 'alpha' (动态平滑因子)
        alpha = (2.0 / (length + 1)) * k

        # 迭代计算 VMA
        vma = np.full_like(series, np.nan)
        if not series.empty:
            vma[0] = series.iloc[0]
        for i in range(1, len(series)):
            prev_vma = vma[i-1]
            if np.isnan(prev_vma):
                 prev_vma = series.iloc[i-1] if i > 0 else series.iloc[i]
            vma[i] = alpha.iloc[i] * series.iloc[i] + (1 - alpha.iloc[i]) * prev_vma

        return pd.Series(vma, index=series.index)

    # --- 填充指标 ---
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        计算并添加所需的技术分析指标到 DataFrame 中。
        """
        # -- 计算用于入场信号的指标 --
        src_buy = dataframe[self.price_src.value]
        dataframe['fast_vma'] = self.calculate_vma(src_buy, self.fast_vma_len.value, self.volatility_len.value)
        dataframe['slow_vma'] = self.calculate_vma(src_buy, self.slow_vma_len.value, self.volatility_len.value)

        # 布林带计算 (基于入场参数)
        bb_basis_buy = dataframe['slow_vma']
        bb_stdev_buy = src_buy.rolling(window=self.BBlength.value).std()
        dataframe['bb_upper_buy'] = bb_basis_buy + self.BBmult.value * bb_stdev_buy

        # -- 计算用于出场信号的指标 --
        src_sell = dataframe[self.exit_price_src.value]
        dataframe['exit_fast_vma'] = self.calculate_vma(src_sell, self.exit_fast_vma_len.value, self.exit_volatility_len.value)
        dataframe['exit_slow_vma'] = self.calculate_vma(src_sell, self.exit_slow_vma_len.value, self.exit_volatility_len.value)

        # 出场信号的布林带计算
        bb_basis_sell = dataframe['exit_slow_vma']
        bb_stdev_sell = src_sell.rolling(window=self.exit_BBlength.value).std()
        dataframe['exit_bb_upper'] = bb_basis_sell + self.exit_BBmult.value * bb_stdev_sell

        # --- 计算交叉信号（不使用 qtpylib）---
        # 入场：fast_vma 上穿 slow_vma
        dataframe['vma_cross_above'] = (
            (dataframe['fast_vma'].shift(1) <= dataframe['slow_vma'].shift(1)) &
            (dataframe['fast_vma'] > dataframe['slow_vma'])
        )

        # 出场：fast_vma 下穿 slow_vma
        dataframe['vma_cross_below'] = (
            (dataframe['exit_fast_vma'].shift(1) >= dataframe['exit_slow_vma'].shift(1)) &
            (dataframe['exit_fast_vma'] < dataframe['exit_slow_vma'])
        )

        # 出场：价格下穿 BB 上轨
        price_series_sell = dataframe[self.exit_price_src.value]
        dataframe['price_cross_below_bb_upper'] = (
            (price_series_sell.shift(1) >= dataframe['exit_bb_upper'].shift(1)) &
            (price_series_sell < dataframe['exit_bb_upper'])
        )

        return dataframe

    # --- 填充入场信号 ---
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        根据技术分析指标，填充 DataFrame 中的入场信号列 ('enter_long', 'enter_short')。
        """
        dataframe.loc[
            dataframe['vma_cross_above'], # 直接使用计算好的交叉列
            'enter_long'] = 1

        dataframe.loc[:, 'enter_short'] = 0
        return dataframe

    # --- 填充出场信号 ---
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        根据技术分析指标，填充 DataFrame 中的出场信号列 ('exit_long', 'exit_short')。
        """
        dataframe.loc[
            (dataframe['vma_cross_below'] | dataframe['price_cross_below_bb_upper']), # 合并两个退出条件
            'exit_long'] = 1

        dataframe.loc[:, 'exit_short'] = 0
        return dataframe

# --- 可选: 绘图配置 ---
# VmaCrossBbStrategyZhNoLib.plot_config = {
#     "main_plot": {
#         "fast_vma": {"color": "green"},
#         "slow_vma": {"color": "blue"},
#         "bb_upper_buy": {"color": "dimgray", "linestyle": "dashed"},
#     },
#     # "subplots": { ... } # 可以添加子图，例如显示内部计算的 CMO
# }