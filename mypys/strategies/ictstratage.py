from freqtrade.strategy import IStrategy, merge_informative_pair
from freqtrade.strategy import DecimalParameter, IntParameter
from pandas import DataFrame, Series
import talib.abstract as ta
import numpy as np
import datetime


class ICT1MinStrategy(IStrategy):
    # 时间周期配置（1分钟）
    timeframe = '1m'

    # 策略参数（可通过UI调整）
    ema_short = IntParameter(10, 30, default=20, space='buy')
    ema_long = IntParameter(40, 100, default=50, space='buy')
    rr_ratio = DecimalParameter(1.5, 3.0, default=2.0, space='sell')
    liquidity_lookback = IntParameter(20, 50, default=30, space='sell')

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        计算所需指标
        """
        # 计算双EMA判断趋势
        dataframe['ema_short'] = ta.EMA(dataframe, timeperiod=self.ema_short.value)
        dataframe['ema_long'] = ta.EMA(dataframe, timeperiod=self.ema_long.value)

        # 流动性池计算（过去30根K线的最高/最低）
        dataframe['liq_high'] = dataframe['high'].rolling(self.liquidity_lookback.value).max()
        dataframe['liq_low'] = dataframe['low'].rolling(self.liquidity_lookback.value).min()

        # 订单块检测（寻找重要反转K线）
        dataframe['ob_size'] = dataframe['close'] - dataframe['open']
        dataframe['ob_bullish'] = (
                (dataframe['ob_size'] > 0) &
                (dataframe['ob_size'] > dataframe['ob_size'].shift(1)) &
                (dataframe['volume'] > dataframe['volume'].shift(1))
        )
        dataframe['ob_bearish'] = (
                (dataframe['ob_size'] < 0) &
                (dataframe['ob_size'].abs() > dataframe['ob_size'].shift(1).abs()) &
                (dataframe['volume'] > dataframe['volume'].shift(1))
        )

        # 市场结构突破检测
        dataframe['higher_high'] = dataframe['high'] > dataframe['high'].shift(1)
        dataframe['lower_low'] = dataframe['low'] < dataframe['low'].shift(1)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        生成买入/卖出信号
        """
        # 多头条件（示例）
        dataframe.loc[
            (
                    (dataframe['ema_short'] > dataframe['ema_long']) &  # 趋势向上
                    (dataframe['close'] > dataframe['liq_high']) &  # 突破流动性高点
                    (dataframe['ob_bullish']) &  # 出现看涨订单块
                    (dataframe['higher_high'])  # 市场结构突破
            ),
            'enter_long'] = 1

        # 空头条件（示例）
        dataframe.loc[
            (
                    (dataframe['ema_short'] < dataframe['ema_long']) &  # 趋势向下
                    (dataframe['close'] < dataframe['liq_low']) &  # 突破流动性低点
                    (dataframe['ob_bearish']) &  # 出现看跌订单块
                    (dataframe['lower_low'])  # 市场结构突破
            ),
            'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        生成退出信号
        """
        # 使用风险回报比设置止盈
        for trade_tag in dataframe['enter_long'].unique():
            if trade_tag != 0:
                long_condition = dataframe['enter_long'] == trade_tag
                entry_price = dataframe.loc[long_condition, 'close'].values[0]
                atr = ta.ATR(dataframe, timeperiod=14)

                # 止盈止损计算
                take_profit = entry_price + (self.rr_ratio.value * atr)
                stop_loss = entry_price - atr

                # 设置退出条件
                dataframe.loc[long_condition, 'exit_long'] = np.where(
                    (dataframe['high'] >= take_profit) |
                    (dataframe['low'] <= stop_loss), 1, 0)

        return dataframe

    # 风险控制参数
    use_custom_stoploss = True

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float,  **kwargs) -> float:
        """
        动态止损（示例：追踪止损）
        """
        if current_profit > 0.05:  # 当盈利超过5%时启动追踪止损
            return current_profit - 0.03  # 保留3%的利润
        return 0.02  # 默认2%的固定止损

    # 策略配置
    stoploss = -0.02
    order_types = {
        'entry': 'market',
        'exit': 'market',
        'stoploss': 'market'
    }