from freqtrade.strategy import IStrategy, merge_informative_pair
from freqtrade.strategy import DecimalParameter, IntParameter
from typing import Dict, List
from datetime import datetime
import talib.abstract as ta
import pandas as pd
import numpy as np


class SMC_FVG_Strategy(IStrategy):
    timeframe = '1m'
    stoploss = -0.99

    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'limit',
        'stoploss_on_exchange': False
    }
    use_exit_signal = True
    ignore_buying_expired_candle_after = 300

    fvg_length = IntParameter(3, 10, default=3, space='buy')
    median_filter_buffer = DecimalParameter(0.005, 0.03, default=0.01, space='buy')
    profit_target = DecimalParameter(0.005, 0.03, default=0.01, space='sell')

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe = self.detect_structures(dataframe)
        # 初始化信号列（防止后续KeyError）
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0

        # 修正FVG检测逻辑：使用滚动窗口计算前三根极值
        # 初始化信号列避免KeyError
        if 'enter_long' not in dataframe.columns:
            dataframe['enter_long'] = 0
        if 'enter_short' not in dataframe.columns:
            dataframe['enter_short'] = 0

        # 修正FVG检测逻辑：使用滚动窗口计算前三根极值
        dataframe['bullish_fvg'] = self.detect_bullish_fvg(dataframe)
        dataframe['bearish_fvg'] = self.detect_bearish_fvg(dataframe)

        dataframe = self.calculate_median_level(dataframe)
        # 修正入场价计算：使用Pandas原生操作
        dataframe['entry_price'] = np.nan  # 先初始化为空值

        dataframe = self.detect_structures(dataframe)
        dataframe['bullish_fvg'] = self.detect_bullish_fvg(dataframe)
        dataframe['bearish_fvg'] = self.detect_bearish_fvg(dataframe)
        dataframe = self.calculate_median_level(dataframe)
        # 仅当信号发生时记录入场价格
        entry_condition = (dataframe['enter_long'] == 1) | (dataframe['enter_short'] == 1)
        dataframe.loc[entry_condition, 'entry_price'] = dataframe['open'].shift(-1)

        # 前向填充有效入场价
        dataframe['entry_price'] = dataframe['entry_price'].ffill()

        return dataframe

    def detect_structures(self, df: pd.DataFrame) -> pd.DataFrame:
        window = self.fvg_length.value * 2
        df['structure_high'] = df['high'].rolling(window, center=True).max().shift(1)
        df['structure_low'] = df['low'].rolling(window, center=True).min().shift(1)
        return df

    def detect_bullish_fvg(self, df: pd.DataFrame) -> pd.Series:
        # 修正：前三根最高价小于当前最低价
        max_high = df['high'].rolling(3).max().shift(1)
        return (max_high < df['low']).astype(int)

    def detect_bearish_fvg(self, df: pd.DataFrame) -> pd.Series:
        # 修正：前三根最低价大于当前最高价
        min_low = df['low'].rolling(3).min().shift(1)
        return (min_low > df['high']).astype(int)

    def calculate_median_level(self, df: pd.DataFrame) -> pd.DataFrame:
        df['median_level'] = (df['structure_high'] + df['structure_low']) / 2
        return df

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # 统一止盈逻辑（移除重复方法）
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0

        # 多头止盈
        dataframe.loc[
            dataframe['close'] >= dataframe['entry_price'] * (1 + self.profit_target.value),
            'exit_long'
        ] = 1

        # 空头止盈
        dataframe.loc[
            dataframe['close'] <= dataframe['entry_price'] * (1 - self.profit_target.value),
            'exit_short'
        ] = 1

        return dataframe

    def custom_price(self, dataframe: pd.DataFrame) -> pd.Series:
        return dataframe['open'].shift(-1)

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float,  ** kwargs) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1]

        if trade.is_short:
            return (last_candle['structure_high'] - current_rate) / current_rate
        else:
            return (current_rate - last_candle['structure_low']) / current_rate

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: str,
                 side: str,  ** kwargs) -> float:
        return 3.0

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # 先执行父类方法初始化列
        super().populate_entry_trend(dataframe, metadata)

        # 定义入场条件（示例逻辑，需根据策略需求调整）
        bullish_condition = (
                (dataframe['bullish_fvg'] == 1) &
                (dataframe['close'] > dataframe['median_level'] * (1 + self.median_filter_buffer.value))
        )

        bearish_condition = (
                (dataframe['bearish_fvg'] == 1) &
                (dataframe['close'] < dataframe['median_level'] * (1 - self.median_filter_buffer.value))
        )

        # 设置信号
        dataframe.loc[bullish_condition, 'enter_long'] = 1
        dataframe.loc[bearish_condition, 'enter_short'] = 1

        return dataframe