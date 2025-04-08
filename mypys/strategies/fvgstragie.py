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

    plot_config = {
        'main_plot': {
            'close': {'color': 'blue'},
            # 看涨FVG填充区域
            'bullish_fvg_high': {
                'color': 'rgba(0,255,0,0.2)',
                'type': 'line',
                'fill_to': 'bullish_fvg_low',
                'fill_color': 'rgba(0,255,0,0.2)'
            },
            # 看跌FVG填充区域
            'bearish_fvg_low': {
                'color': 'rgba(255,0,0,0.2)',
                'type': 'line',
                'fill_to': 'bearish_fvg_high',
                'fill_color': 'rgba(255,0,0,0.2)'
            }
        }
    }

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        df=dataframe.copy()

        # ===== 1. 先创建所有需要的列 =====
        # 看涨FVG条件
        bullish_condition = (
            (df['high'].shift(1) < df['low'].shift(2)) &
            (df['low'] > df['high'].shift(1))
        )
        # 生成列
        df['bullish_fvg_high'] = np.where(bullish_condition, df['low'].shift(2), np.nan)
        df['bullish_fvg_low'] = np.where(bullish_condition, df['high'].shift(1), np.nan)

        # 看跌FVG条件
        bearish_condition = (
            (df['low'].shift(1) > df['high'].shift(2)) &
            (df['high'] < df['low'].shift(1))
        )
        df['bearish_fvg_high'] = np.where(bearish_condition, df['low'].shift(1), np.nan)
        df['bearish_fvg_low'] = np.where(bearish_condition, df['high'].shift(2), np.nan)

        # ===== 2. 动态填补逻辑 =====
        for fill_type in ['bullish', 'bearish']:
            high_col = f'{fill_type}_fvg_high'
            low_col = f'{fill_type}_fvg_low'
            # 确保列存在
            if high_col not in df.columns or low_col not in df.columns:
                continue  # 跳过未生成的列
            filled = (
                (df['high'] >= df[low_col]) & 
                (df['low'] <= df[high_col])
            ) if fill_type == 'bullish' else (
                (df['low'] <= df[low_col]) & 
                (df['high'] >= df[high_col])
            )
            # 向前填充未填补的区域
            df[high_col] = df[high_col].where(~filled.ffill().astype(bool)).ffill()
            df[low_col] = df[low_col].where(~filled.ffill().astype(bool)).ffill()
        
        # 3. 仅打印有数值的行
        if 'bullish_fvg_high' in df.columns and 'bullish_fvg_low' in df.columns:
            # 过滤非空值
            non_empty = df[['date', 'close', 'bullish_fvg_high', 'bullish_fvg_low']].dropna(
                subset=['bullish_fvg_high', 'bullish_fvg_low'],
                how='any'
            )
            if not non_empty.empty:
                print("非空的 bullish_fvg 行:")
                print(non_empty.tail(20))  # 打印最近20条非空记录
            else:
                print("警告：未找到任何非空的 bullish_fvg 行")
        else:
            print("错误：bullish_fvg 列未生成")

        return df

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:

        return dataframe

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: str,
                 side: str,  ** kwargs) -> float:
        return 1.0

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:

        return dataframe