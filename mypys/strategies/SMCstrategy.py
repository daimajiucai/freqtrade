from freqtrade.strategy import IStrategy, merge_informative_pair
from typing import Dict, List
from pandas import DataFrame
import talib.abstract as ta


class SMCScalp(IStrategy):
    # 基础参数设置
    timeframe = '1m'#继承，必要
    minimal_roi = {"0": 1}  # 由自定义退出逻辑覆盖

    # 风险管理
    stoploss = -0.99  # 禁用常规止损，使用自定义限价止损#继承，必要
    use_custom_stoploss = True#继承，必要
    risk_per_trade = 0.02  # 每笔交易风险2%#继承，必要

    # 订单类型设置#继承，必要
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'limit',  # 限价止损单
        'stoploss_on_exchange': False
    }

    # 启用退出信号
    use_exit_signal = True #继承，必要
    exit_profit_only = False #继承，必要

    # n根k线后退出限价单
    ignore_buying_expired_candle_after = 10

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # SMC基础指标（示例用EMA，需替换实际SMC逻辑）
        dataframe['ema_fast'] = ta.EMA(dataframe, timeperiod=9)
        dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=21)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 多头入场条件
        dataframe.loc[
            (dataframe['ema_fast'] > dataframe['ema_slow']),
            'enter_long'] = 1

        # 空头入场条件
        dataframe.loc[
            (dataframe['ema_fast'] < dataframe['ema_slow']),
            'enter_short'] = 1
        return dataframe

    # 新增必须的退出趋势方法
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        即使使用custom_exit也必须实现该方法
        此处设置空退出信号（通过custom_exit控制实际退出）
        """
        # 必须保留退出信号列
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        return dataframe

    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs) -> str:
        # 根据持仓方向处理退出
        if trade.is_short:
            # 空头退出逻辑（盈利为正表示价格下跌）
            if current_profit > 0.0015:  # 空头止盈
                return 'short_profit'
            if current_profit < -0.0015:  # 空头止损
                return 'short_stop'
        else:
            # 多头退出逻辑保持不变
            if current_profit > 0.0015:
                return 'long_profit'
            if current_profit < -0.0015:
                return 'long_stop'
        return None

    def custom_entry_price(self, pair: str, current_time: 'datetime', proposed_rate: float,
                           entry_tag: str, **kwargs) -> float:
        # 限价入场价格计算（示例：当前收盘价下方0.05%挂单）
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_close = dataframe['close'].iloc[-1]
        # 根据入场方向设置偏移
        if 'long' in entry_tag:
            return last_close * 0.9995  # 多头在下方挂单
        elif 'short' in entry_tag:
            return last_close * 1.0005  # 空头在上方挂单
        return proposed_rate

    def custom_exit_price(self, pair: str, trade: 'Trade',
                          current_time: 'datetime', proposed_rate: float,
                          exit_tag: str, **kwargs) -> float:
        # 根据退出类型和方向定价
        if trade.is_short:
            if exit_tag == 'short_profit':
                return trade.open_rate * 0.9985  # 空头止盈价
            elif exit_tag == 'short_stop':
                return trade.open_rate * 1.0015  # 空头止损价
        else:
            if exit_tag == 'long_profit':
                return trade.open_rate * 1.0015  # 多头止盈
            elif exit_tag == 'long_stop':
                return trade.open_rate * 0.9985  # 多头止损
        return proposed_rate

    def custom_stake_amount(self, pair: str, current_time: 'datetime', current_rate: float,
                            proposed_stake: float, min_stake: float, max_stake: float,
                            entry_tag: str, **kwargs) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        entry_price = self.custom_entry_price(pair, current_time, current_rate, entry_tag)

        # 根据方向计算止损价
        if 'short' in entry_tag:
            stoploss_price = entry_price * 1.0015  # 空头止损在上方
        else:
            stoploss_price = entry_price * 0.9985  # 多头止损在下方

        risk_per_share = abs(entry_price - stoploss_price)
        dollar_risk = self.wallets.get_total_stake_amount() * self.risk_per_trade
        position_size = dollar_risk / risk_per_share

        return max(min(position_size, max_stake), min_stake)

    def leverage(self, pair: str, current_time: 'datetime', current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: str,
                 side: str, **kwargs) -> float:
        # 禁用杠杆
        return 1.0