# --- 请勿移除这些库 ---
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from functools import reduce
from datetime import datetime, timedelta, timezone

from freqtrade.strategy import (
    BooleanParameter,  # 布尔型参数
    CategoricalParameter,  # 类别型参数
    DecimalParameter,  # 小数型参数
    IStrategy,  # 策略基类
    IntParameter,  # 整数型参数
    RealParameter,  # 实数型参数
)

# --- 在此添加您需要导入的库 ---
import talib.abstract as ta  # 导入TA-Lib库
import freqtrade.vendor.qtpylib.indicators as qtpylib  # 导入Freqtrade的指标库
import logging  # 日志记录

logger = logging.getLogger(__name__)  # 获取日志记录器


class MyChannelBreakout2(IStrategy):
    """
    我的通道突破策略2 (My Channel Breakout Strategy 2)
    从 TradingView Pine Script 翻译而来
    """

    INTERFACE_VERSION = 3  # Freqtrade 策略接口版本
    # 策略的最佳时间周期
    timeframe = "15m"  # 根据您的需求调整时间周期

    # 此策略是否可以做空?
    can_short: bool = True

    # 为策略设计的最小投资回报率 (ROI)
    # 如果配置文件包含 "minimal_roi"，此属性将被覆盖
    # 我们使用 custom_stoploss 进行退出，因此 ROI 表可能不经常触发
    minimal_roi = {"0": 100}  # 实际上已禁用，依赖于自定义止损

    # 为策略设计的最佳止损
    # 如果配置文件包含 "stoploss"，此属性将被覆盖
    # 我们使用 custom_stoploss，因此这通常是一个备用值，或者在 custom_stoploss 返回 None 时的初始值
    stoploss = -0.99  # 强制性参数，但 custom_stoploss 通常会覆盖它

    # 追踪止损
    # Pine 脚本有自定义的追踪逻辑，因此 Freqtrade 内置的追踪止损可能会冲突或冗余
    # 我们将在 custom_stoploss 中实现 Pine 的追踪逻辑
    # trailing_stop = False
    # trailing_stop_positive = None
    # trailing_stop_positive_offset = 0.0
    # trailing_only_offset_is_reached = False

    # 自定义止损
    use_custom_stoploss = True

    # 仅对新的 K 线运行 "populate_indicators()"
    process_only_new_candles = True

    # 这些值可以在配置文件中覆盖
    startup_candle_count: int = 205  # n_period (200) + 一些用于移位和 ATR 的缓冲区

    # 可选的订单类型
    order_types = {
        "entry": "limit",  # 入场订单类型
        "exit": "limit",  # 出场订单类型
        "stoploss": "market",  # 止损订单类型
        "stoploss_on_exchange": False,  # 是否在交易所设置止损单
    }

    # 可选的订单有效时间
    order_time_in_force = {"entry": "gtc", "exit": "gtc"}  # gtc: Good 'til Canceled

    # --- 超参数 (Hyperparameters) ---
    # 入场参数
    n_period = IntParameter(150, 2050, default=800, space="buy", optimize=True, load=True, description="通道周期 (长)")
    n_period2 = IntParameter(30, 700, default=100, space="buy", optimize=True, load=True, description="通道周期2 (短)")
    cold = IntParameter(5, 20, default=20, space="buy", optimize=True, load=True, description="下单冷却期 (K线数量)")

    # 出场参数 (用于 custom_stoploss)
    isatrSL = BooleanParameter(default=False, space="sell", optimize=True, load=True, description="是否使用ATR止损")
    atr_period = IntParameter(10, 20, default=14, space="sell", optimize=True, load=True, description="ATR周期")
    atr_multiplier = RealParameter(2.0, 5.0, default=3.0, space="sell", optimize=True, load=True, decimals=1,
                                   description="ATR倍数 (止损)")
    SL = RealParameter(0.01, 0.10, default=0.02, space="sell", optimize=True, load=True, decimals=3,
                       description="止损比例% (例如, 0.02 代表 2%)")  # 从 % 转换为小数

    trailing_stop_interval_pct_param = RealParameter(
        0.005, 0.05, default=0.01, space="sell", optimize=True, load=True, decimals=3,
        description="追踪止损激活盈利百分比 (例如, 0.01 代表 1%)"
    )
    profit_take_retrace_pct_param = RealParameter(
        0.005, 0.05, default=0.01, space="sell", optimize=True, load=True, decimals=3,
        description="浮盈回撤止损百分比 (例如, 0.01 代表 1%)"
    )

    # risk_reward_ratio = RealParameter(1.5, 5.0, default=3.0, space="sell", optimize=True, load=True, description="盈亏比") # 在 Pine 中定义，但未在退出逻辑中使用

    # 此参数来自 Pine Script，但 Freqtrade 处理开仓量的方式不同。
    # 包含它是为了参考，或者如果您手动调整 Freqtrade 的 stake_amount。
    equity_percent_for_entry = RealParameter(
        1.0, 100.0, default=10.0, space="buy", optimize=False, load=True,
        description="信息性: 用于入场的权益百分比 (Pine Script 原版)。不直接被 Freqtrade 的自动开仓量使用。"
    )

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        # 全局最后一次入场时间，用于实现 'cold' 冷却期
        self.global_last_entry_time: pd.Timestamp = pd.Timestamp(0, tz='UTC')
        # 字典，用于存储自定义交易数据（如初始止损价，入场后的最高/最低收盘价）
        # 以 trade.id 作为键，确保唯一性
        self.custom_trade_data = {}

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        向给定的 DataFrame 添加多个指标
        """
        # --- 基于 n_period 计算指标 ---
        dataframe["highhigh"] = dataframe["high"].rolling(self.n_period.value).max()  # n_period周期内最高价的最高价
        dataframe["lowlow"] = dataframe["low"].rolling(self.n_period.value).min()  # n_period周期内最低价的最低价
        dataframe["midLevel"] = (dataframe["highhigh"] - dataframe["lowlow"]) * 0.5 + dataframe[
            "lowlow"]  # n_period通道中轨
        dataframe["upperBand"] = (dataframe["highhigh"] - dataframe["midLevel"]) * 0.618 + dataframe[
            "midLevel"]  # n_period通道上轨 (黄金分割)
        dataframe["lowerBand"] = (dataframe["midLevel"] - dataframe["lowlow"]) * (1 - 0.618) + dataframe[
            "lowlow"]  # n_period通道下轨 (黄金分割)

        # --- 基于 n_period2 计算指标 ---
        dataframe["highhigh2"] = dataframe["high"].rolling(self.n_period2.value).max()  # n_period2周期内最高价的最高价
        dataframe["lowlow2"] = dataframe["low"].rolling(self.n_period2.value).min()  # n_period2周期内最低价的最低价
        dataframe["midLevel2"] = (dataframe["highhigh2"] - dataframe["lowlow2"]) * 0.5 + dataframe[
            "lowlow2"]  # n_period2通道中轨
        dataframe["upperBand2"] = (dataframe["highhigh2"] - dataframe["midLevel2"]) * 0.618 + dataframe[
            "midLevel2"]  # n_period2通道上轨 (黄金分割)
        dataframe["lowerBand2"] = (dataframe["midLevel2"] - dataframe["lowlow2"]) * (1 - 0.618) + dataframe[
            "lowlow2"]  # n_period2通道下轨 (黄金分割)

        # --- ATR ---
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=self.atr_period.value)  # 计算ATR

        # --- 趋势状态 (Trend Status) ---
        # isUpTrendSignal: 连续3次收盘价高于 upperBand
        isUpTrendSignal = (
                (dataframe["close"] > dataframe["upperBand"]) &
                (dataframe["close"].shift(1) > dataframe["upperBand"].shift(1)) &
                (dataframe["close"].shift(2) > dataframe["upperBand"].shift(2))
        )
        # isDownTrendSignal: 连续3次收盘价低于 lowerBand
        isDownTrendSignal = (
                (dataframe["close"] < dataframe["lowerBand"]) &
                (dataframe["close"].shift(1) < dataframe["lowerBand"].shift(1)) &
                (dataframe["close"].shift(2) < dataframe["lowerBand"].shift(2))
        )

        dataframe["trend_status"] = 0  # 初始化趋势状态为0 (中性)
        dataframe.loc[isUpTrendSignal, "trend_status"] = 1  # 上涨趋势信号，状态设为1
        dataframe.loc[isDownTrendSignal, "trend_status"] = -1  # 下跌趋势信号，状态设为-1

        # 向前填充 trend_status 以便在状态改变前保持不变
        dataframe["trend_status"] = dataframe["trend_status"].replace(0, method="ffill").fillna(0)

        # Pine Script 中被注释掉的 trend_status 重置为 0 的逻辑此处未包含:
        # if(trend_status==1)
        #     if(close < midLevel and close[1] < midLevel[1] and close[2] < midLevel[2])
        #         trend_status:=0
        # if(trend_status==-1)
        #     if(close > midLevel and close[1] > midLevel[1] and close[2] > midLevel[2])
        #         trend_status:=0

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        基于TA指标，为给定的dataframe填充入场信号
        """
        # --- 多头入场条件 ---
        # (ta.crossover(close, lowerBand2)) and trend_status==1 and coldcondition==0
        conditions_long = [
            qtpylib.crossed_above(dataframe["close"], dataframe["lowerBand2"]),  # 收盘价上穿短周期通道下轨 lowerBand2
            dataframe["trend_status"] == 1,  # 趋势状态为多头 (1)
            # coldcondition (冷却期) 的检查在 confirm_trade_entry 中处理
        ]
        dataframe.loc[reduce(lambda x, y: x & y, conditions_long), "enter_long"] = 1  # 满足所有条件，设置多头入场信号
        dataframe.loc[reduce(lambda x, y: x & y, conditions_long), "enter_tag"] = "long_突破下轨2_趋势1"  # 为多头入场设置标签

        # --- 空头入场条件 ---
        # (ta.crossunder(close, upperBand2)) and trend_status==-1 and coldcondition==0
        conditions_short = [
            qtpylib.crossed_below(dataframe["close"], dataframe["upperBand2"]),  # 收盘价下穿短周期通道上轨 upperBand2
            dataframe["trend_status"] == -1,  # 趋势状态为空头 (-1)
            # coldcondition (冷却期) 的检查在 confirm_trade_entry 中处理
        ]
        dataframe.loc[reduce(lambda x, y: x & y, conditions_short), "enter_short"] = 1  # 满足所有条件，设置空头入场信号
        dataframe.loc[reduce(lambda x, y: x & y, conditions_short), "enter_tag"] = "short_跌穿上轨2_趋势-1"  # 为空头入场设置标签

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        基于TA指标，为给定的dataframe填充出场信号
        此策略中的出场由 custom_stoploss 处理。
        """
        # dataframe.loc[:, "exit_long"] = 0 # 无明确出场信号，依赖 custom_stoploss
        # dataframe.loc[:, "exit_short"] = 0 # 无明确出场信号，依赖 custom_stoploss
        return dataframe

    def confirm_trade_entry(
            self,
            pair: str,  # 交易对，例如 "BTC/USDT"
            order_type: str,  # 订单类型，例如 "limit"
            amount: float,  # 订单数量
            rate: float,  # 订单价格
            time_in_force: str,  # 订单有效时间，例如 "gtc"
            current_time: datetime,  # 当前K线的时间 (UTC)
            entry_tag: str,  # 入场标签 (来自 populate_entry_trend)
            side: str,  # 交易方向 ('long' 或 'short')
            **kwargs,
    ) -> bool:  # 返回 True 允许交易，False 阻止交易

        # --- 全局冷却期逻辑 (Pine Script 中的 `coldcondition`) ---
        # 计算冷却期的结束时间
        timeframe_minutes = self.timeframe_to_minutes(self.timeframe)  # 将时间周期转换为分钟
        cooldown_duration_candles = self.cold.value  # 从超参数获取冷却期K线数
        cooldown_end_time = self.global_last_entry_time + timedelta(
            minutes=cooldown_duration_candles * timeframe_minutes)

        if current_time < cooldown_end_time:
            # logger.info(f"{pair} - 在 {current_time} 的入场尝试因全局冷却期被阻止。上次入场: {self.global_last_entry_time}. 冷却至: {cooldown_end_time}")
            return False  # 仍处于冷却期

        # 如果允许交易，则更新全局最后入场时间
        # 这模拟了 Pine 中 `coldcondition := cold` 的行为，它在 `strategy.entry` 之后发生。
        # 这里，我们允许入场，然后立即认为冷却期开始。
        self.global_last_entry_time = current_time
        # logger.info(f"{pair} - 允许入场。全局冷却期从 {current_time} 开始。")
        return True

    def custom_stoploss(
            self,
            pair: str,  # 交易对
            trade: "Trade",  # Freqtrade 的交易对象
            current_time: datetime,  # 当前K线的时间 (UTC)
            current_rate: float,  # 当前价格
            current_profit: float,  # 当前盈利百分比
            **kwargs,
    ) -> float:  # 返回止损价格；如果返回 -1，则不设置/更新止损
        """
        自定义止损逻辑，实现 Pine Script 的止损和追踪止损。
        此函数会被频繁调用，因此应保持高效。
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)  # 获取分析后的DataFrame
        if dataframe.empty:
            logger.warning(f"获取 {pair} 的DataFrame失败，在custom_stoploss中返回-1")
            return -1  # 不应发生，但作为保险

        # --- 获取或初始化特定于交易的数据 ---
        trade_data = self.custom_trade_data.get(trade.id)

        if trade_data is None:  # 此交易首次被 custom_stoploss 处理
            trade_data = {}

            # --- 计算初始止损价 ---
            # 使用入场K线的ATR (假设 process_only_new_candles = True)
            # 我们需要找到包含 trade.open_date_utc 的K线或其前一根K线
            try:
                # 获取包含 trade.open_date_utc 的K线，或紧随其前的K线
                entry_candle_iloc = dataframe.index.get_loc(trade.open_date_utc, method='ffill')
                entry_candle = dataframe.iloc[entry_candle_iloc]
                atr_at_entry = entry_candle['atr']

                if pd.isna(atr_at_entry):  # 如果ATR为NaN (例如，在数据非常早期)
                    logger.warning(f"{pair} 交易 {trade.id} 的入场ATR为NaN。使用上一个可用的ATR。")
                    # 回退到当前K线的前一根K线的ATR
                    previous_candle_index = dataframe.index.get_loc(current_time, method='ffill') - 1
                    if previous_candle_index >= 0:
                        atr_at_entry = dataframe['atr'].iloc[previous_candle_index]
                    else:  # 进一步回退
                        atr_at_entry = dataframe['atr'].iloc[-2] if len(dataframe) >= 2 else np.nan

                    if pd.isna(atr_at_entry):
                        logger.error(f"严重: {pair} 交易 {trade.id} 的ATR仍然为NaN。无法设置ATR止损。")
                        return -1  # 无法确定止损价
            except Exception as e:
                logger.error(f"获取 {pair} 交易 {trade.id} 入场ATR时出错: {e}。使用上一个可用的ATR。")
                atr_at_entry = dataframe['atr'].iloc[-2] if len(dataframe) >= 2 else np.nan  # 回退
                if pd.isna(atr_at_entry):
                    logger.error(f"严重: {pair} 交易 {trade.id} 的ATR回退值仍为NaN。无法设置ATR止损。")
                    return -1  # 无法确定止损价

            if trade.is_short:  # 如果是空头交易
                if self.isatrSL.value:  # 如果使用ATR止损
                    stop_loss_distance = self.atr_multiplier.value * atr_at_entry
                else:  # 使用百分比止损
                    stop_loss_distance = trade.open_rate * self.SL.value
                trade_data["stop_price"] = trade.open_rate + stop_loss_distance  # 空头止损价 = 开仓价 + 止损距离
                trade_data["lowest_close_since_entry"] = trade.open_rate  # 为空头初始化入场以来最低收盘价
            else:  # 如果是多头交易
                if self.isatrSL.value:  # 如果使用ATR止损
                    stop_loss_distance = self.atr_multiplier.value * atr_at_entry
                else:  # 使用百分比止损
                    stop_loss_distance = trade.open_rate * self.SL.value
                trade_data["stop_price"] = trade.open_rate - stop_loss_distance  # 多头止损价 = 开仓价 - 止损距离
                trade_data["highest_close_since_entry"] = trade.open_rate  # 为多头初始化入场以来最高收盘价

            self.custom_trade_data[trade.id] = trade_data
            # logger.info(f"{pair} 交易 {trade.id}: 初始止损价设置为 {trade_data['stop_price']:.5f} (入场ATR: {atr_at_entry:.5f})")

        # --- 更新自入场以来的最高/最低收盘价 ---
        if trade.is_short:
            trade_data["lowest_close_since_entry"] = min(
                trade_data.get("lowest_close_since_entry", current_rate), current_rate  # 如果键不存在，则使用 current_rate 初始化
            )
        else:  # 多头
            trade_data["highest_close_since_entry"] = max(
                trade_data.get("highest_close_since_entry", current_rate), current_rate  # 如果键不存在，则使用 current_rate 初始化
            )

        # --- 追踪止损逻辑 (来自 Pine Script) ---
        current_stop_price = trade_data["stop_price"]
        new_potential_stop = current_stop_price

        if trade.is_short:
            # Pine: if close < entry_price * (1 - trailing_stop_interval_pct)
            if current_rate < trade.open_rate * (1 - self.trailing_stop_interval_pct_param.value):
                # Pine: retrace_level = lowest_close_since_entry * (1 + profit_take_retrace_pct)
                retrace_level = trade_data["lowest_close_since_entry"] * (1 + self.profit_take_retrace_pct_param.value)
                # Pine: short_stop_price := math.min(short_stop_price, retrace_level)
                new_potential_stop = min(current_stop_price, retrace_level)  # 止损只向下（对空头有利）移动
        else:  # 多头
            # Pine: if close > entry_price * (1 + trailing_stop_interval_pct)
            if current_rate > trade.open_rate * (1 + self.trailing_stop_interval_pct_param.value):
                # Pine: retrace_level = highest_close_since_entry * (1 - profit_take_retrace_pct)
                retrace_level = trade_data["highest_close_since_entry"] * (1 - self.profit_take_retrace_pct_param.value)
                # Pine: long_stop_price := math.max(long_stop_price, retrace_level)
                new_potential_stop = max(current_stop_price, retrace_level)  # 止损只向上（对多头有利）移动

        # 仅当新的止损价更有利（或相同）时才更新
        # 对于空头，新的止损价应该更低（或等于）当前止损价
        # 对于多头，新的止损价应该更高（或等于）当前止损价
        if (trade.is_short and new_potential_stop <= current_stop_price) or \
                (not trade.is_short and new_potential_stop >= current_stop_price):
            if new_potential_stop != current_stop_price:
                # logger.info(f"{pair} 交易 {trade.id}: 追踪止损从 {current_stop_price:.5f} 更新至 {new_potential_stop:.5f}")
                pass
            trade_data["stop_price"] = new_potential_stop

        self.custom_trade_data[trade.id] = trade_data
        return trade_data["stop_price"]  # 返回计算出的止损价格

    # 可选: 当交易关闭时清理自定义数据
    # 这有助于防止在长时间运行的机器人和大量交易中发生内存泄漏。
    # Freqtrade 通常能正确处理交易对象，但显式清理自定义字典是个好习惯。
    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        # 此方法用于自定义出场信号，而非止损。
        # 如果 Pine Script 有 `strategy.close()` 或 `strategy.exit()` 用于与止损无关的止盈目标，
        # 它们将在此处实现。
        pass  # 此 Pine Script 中没有自定义出场信号，只有基于止损的退出。

    # 当一笔交易被关闭时（无论是通过ROI、SL、TSL、custom_exit等），此方法会被调用
    def trade_closed(self, trade: 'Trade', **kwargs):  # type: ignore
        """
        当一笔交易被关闭时调用。
        清理自定义交易数据。
        """
        if trade.id in self.custom_trade_data:
            # logger.info(f"交易 {trade.id} ({trade.pair}) 已关闭。清理自定义数据。")
            del self.custom_trade_data[trade.id]
        super().trade_closed(trade=trade, **kwargs)