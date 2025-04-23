# --- 必要的库 ---
from freqtrade.strategy import IStrategy, CategoricalParameter, DecimalParameter, IntParameter
from pandas import DataFrame
import talib.abstract as ta
# import pandas_ta as pta # 引入 pandas_ta，有时更方便
import pandas as pd
import functools
import operator
from typing import Optional, Tuple
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.persistence import Trade
from datetime import datetime, timezone

# --- 可选的库 ---
import logging
import numpy as np # 用于处理可能的数据问题

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# 策略名称: 布林带 + RSI + 高周期EMA趋势过滤策略 (Freqtrade版)
# 作者:     (你的名字 / AI生成)
# 基于 TradingView 脚本: "布林+RSI+ATR+HTF EMA Cross过滤 (AI Mod)"
#
# 策略描述 (中文):
# 使用布林带和RSI在基础时间周期寻找入场信号。
# 使用一个更高的时间周期 (HTF) 的双EMA交叉作为趋势过滤器。
# 做多条件: HTF快线EMA > HTF慢线EMA 且 价格下穿布林带下轨 且 RSI < rsi做多阈值
# 做空条件: HTF快线EMA < HTF慢线EMA 且 价格上穿布林带上轨 且 RSI > rsi做空阈值
# 止盈: 固定百分比
# 止损: 固定百分比
# -----------------------------------------------------------------------------

class BollingerRsiHtfEma_CN(IStrategy):
    """
    Freqtrade 策略类，实现了源自 TradingView 的 布林带 + RSI + HTF EMA 交叉过滤逻辑。
    带有中文注释。
    """
    INTERFACE_VERSION = 3 # Freqtrade 策略接口版本

    # === 策略通用配置 ===
    # 策略最佳时间周期 (与 TV 脚本中的基础周期匹配)
    # 建议在 config.json 中设置, 但在这里定义也是好习惯
    timeframe = '1m' # 示例: 设置为你运行此策略的 K 线周期

    # 用于趋势计算的更高时间周期 (对应 TV 的 'trendTimeframe' 输入)
    trend_timeframe = '15m' # 示例: 必须大于 'timeframe'

    # 此策略是否可以做空?
    can_short = True

    # === 止损配置 ===
    # 对应 TV 的 'stopLossPercent' 输入 (5.0% = 0.05)
    # 注意: PineScript 源码注释提到 1%, 但代码实际使用 5%。这里使用 5%。
    stoploss = -0.03

    # === 止盈配置 (ROI - Return On Investment) ===
    # 对应 TV 的 'takeProfitPercent' 输入 (0.5% = 0.005)
    # Freqtrade ROI 表: key=持仓时间(分钟), value=盈利百分比
    # 对于固定止盈，我们设置时间为 0，表示尽快达到目标就止盈
    minimal_roi = {
        "0": 0.005  # 达到 0.5% 盈利即止盈
    }

    # === 追踪止损配置 ===
    # 可选: 禁用追踪止损 (Trailing StopLoss)
    trailing_stop = False
    # trailing_stop_positive = 0.01 # 示例: 盈利 1% 后开始追踪
    # trailing_stop_positive_offset = 0.02 # 示例: 追踪距离设置为 2%
    # trailing_only_offset_is_reached = False # 示例: 是否只在达到盈利偏移后才开始追踪

    # === 保护机制 ===
    # 可选: 如果需要可以添加保护机制
    use_exit_signal = True # 如果在 populate_exit_trend 定义了自定义退出信号，则使用它们
    exit_profit_only = False # 是否只在盈利时退出
    ignore_roi_if_entry_signal = False # 当有新的入场信号时，是否忽略 ROI 退出规则

    # === 超参数 (Hyperparameters) ===
    # 定义用于优化或手动调整的参数

    # -- 布林带 (Bollinger Bands) --
    bb_length = IntParameter(15, 30, default=20, space="buy", optimize=True, load=True, description="布林带周期长度")
    bb_stddev = DecimalParameter(1.5, 3.0, default=2.0, decimals=1, space="buy", optimize=True, load=True, description="布林带标准差倍数")

    # -- RSI --
    rsi_length = IntParameter(2, 14, default=2, space="buy", optimize=True, load=True, description="RSI 周期长度")
    rsi_lower = IntParameter(1, 20, default=5, space="buy", optimize=True, load=True, description="RSI 做多阈值 (<)")
    # 做空参数通常放在 'sell' 空间，但由于 Freqtrade 处理 shorting 时入场逻辑都在 'buy' 空间判断（通过 can_short=True），
    # 将其放在 'buy' 空间通常也可以，或者根据优化目标调整。这里放在 'sell' 空间更符合语义。
    rsi_upper = IntParameter(80, 99, default=95, space="sell", optimize=True, load=True, description="RSI 做空阈值 (>)")

    # -- 高时间周期 EMA (Higher Timeframe EMAs) --
    # 注意: trend_timeframe 本身在上面设置，这里不作为超参数（简化处理）。
    # 如果需要优化时间周期本身，需要更高级的技术。
    ema_fast_len_htf = IntParameter(20, 100, default=50, space="buy", optimize=True, load=True, description="高周期 快速EMA长度")
    ema_slow_len_htf = IntParameter(100, 300, default=200, space="buy", optimize=True, load=True, description="高周期 慢速EMA长度")

    # --- PineScript 中提到但未在核心逻辑中使用的指标 ---
    # atrLength = 14 # ATR 长度 (未在入场/出场逻辑中使用)
    # atrPercentThreshold = 1.0 # ATR 百分比阈值 (未使用)
    # useErFilter = False # ER 过滤器开关 (未使用)
    # erLength = 10 # ER 长度 (未使用)
    # erThreshold = 0.6 # ER 阈值 (未使用)

    # --- Freqtrade 数据提供者和策略上下文 ---
    # process_only_new_candles = False # 设为 True 可以提高性能，但某些依赖历史数据的功能可能受影响

    def informative_pairs(self):
        """
        定义策略需要的其他时间周期的数据 (Informative Pairs)。
        这告诉 Freqtrade 需要下载并提供这些数据。
        """
        pair = self.dp.current_whitelist()[0] # 获取当前交易对列表中的第一个作为示例
        return [(pair, self.trend_timeframe)] # 返回一个列表，包含 (交易对, 时间周期) 元组

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        计算并添加技术指标到 DataFrame。

        Args:
            dataframe (DataFrame): 包含 OHLCV 数据的 DataFrame
            metadata (dict): 附加信息 (例如 'pair')

        Returns:
            DataFrame: 更新后的包含指标的 DataFrame
        """
        # --- 基础时间周期指标 ---

        # 布林带 (Bollinger Bands)
        # 使用 qtpylib 计算布林带，它需要一个价格序列 (这里用 typical_price)
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=self.bb_length.value, stds=self.bb_stddev.value)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid'] # 中轨 (Basis)
        dataframe['bb_upperband'] = bollinger['upper']

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=self.rsi_length.value)

        # ATR (计算保留，但默认逻辑未使用)
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14) # atrLength 固定为 14
        dataframe['atr_perc'] = (dataframe['atr'] / dataframe['close']) * 100

        # --- 高时间周期 (HTF) 指标 ---

        # 获取 HTF 数据
        # informative_pairs() 方法确保了数据可用
        # self.dp (DataProvider) 用于访问这些数据
        informative_dataframe = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.trend_timeframe)

        # 在 HTF 数据上计算 EMA
        informative_dataframe[f'ema_fast_htf'] = ta.EMA(informative_dataframe, timeperiod=self.ema_fast_len_htf.value)
        informative_dataframe[f'ema_slow_htf'] = ta.EMA(informative_dataframe, timeperiod=self.ema_slow_len_htf.value)

        # --- 将 HTF 指标合并回基础时间周期 DataFrame ---
        # 使用 ffill=True 来填充两个时间周期之间的 NaN 空隙
        dataframe = merge_informative_pair(dataframe, informative_dataframe, self.timeframe, self.trend_timeframe, ffill=True)

        # --- 基于 HTF EMA 定义趋势 ---
        # 确保合并后的列存在再使用它们
        htf_fast_ema_col = f'ema_fast_htf_{self.trend_timeframe}'
        htf_slow_ema_col = f'ema_slow_htf_{self.trend_timeframe}'

        # 检查列是否存在，防止启动初期或数据问题导致错误
        if htf_fast_ema_col in dataframe.columns and htf_slow_ema_col in dataframe.columns:
            # 注意：比较可能产生 NaN，如果 HTF EMA 值为 NaN
            dataframe['htf_uptrend'] = (dataframe[htf_fast_ema_col] > dataframe[htf_slow_ema_col]).astype('int')
            dataframe['htf_downtrend'] = (dataframe[htf_fast_ema_col] < dataframe[htf_slow_ema_col]).astype('int')
            # 填充可能因比较 NaN 产生的 NaN 为 0 (无趋势)
            dataframe['htf_uptrend'].fillna(0, inplace=True)
            dataframe['htf_downtrend'].fillna(0, inplace=True)
        else:
            # 如果 HTF 列不存在，则标记为无趋势
            logger.warning(f"交易对 {metadata['pair']} 的 HTF EMA 列 ({htf_fast_ema_col}, {htf_slow_ema_col}) 未找到。"
                           f"将跳过 HTF 趋势计算，标记为无趋势。请检查数据和 informative_pairs 设置。")
            dataframe['htf_uptrend'] = 0
            dataframe['htf_downtrend'] = 0

        # 可选: ER (Efficiency Ratio) 计算 (来自 pinescript, 未在核心逻辑中使用)
        # er_length = 10
        # change = abs(dataframe['close'] - dataframe['close'].shift(er_length))
        # volatility_sum = abs(dataframe['close'] - dataframe['close'].shift(1)).rolling(er_length).sum()
        # dataframe['er_value'] = (change / volatility_sum).replace([np.inf, -np.inf], np.nan) # 处理除零
        # dataframe['er_value'].fillna(0, inplace=True) # 填充 NaN
        # is_trending_strong_er = dataframe['er_value'] > 0.6 # erThreshold = 0.6

        return dataframe


    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        基于技术指标，填充 'enter_long' 和 'enter_short' 列来产生交易信号。

        Args:
            dataframe (DataFrame): 包含指标的 DataFrame
            metadata (dict): 附加信息 (例如 'pair')

        Returns:
            DataFrame: 包含入场信号的 DataFrame
        """

        # === 做多入场条件 ===
        # 1. 价格向下穿越布林带下轨 (使用 qtpylib.crossed_below)
        # 2. RSI < 做多阈值 (rsi_lower)
        # 3. HTF 趋势为上升 (htf_uptrend == 1)

        # 确保 htf_uptrend 列存在且已计算
        if 'htf_uptrend' not in dataframe.columns:
             dataframe['enter_long'] = 0
             dataframe['enter_short'] = 0
             logger.warning(f"交易对 {metadata['pair']} 的 'htf_uptrend' 列不存在，无法生成做多信号。")
             return dataframe

        long_conditions = [] # 创建一个列表存储所有做多条件
        long_conditions.append(dataframe['htf_uptrend'] == 1) # 条件3: HTF 上升趋势
        long_conditions.append(qtpylib.crossed_below(dataframe['close'], dataframe['bb_lowerband'])) # 条件1: 下穿下轨
        long_conditions.append(dataframe['rsi'] < self.rsi_lower.value) # 条件2: RSI 低于阈值
        # 可选的 ER 过滤器:
        # if self.useErFilter.value: # 如果要使用，需要定义 useErFilter 超参数
        #     long_conditions.append(dataframe['er_value'] > self.erThreshold.value) # 需要 erThreshold 超参数

        # 合并所有条件 - 必须全部为 True
        if long_conditions:
            # 使用 functools.reduce 和 operator.and_ 来合并所有条件 (逻辑与)
            # 这相当于 condition1 & condition2 & condition3 ...
            final_long_condition = functools.reduce(operator.and_, long_conditions)
            dataframe.loc[final_long_condition, 'enter_long'] = 1 # 满足所有条件时，设置 enter_long=1
        else:
             dataframe['enter_long'] = 0 # 如果没有条件，则不产生信号


        # === 做空入场条件 ===
        # 1. 价格向上穿越布林带上轨 (使用 qtpylib.crossed_above)
        # 2. RSI > 做空阈值 (rsi_upper)
        # 3. HTF 趋势为下降 (htf_downtrend == 1)

        # 确保 htf_downtrend 列存在且已计算
        if 'htf_downtrend' not in dataframe.columns:
             # enter_long 已在上面处理，这里只关注 enter_short
             dataframe['enter_short'] = 0
             logger.warning(f"交易对 {metadata['pair']} 的 'htf_downtrend' 列不存在，无法生成做空信号。")
             return dataframe

        short_conditions = [] # 创建一个列表存储所有做空条件
        short_conditions.append(dataframe['htf_downtrend'] == 1) # 条件3: HTF 下降趋势
        short_conditions.append(qtpylib.crossed_above(dataframe['close'], dataframe['bb_upperband'])) # 条件1: 上穿上轨
        short_conditions.append(dataframe['rsi'] > self.rsi_upper.value) # 条件2: RSI 高于阈值
        # 可选的 ER 过滤器:
        # if self.useErFilter.value:
        #     short_conditions.append(dataframe['er_value'] > self.erThreshold.value)

        # 合并所有条件 - 必须全部为 True
        if short_conditions:
            # 使用 functools.reduce 和 operator.and_ 来合并所有条件 (逻辑与)
            final_short_condition = functools.reduce(operator.and_, short_conditions)
            dataframe.loc[final_short_condition, 'enter_short'] = 1 # 满足所有条件时，设置 enter_short=1
        else:
            dataframe['enter_short'] = 0 # 如果没有条件，则不产生信号

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        基于技术指标，填充 'exit_long' 和 'exit_short' 列来产生自定义退出信号。
        这个策略主要依赖 ROI 和止损进行退出，所以此方法默认不产生信号。

        Args:
            dataframe (DataFrame): 包含指标的 DataFrame
            metadata (dict): 附加信息 (例如 'pair')

        Returns:
            DataFrame: 包含退出信号的 DataFrame (在这个策略里默认为 0)
        """
        # Freqtrade 会自动处理在 minimal_roi 和 stoploss 中定义的退出。
        # PineScript 中的 TP/SL 逻辑已经通过这两个属性实现。
        # 因此，除非你想添加基于指标的额外退出条件 (例如 RSI 回到某个水平退出)，
        # 否则不需要在这里添加逻辑。
        # dataframe.loc[:, ['exit_long', 'exit_short']] = 0 # 如果需要明确重置信号，可以取消注释
        return dataframe

    # --- 修改后的方法签名，移除了 current_entry_signal_time ---
    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: Optional[float], max_stake: float,
                              **kwargs) -> Optional[float]:
        """
        自定义加仓逻辑：当出现新的相同方向入场信号时加仓。
        (兼容旧版 Freqtrade, 使用最近时间戳匹配, 并处理回测时序差异)。

        Args:
            trade: 当前持有的交易对象 (Trade)。
            current_time: 当前 K 线结束的时间 (UTC)。
            current_rate: 当前价格。
            current_profit: 当前未实现盈亏百分比。
            min_stake: 最小允许的单次交易额。
            max_stake: 最大允许的单次交易额。
            **kwargs: 其他可能由旧版本传入的参数（忽略）。
        Returns:
            Optional[float]: 要追加的仓位大小 (quote 货币)，或者 None (不加仓)。
        """
        # 0. 检查总开关是否在配置中启用
        if not self.config.get('position_adjustment_enable', False):
            return None

        # 1. 检查是否已达到最大加仓次数
        max_adjustments = self.config.get('max_entry_position_adjustment', 0)
        if trade.nr_of_successful_entries > max_adjustments:
            # logger.info(f"交易 {trade.id}: 已达到最大加仓次数 ({max_adjustments})，已有 {trade.nr_of_successful_entries} 次入场。")
            return None

        # 2. 获取最近完成 K 线的数据和信号
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
            if dataframe.empty:
                 logger.warning(f"交易对 {trade.pair} 的分析数据帧为空。")
                 return None

            # 确保 current_time 是 UTC 时区
            if current_time.tzinfo is None:
                current_time_utc = current_time.replace(tzinfo=timezone.utc)
            else:
                current_time_utc = current_time.astimezone(timezone.utc)

            # 查找时间戳最接近 current_time_utc 的行的索引
            try:
                 nearest_index = dataframe['date'].sub(current_time_utc).abs().idxmin()
            except ValueError as e:
                 logger.error(f"无法为 {trade.pair} 在 {current_time_utc} 找到最近的索引: {e}")
                 return None

            # 获取最近 K 线的数据 (结果是 pd.Series)
            current_candle_series = dataframe.loc[nearest_index]
            # 获取找到的 K 线的时间戳，并确保是 UTC
            found_time = pd.Timestamp(current_candle_series['date']).tz_convert(timezone.utc)

            # --- 调整后的时间验证逻辑 ---
            # 计算当前时间周期的长度
            timeframe_duration = pd.Timedelta(self.timeframe)
            # 计算期望的上一根 K 线的时间戳 (当前时间 - 一个时间周期)
            expected_previous_candle_time = current_time_utc - timeframe_duration

            # 检查找到的时间戳是否是当前时间戳 或 正好是上一根K线的时间戳
            # 允许非常小的时间误差 (例如1秒) 来处理可能的浮点精度问题
            is_current_candle = abs(found_time - current_time_utc) < pd.Timedelta(seconds=1)
            is_previous_candle = abs(found_time - expected_previous_candle_time) < pd.Timedelta(seconds=1)

            # 如果找到的时间既不是当前 K 线也不是上一根 K 线，说明可能有问题（例如数据缺失）
            if not (is_current_candle or is_previous_candle):
                logger.warning(
                    f"交易对 {trade.pair} 找到的最近 K 线时间 {found_time} "
                    f"既不是当前时间 ({current_time_utc}) 也不是上一根 K 线时间 ({expected_previous_candle_time})。"
                    f"可能存在数据缺失或问题。跳过本次加仓检查。"
                )
                return None
            # else:
                # # 记录我们实际使用的是哪根 K 线的信号
                # logger.debug(f"在 {current_time_utc} 时刻，使用 K 线 {found_time} 的信号进行决策。")

            # 从找到的 K 线数据中获取信号 (使用 .get() 更安全，防止列不存在)
            enter_long_signal = current_candle_series.get('enter_long', 0)
            enter_short_signal = current_candle_series.get('enter_short', 0)

        except Exception as e:
            logger.error(f"在 adjust_trade_position 中为 {trade.pair} 在 {current_time_utc} 访问数据帧或信号时出错: {e}", exc_info=True) # exc_info=True 记录详细错误堆栈
            return None

        # 3. 判断是否需要加仓 (基于找到的 K 线的信号)
        add_stake = False
        # 获取交易开仓时间，并确保是 UTC
        trade_open_time_utc = trade.open_date.replace(tzinfo=timezone.utc)

        # !! 关键检查：确保我们不是在交易开仓的同一根 K 线（或更早的K线）上加仓 !!
        # 比较的是产生信号的 K 线时间 (found_time) 和 交易开仓时间
        if found_time <= trade_open_time_utc:
             # logger.debug(f"信号 K 线时间 {found_time} 不晚于交易开仓时间 {trade_open_time_utc}。跳过加仓。")
             return None # 如果信号来自开仓 K 线或更早，则不加仓

        if trade.is_short:
            # 如果是空头持仓，检查是否有做空信号
            if enter_short_signal == 1:
                add_stake = True
                logger.info(f"交易 {trade.id} ({trade.pair}): 在 {current_time_utc} 检测到 K 线 {found_time} 的 'enter_short' 信号。考虑加仓。")
        else: # is long
            # 如果是多头持仓，检查是否有做多信号
            if enter_long_signal == 1:
                add_stake = True
                logger.info(f"交易 {trade.id} ({trade.pair}): 在 {current_time_utc} 检测到 K 线 {found_time} 的 'enter_long' 信号。考虑加仓。")

        # 4. 如果需要加仓，计算加仓金额
        if add_stake:
            try:
                max_trades_in_strategy = self.config.get('max_open_trades')
                # logger.info(
                #     f"交易 {trade.id}: 准备计算加仓金额。此时配置中的 max_open_trades = {max_trades_in_strategy} (类型: {type(max_trades_in_strategy)})")
                # 获取配置中定义的标准开仓金额作为加仓金额
                stake_amount_to_add = self.wallets.get_trade_stake_amount(trade.pair, max_trades_in_strategy, update=False)

                if stake_amount_to_add is None or stake_amount_to_add <= 0: # 检查小于等于 0
                     logger.warning(f"交易 {trade.id}: 计算出的加仓金额为零或负数 ({stake_amount_to_add})。跳过加仓。")
                     return None
                if min_stake is not None and stake_amount_to_add < min_stake:
                     logger.warning(f"交易 {trade.id}: 计算出的加仓金额 {stake_amount_to_add:.8f} 小于最小限制 {min_stake:.8f}。将使用最小限制值。")
                     stake_amount_to_add = min_stake
                # 可选：检查加仓后是否超过最大总投入等限制

                # logger.info(f"交易 {trade.id} ({trade.pair}): 基于 K 线 {found_time} 的信号进行加仓。增加 {stake_amount_to_add:.8f} 。")
                return stake_amount_to_add

            except Exception as e:
                logger.error(f"为交易 {trade.id} 计算加仓金额时出错: {e}", exc_info=True)
                return None

        # 如果没有匹配信号或不满足条件，则不加仓
        return None

# --- 合并 informative 数据帧的辅助函数 ---
# 确保这个函数在你的策略文件中可用，或者从其他地方正确导入
# 这个是 Freqtrade 常用的标准辅助函数
def merge_informative_pair(dataframe: DataFrame, informative: DataFrame, timeframe: str, inf_tf: str, ffill: bool = True) -> DataFrame:
    """
    辅助函数，用于将 informative 时间周期的数据合并到基础时间周期的数据帧上。
    """
    informative = informative.copy()
    # 重命名 informative 数据帧的列，添加后缀以区分，但保留 'date' 列
    informative.rename(columns=lambda x: f"{x}_{inf_tf}" if x not in ['date'] else x, inplace=True)
    # 基于日期列进行左合并 (将 informative 合并到 dataframe)
    dataframe = DataFrame.merge(dataframe, informative, on='date', how='left')
    # 向前填充 (Forward fill) NaN 值，用 informative 时间周期的最后一个已知值填充基础时间周期的空隙
    if ffill:
        dataframe.ffill(inplace=True)
    return dataframe