# --- 策略依赖 ---
import logging
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import talib.abstract as ta
from freqtrade.persistence import Trade
from freqtrade.strategy import (IStrategy, IntParameter, RealParameter, CategoricalParameter, stoploss_from_open)
from pandas import DataFrame, Series
from datetime import datetime, timedelta, timezone
from functools import reduce

logger = logging.getLogger(__name__)

# --- 策略类 ---
class SmcFvgStrategy(IStrategy):
    """
    基于 Freqtrade 的 SMC FVG 策略

    核心逻辑:
    1. 检测并存储看涨/看跌 FVG (Fair Value Gaps)。
    2. FVG 被价格触碰后失效 (Mitigation)。
    3. 基于近期 Swing High/Low 定义 Premium/Discount 区域。
    4. 只在 Discount 区寻找看涨 FVG 入场机会。
    5. 只在 Premium 区寻找看跌 FVG 入场机会。
    6. 在有效 FVG 的 50% 位置挂限价单。
    7. 止盈设在 Swing High/Low，止损设在 Swing Low/High。
    8. 时间周期: 1m
    """
    INTERFACE_VERSION = 3

    # --- 策略参数 ---
    timeframe = '1m'

    # 止损设置 - 我们将使用自定义止损，这里可以设一个较大的值或禁用
    stoploss = -0.99 # 实际止损由 custom_stoploss 控制
    use_custom_stoploss = True

    # 止盈设置 - 我们将使用自定义退出逻辑，这里 ROI 可以设一个较大的值
    minimal_roi = {"0": 10.0} # 实际止盈由 custom_exit 控制
    use_exit_signal = True
    exit_profit_only = False # 允许止损退出
    ignore_roi_if_entry_signal = False

    # 订单类型设置
    order_types = {
        'entry': 'limit', # 在 FVG 50% 挂限价单
        'exit': 'limit', # 止盈使用市场价退出（也可以改 limit）
        'stoploss': 'limit', # 止损使用市场价退出（更可靠）
        'stoploss_on_exchange': False # Freqtrade 处理止损
    }

    # 订单有效时间
    order_time_in_force = {
        'entry': 'gtc', # Good 'til cancelled - 允许挂单等待成交
        'exit': 'gtc'
    }

    # 策略参数 - 可用于优化
    # swing_lookback = IntParameter(50, 200, default=100, space="buy", optimize=True) # 计算前高/前低的 K线数量
    swing_lookback = 100 # 固定值，如果不需要优化
    # fvg_mitigation_lookback = IntParameter(100, 500, default=200, space="buy", optimize=True) # FVG 存储和缓解检查的回溯 K线数量
    fvg_mitigation_lookback = 200 # 固定值

    # --- 内部状态变量 ---
    # 使用字典来存储每个交易对的活跃 FVG 列表
    # 结构: self.active_fvgs[pair] = {'bullish': [fvg_dict, ...], 'bearish': [fvg_dict, ...]}
    # fvg_dict 结构: {'index': candle_index, 'top': price, 'bottom': price, 'mid_price': price, 'mitigated': False, 'swing_high': price, 'swing_low': price}
    active_fvgs: Dict[str, Dict[str, List[Dict]]] = {}

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        # 在策略初始化时清空存储的 FVG，确保每次启动是干净的
        self.active_fvgs = {}
        logger.info("SMC FVG 策略已初始化。")
        logger.info(f"Swing Point Lookback: {self.swing_lookback}")
        logger.info(f"FVG Mitigation Lookback: {self.fvg_mitigation_lookback}")

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        计算指标、检测 FVG、管理 FVG 状态、计算入场条件
        """
        pair = metadata['pair']

        # --- 0. 初始化/获取 FVG 存储 ---
        if pair not in self.active_fvgs:
            self.active_fvgs[pair] = {'bullish': [], 'bearish': []}
            logger.info(f"为交易对 {pair} 初始化 FVG 存储区。")

        pair_fvgs = self.active_fvgs[pair]
        current_time = dataframe.iloc[-1]['date'].replace(tzinfo=timezone.utc)
        min_fvg_time = current_time - timedelta(minutes=self.fvg_mitigation_lookback * int(self.timeframe[:-1])) # FVG 最早保留时间

        # --- 1. FVG 缓解检查与清理 ---
        last_candle = dataframe.iloc[-1]
        last_low = last_candle['low']
        last_high = last_candle['high']

        for fvg_type in ['bullish', 'bearish']:
            active_list = pair_fvgs[fvg_type]
            # 迭代副本以安全地修改原始列表
            for fvg in active_list[:]:
                fvg_time = fvg['time'] # 获取 FVG 的时间戳
                # 检查是否已缓解
                if not fvg['mitigated']:
                    # 如果当前 K 线触及 FVG 区域
                    if last_low <= fvg['top'] and last_high >= fvg['bottom']:
                        fvg['mitigated'] = True
                        logger.debug(f"[{pair}] {fvg_type.capitalize()} FVG @ {fvg_time} (Index: {fvg['index']}) Price: {fvg['bottom']:.5f}-{fvg['top']:.5f} 已被缓解 (Mitigated)。")

                # 清理过时或已缓解的 FVG
                # if fvg['mitigated'] or fvg['index'] < dataframe.index[0] or fvg_time < min_fvg_time: # 根据时间或索引清理
                if fvg['mitigated'] or fvg_time < min_fvg_time: # 根据时间清理更可靠
                     try:
                         active_list.remove(fvg)
                         # logger.debug(f"[{pair}] 清理 {'过时' if fvg_time < min_fvg_time else '已缓解'} 的 {fvg_type} FVG @ {fvg_time} (Index: {fvg['index']})")
                     except ValueError:
                         pass # 可能已被移除

        # --- 2. 检测新的 FVG (基于倒数第二根 K 线完成时) ---
        if len(dataframe) >= 3:
            # FVG 由 K线 i-2, i-1, i 形成，缺口在 K线 i-2 和 i 之间
            candle_i = dataframe.iloc[-1]  # 当前最新（可能未完成）的 K 线
            candle_i_minus_1 = dataframe.iloc[-2] # 已完成的 K 线
            candle_i_minus_2 = dataframe.iloc[-3] # 更早的已完成 K 线

            fvg_index = dataframe.index[-2] # FVG 形成在 index i-1 的 K 线处
            fvg_time = candle_i_minus_1['date'].replace(tzinfo=timezone.utc) # FVG 形成的时间戳

            # 检查重复 FVG (基于索引和价格)
            def fvg_exists(fvg_list, index, top, bottom):
                return any(f['index'] == index and f['top'] == top and f['bottom'] == bottom for f in fvg_list)

            # 看涨 FVG: K线 i-2 的 high < K线 i 的 low
            bull_fvg_bottom = candle_i_minus_2['high']
            bull_fvg_top = candle_i['low']
            if bull_fvg_bottom < bull_fvg_top:
                if not fvg_exists(pair_fvgs['bullish'], fvg_index, bull_fvg_top, bull_fvg_bottom):
                    new_bull_fvg = {
                        'index': fvg_index,
                        'time': fvg_time,
                        'top': bull_fvg_top,
                        'bottom': bull_fvg_bottom,
                        'mid_price': (bull_fvg_top + bull_fvg_bottom) / 2,
                        'mitigated': False,
                        # Swing High/Low 在找到 FVG 时先置空, 后面统一计算
                        'swing_high': None,
                        'swing_low': None,
                    }
                    pair_fvgs['bullish'].append(new_bull_fvg)
                    logger.debug(f"[{pair}] 检测到新的 Bullish FVG @ {fvg_time} (Index: {fvg_index}) Price: {new_bull_fvg['bottom']:.5f}-{new_bull_fvg['top']:.5f}")
                    # ---> 添加日志 <---
                    logger.info(
                        f"[{pair}] *** DETECTED Bullish FVG *** @ {fvg_time} (Index: {fvg_index}) Price: {new_bull_fvg['bottom']:.5f}-{new_bull_fvg['top']:.5f}")

            # 看跌 FVG: K线 i-2 的 low > K线 i 的 high
            bear_fvg_top = candle_i_minus_2['low']
            bear_fvg_bottom = candle_i['high']
            if bear_fvg_top > bear_fvg_bottom:
                 if not fvg_exists(pair_fvgs['bearish'], fvg_index, bear_fvg_top, bear_fvg_bottom):
                    new_bear_fvg = {
                        'index': fvg_index,
                        'time': fvg_time,
                        'top': bear_fvg_top,
                        'bottom': bear_fvg_bottom,
                        'mid_price': (bear_fvg_top + bear_fvg_bottom) / 2,
                        'mitigated': False,
                        # Swing High/Low 在找到 FVG 时先置空, 后面统一计算
                        'swing_high': None,
                        'swing_low': None,
                    }
                    pair_fvgs['bearish'].append(new_bear_fvg)
                    logger.debug(f"[{pair}] 检测到新的 Bearish FVG @ {fvg_time} (Index: {fvg_index}) Price: {new_bear_fvg['bottom']:.5f}-{new_bear_fvg['top']:.5f}")
                    # ---> 添加日志 <---
                    logger.info(
                        f"[{pair}] *** DETECTED Bearish FVG *** @ {fvg_time} (Index: {fvg_index}) Price: {new_bear_fvg['bottom']:.5f}-{new_bear_fvg['top']:.5f}")

        # --- 3. 计算 Swing High / Swing Low / Premium / Discount ---
        # 使用过去 N 根 K 线来确定结构点
        lookback_data = dataframe.tail(self.swing_lookback)
        if not lookback_data.empty:
            swing_high = lookback_data['high'].max()
            swing_low = lookback_data['low'].min()
            mid_point = (swing_high + swing_low) / 2

            # 将计算结果存入 dataframe (主要用于调试或可视化)
            dataframe.loc[dataframe.index[-1], 'swing_high'] = swing_high
            dataframe.loc[dataframe.index[-1], 'swing_low'] = swing_low
            dataframe.loc[dataframe.index[-1], 'mid_point'] = mid_point

            # 更新活跃 FVG 的 swing high/low (用于后续入场和 SL/TP)
            # 注意：这里用的是 *当前* 的 swing high/low 来评估 *所有活跃* 的 FVG
            # 这是一种简化处理，更复杂的逻辑可能是用 FVG *形成时* 的 swing high/low
            for fvg_type in ['bullish', 'bearish']:
                 for fvg in pair_fvgs[fvg_type]:
                     if not fvg['mitigated']: # 只更新未缓解的
                        fvg['swing_high'] = swing_high
                        fvg['swing_low'] = swing_low

        else:
            # 数据不足时设置默认值
            swing_high = dataframe['high'].iloc[-1] if not dataframe.empty else np.nan
            swing_low = dataframe['low'].iloc[-1] if not dataframe.empty else np.nan
            mid_point = (swing_high + swing_low) / 2 if pd.notna(swing_high) and pd.notna(swing_low) else np.nan
            dataframe.loc[dataframe.index[-1], 'swing_high'] = swing_high
            dataframe.loc[dataframe.index[-1], 'swing_low'] = swing_low
            dataframe.loc[dataframe.index[-1], 'mid_point'] = mid_point

        # --- 4. 寻找有效的入场 FVG ---
        # 清除旧的入场信号相关列
        dataframe['enter_long_signal_price'] = np.nan
        dataframe['enter_short_signal_price'] = np.nan
        dataframe['sl_price'] = np.nan
        dataframe['tp_price'] = np.nan

        valid_bull_fvg_entry = None
        valid_bear_fvg_entry = None

        # 寻找多头入场 (看涨 FVG 在 折扣区 Discount)
        # debug暂时屏蔽折扣区检查potential_bull_fvgs = [fvg for fvg in pair_fvgs['bullish'] if not fvg['mitigated'] and fvg['top'] < mid_point and fvg['swing_low'] is not None]
        potential_bull_fvgs = [fvg for fvg in pair_fvgs['bullish'] if not fvg['mitigated']]  # 暂时移除 P/D 检查
        if potential_bull_fvgs:
            # 选择一个 FVG，例如最新的一个，或者最接近当前价格的一个？
            # 这里简单选择最新的一个有效 FVG
            valid_bull_fvg_entry = max(potential_bull_fvgs, key=lambda f: f['index']) # 选择最新的
            logger.info(f"[{pair}] ---> (DEBUGGING) Found ANY Bullish FVG! <--- Index: {fvg['index']}")  # 添加调试日志
            # 或者选择最接近当前价格下方的 FVG midpoint？
            # current_close = dataframe['close'].iloc[-1]
            # potential_bull_fvgs_below = [f for f in potential_bull_fvgs if f['mid_price'] < current_close]
            # if potential_bull_fvgs_below:
            #    valid_bull_fvg_entry = max(potential_bull_fvgs_below, key=lambda f: f['mid_price']) # 选择下方最近的

        # 寻找空头入场 (看跌 FVG 在 溢价区 Premium)
        # debug暂时屏蔽溢价区检查potential_bear_fvgs = [fvg for fvg in pair_fvgs['bearish'] if not fvg['mitigated'] and fvg['bottom'] > mid_point and fvg['swing_high'] is not None]
        potential_bear_fvgs = [fvg for fvg in pair_fvgs['bearish'] if not fvg['mitigated']]  # 暂时移除 P/D 检查
        if potential_bear_fvgs:
            # 选择一个 FVG，例如最新的一个
            valid_bear_fvg_entry = max(potential_bear_fvgs, key=lambda f: f['index']) # 选择最新的
            logger.info(f"[{pair}] ---> (DEBUGGING) Found ANY Bearish FVG! <--- Index: {fvg['index']}")  # 添加调试日志
            # 或者选择最接近当前价格上方的 FVG midpoint？
            # current_close = dataframe['close'].iloc[-1]
            # potential_bear_fvgs_above = [f for f in potential_bear_fvgs if f['mid_price'] > current_close]
            # if potential_bear_fvgs_above:
            #     valid_bear_fvg_entry = min(potential_bear_fvgs_above, key=lambda f: f['mid_price']) # 选择上方最近的


        # --- 5. 在 DataFrame 中标记入场信号和 SL/TP ---
        # 注意：这里是在 *当前* K 线标记潜在入场点。实际挂单由 populate_entry_trend 和 custom_entry_price 处理
        current_idx = dataframe.index[-1]
        if valid_bull_fvg_entry:
            dataframe.loc[current_idx, 'enter_long_signal_price'] = valid_bull_fvg_entry['mid_price']
            dataframe.loc[current_idx, 'sl_price'] = valid_bull_fvg_entry['swing_low'] # SL 是对应的 Swing Low
            dataframe.loc[current_idx, 'tp_price'] = valid_bull_fvg_entry['swing_high'] # TP 是对应的 Swing High
            logger.debug(f"[{pair}] 标记潜在 Long 入场 @ {valid_bull_fvg_entry['mid_price']:.5f} (FVG Index: {valid_bull_fvg_entry['index']}). SL: {valid_bull_fvg_entry['swing_low']:.5f}, TP: {valid_bull_fvg_entry['swing_high']:.5f}")

        if valid_bear_fvg_entry:
            dataframe.loc[current_idx, 'enter_short_signal_price'] = valid_bear_fvg_entry['mid_price']
            dataframe.loc[current_idx, 'sl_price'] = valid_bear_fvg_entry['swing_high'] # SL 是对应的 Swing High
            dataframe.loc[current_idx, 'tp_price'] = valid_bear_fvg_entry['swing_low'] # TP 是对应的 Swing Low
            logger.debug(f"[{pair}] 标记潜在 Short 入场 @ {valid_bear_fvg_entry['mid_price']:.5f} (FVG Index: {valid_bear_fvg_entry['index']}). SL: {valid_bear_fvg_entry['swing_high']:.5f}, TP: {valid_bear_fvg_entry['swing_low']:.5f}")

        # logger.debug(f"[{pair}] Active Bullish FVGs: {len(pair_fvgs['bullish'])}")
        # logger.debug(f"[{pair}] Active Bearish FVGs: {len(pair_fvgs['bearish'])}")

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        根据 populate_indicators 计算的信号决定是否入场
        """
        # 当 dataframe 中标记了有效的入场信号价格时，生成入场信号
        dataframe.loc[
            (pd.notna(dataframe['enter_long_signal_price'])),
            'enter_long'] = 1

        dataframe.loc[
            (pd.notna(dataframe['enter_short_signal_price'])),
            'enter_short'] = 1

        return dataframe

    def custom_entry_price(self, pair: str, current_time: datetime, proposed_rate: float, entry_tag: Optional[str], side: str, **kwargs) -> float:
        """
        自定义入场价格，返回 FVG 50% 的价格作为限价单价格
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        if side == 'long' and pd.notna(last_candle['enter_long_signal_price']):
            entry_price = last_candle['enter_long_signal_price']
            logger.info(f"[{pair}] 准备在 {entry_price:.5f} 挂 Long 限价单 (基于 FVG 50%)")
            return entry_price
        elif side == 'short' and pd.notna(last_candle['enter_short_signal_price']):
            entry_price = last_candle['enter_short_signal_price']
            logger.info(f"[{pair}] 准备在 {entry_price:.5f} 挂 Short 限价单 (基于 FVG 50%)")
            return entry_price
        else:
            # 如果没有找到明确的价格（理论上不应发生，因为 populate_entry_trend 会检查）
            # 返回 proposed_rate 或一个无效价格触发取消
            logger.warning(f"[{pair}] custom_entry_price: 未找到 {side} 的 FVG 入场价格信号，使用 proposed_rate: {proposed_rate}")
            # return proposed_rate # 或者返回 None 取消订单? Freqtrade 不支持返回 None
            # 返回一个远离当前价的价格可能导致订单无法成交，但避免了意外入场
            if side == 'long':
                return proposed_rate * 0.9 # 挂一个低价
            else:
                return proposed_rate * 1.1 # 挂一个高价

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime, current_rate: float, current_profit: float, **kwargs) -> Optional[float]:
        """
        自定义止损价格
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        # 获取与交易开仓时间匹配的 K 线数据
        entry_candle = dataframe[(dataframe['date'] < trade.open_date_utc)] # .iloc[-1]
        if entry_candle.empty:
             # 尝试用索引（如果时间戳不完全匹配） - 这比较脆弱
             try:
                 # Freqtrade < 2023.7: trade.open_date -> trade.open_date_utc
                 entry_candle = dataframe.loc[dataframe['date'] == trade.open_date_utc]
                 if entry_candle.empty:
                     # 如果精确时间匹配不到，尝试找最近的一个
                    entry_candle = dataframe[dataframe['date'] <= trade.open_date_utc].iloc[-1:] # 取最后一行
             except Exception:
                 logger.warning(f"[{pair}] custom_stoploss: 无法找到交易 {trade.id} 的入场 K 线数据。")
                 return None # 返回 None 使用默认或无止损

        if not entry_candle.empty:
            sl_price_series = entry_candle['sl_price'].iloc[-1] # 取对应行的 sl_price
            if pd.notna(sl_price_series):
                stoploss_price = float(sl_price_series)
                # 确保止损价对交易方向有效
                if trade.is_short:
                     # 空单止损价应高于开仓价
                    if stoploss_price > trade.open_rate:
                        logger.debug(f"[{pair}] Trade {trade.id} (Short) 自定义止损价: {stoploss_price:.5f}")
                        return stoploss_price
                    else:
                        logger.warning(f"[{pair}] Trade {trade.id} (Short) 计算的止损价 {stoploss_price:.5f} <= 开仓价 {trade.open_rate:.5f}，止损无效。")
                        # 可以返回一个基于开仓价的固定比例止损作为后备
                        # return trade.open_rate * (1 + 0.01) # 例如 1% 止损
                else:
                    # 多单止损价应低于开仓价
                    if stoploss_price < trade.open_rate:
                        logger.debug(f"[{pair}] Trade {trade.id} (Long) 自定义止损价: {stoploss_price:.5f}")
                        return stoploss_price
                    else:
                        logger.warning(f"[{pair}] Trade {trade.id} (Long) 计算的止损价 {stoploss_price:.5f} >= 开仓价 {trade.open_rate:.5f}，止损无效。")
                        # return trade.open_rate * (1 - 0.01) # 例如 1% 止损

        logger.warning(f"[{pair}] custom_stoploss: 未能为交易 {trade.id} 获取有效的自定义止损价。")
        # 如果无法获取 SL 价格，返回 None，依赖于配置中的 stoploss 参数 (但可能不是我们想要的)
        # 或者返回一个基于开仓价的固定止损
        # stop_pct = 0.01 # 1% 应急止损
        # if trade.is_short:
        #     return trade.open_rate * (1 + stop_pct)
        # else:
        #     return trade.open_rate * (1 - stop_pct)
        return None # 更安全的选择是返回 None，依赖框架处理或不设止损

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        必须实现的退出信号填充方法。
        在这个策略中，主要的退出逻辑（止盈/止损）由 custom_exit 和 custom_stoploss 处理。
        因此，此方法仅用于满足框架要求，不生成基于指标的退出信号。
        我们将 exit_long 和 exit_short 设置为 0。
        """
        # Freqtrade 要求 exit_long 和 exit_short 列存在（当 use_exit_signal=True 时）
        # 由于我们使用 custom_exit 和 custom_stoploss，这里不产生信号
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0

        return dataframe

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float, current_profit: float, **kwargs) -> Optional[Union[str, bool]]:
        """
        自定义退出逻辑，用于处理止盈
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        # 获取与交易开仓时间匹配的 K 线数据 (与 custom_stoploss 类似)
        entry_candle = dataframe[(dataframe['date'] < trade.open_date_utc)]
        if entry_candle.empty:
             try:
                 entry_candle = dataframe.loc[dataframe['date'] == trade.open_date_utc]
                 if entry_candle.empty:
                    entry_candle = dataframe[dataframe['date'] <= trade.open_date_utc].iloc[-1:]
             except Exception:
                 logger.warning(f"[{pair}] custom_exit: 无法找到交易 {trade.id} 的入场 K 线数据。")
                 return None

        if not entry_candle.empty:
            tp_price_series = entry_candle['tp_price'].iloc[-1]
            if pd.notna(tp_price_series):
                take_profit_price = float(tp_price_series)
                logger.debug(f"[{pair}] Trade {trade.id} 目标止盈价: {take_profit_price:.5f}")

                # 检查是否达到止盈目标
                if trade.is_short:
                    # 空单止盈: 当前价格 <= 目标价
                    if current_rate <= take_profit_price:
                        logger.info(f"[{pair}] Trade {trade.id} (Short) 触发自定义止盈 @ {current_rate:.5f} (目标: {take_profit_price:.5f})")
                        return 'fvg_take_profit' # 返回自定义退出原因
                else:
                    # 多单止盈: 当前价格 >= 目标价
                    if current_rate >= take_profit_price:
                        logger.info(f"[{pair}] Trade {trade.id} (Long) 触发自定义止盈 @ {current_rate:.5f} (目标: {take_profit_price:.5f})")
                        return 'fvg_take_profit'

        # 如果没有触发自定义止盈，返回 None，让其他退出机制（如 ROI, 止损）处理
        return None

    def custom_stake_amount(self, pair: str, current_time: 'datetime', current_rate: float,
                            proposed_stake: float, min_stake: float, max_stake: float,
                            entry_tag: str, **kwargs) -> float:
        # 获取可用余额（仅未被占用的资金）
        free_balance = self.wallets.get_free('USDT')

        # 动态计算（例如：可用资金的50%）
        stake_amount = free_balance * 0.1
        return stake_amount

    # --- (可选) 辅助函数 ---
    # 你可以在这里添加其他辅助函数，例如更复杂的 Swing High/Low 检测逻辑

# --- (可选) 用于绘图的 Plot 配置 ---
# 如果你想在 FreqUI 或绘图脚本中可视化 FVG 和 Swing Points，可以取消注释并实现 plot_config
# class SmcFvgStrategyV1(SmcFvgStrategy):
#     plot_config = {
#         'main_plot': {
#             'swing_high': {'color': 'red', 'type': 'line', 'style': 'dotted'},
#             'swing_low': {'color': 'green', 'type': 'line', 'style': 'dotted'},
#             'mid_point': {'color': 'blue', 'type': 'line', 'style': 'dashed'},
#             # 注意：直接绘制 FVG 区域比较复杂，通常需要自定义绘图脚本
#             # 'bull_fvg_top': {'color': 'lightgreen'},
#             # 'bull_fvg_bottom': {'color': 'lightgreen'},
#             # 'bear_fvg_top': {'color': 'lightcoral'},
#             # 'bear_fvg_bottom': {'color': 'lightcoral'},
#         },
#         'subplots': {
#             "signals": {
#                 'enter_long_signal_price': {'color': 'cyan', 'type': 'scatter', 'marker': '^'},
#                 'enter_short_signal_price': {'color': 'magenta', 'type': 'scatter', 'marker': 'v'},
#                 'sl_price': {'color': 'orange'},
#                 'tp_price': {'color': 'purple'}
#            }
#         }
#     }