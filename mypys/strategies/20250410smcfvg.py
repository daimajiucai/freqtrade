# --- 请勿删除这些库 ---
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter, CategoricalParameter
from freqtrade.persistence import Trade

# --- 技术分析包 ---
# import talib.abstract as ta # 未直接使用，但可用
# import freqtrade.vendor.qtpylib.indicators as qtpylib # 未直接使用，但可用

# --- 优化参数 (可选) ---
# 如果您想稍后进行优化，可以取消注释并定义这些参数
# class FVGCrossStrategy(IStrategy):
#     fvg_lookback_period = IntParameter(10, 50, default=20, space="buy", optimize=True) # FVG 回看周期
#     take_profit_pct = DecimalParameter(0.003, 0.02, default=0.005, decimals=3, space="sell", optimize=True) # 止盈百分比
#     # stoploss_from_mid = CategoricalParameter([True, False], default=True, space="buy", optimize=True) # 是否从中点设置止损（示例）


class FVGCrossStrategy0410(IStrategy):
    """
    基于 SMC FVG 交叉的 Freqtrade 策略
    入场:
        - 做多: 收盘价上穿最近的看跌 FVG 顶部。
        - 做空: 收盘价下穿最近的看涨 FVG 底部。
    止损:
        - 做多: 触发入场的看跌 FVG 的中点。
        - 做空: 触发入场的看涨 FVG 的中点。
    止盈:
        - 固定 0.5%
    """

    # --- 策略配置 ---
    INTERFACE_VERSION = 3 # Freqtrade 2023.7+ 版本必需

    # ROI 表: 设置很高以依赖自定义退出逻辑
    minimal_roi = {
        "0": 100.0 # 实际上禁用 ROI，使用 custom_exit 进行止盈
    }

    # 止损: 设置一个较大的默认值，会被 custom_stoploss 覆盖
    stoploss = -0.99 # 必须存在，但 custom_stoploss 优先

    # 追踪止损 (禁用)
    trailing_stop = False
    # trailing_stop_positive = 0.001
    # trailing_stop_positive_offset = 0.005
    # trailing_only_offset_is_reached = True

    timeframe = '1m' # 定义您想要的时间周期

    # 仅在新 K 线产生时运行 "populate_indicators()"
    process_only_new_candles = True

    # 这些值可以在配置文件中覆盖
    use_exit_signal = True    # 使用退出信号
    exit_profit_only = False  # 不仅仅在盈利时退出
    ignore_roi_if_entry_signal = False # 如果有入场信号，不忽略 ROI

    # 自定义退出定义
    use_custom_exit = True # 启用 custom_exit 方法
    use_custom_stoploss = True # 启用 custom_stoploss 方法

    # 策略产生有效信号前需要的 K 线数量
    # FVG 需要至少 3 根 K 线 (当前, i-1, i-2) + 回看周期
    startup_candle_count: int = 30 # 根据 fvg_lookback_period 调整

    # 可选的订单有效时间 (Time in Force)
    order_time_in_force = {
        'entry': 'gtc', # Good 'til Canceled - 一直有效直到取消
        'exit': 'gtc'
    }

    # --- 策略参数 ---
    # 考虑稍后使用 IntParameter 等使其可优化
    fvg_lookback_period = 50 # 考虑多少根过去的 K 线来寻找存在的 FVG
    take_profit_pct = 0.005  # 0.5% 止盈

    # --- 自定义存储 ---
    # 使用类变量在多次调用之间存储 FVG 信息
    # 如果在同一个机器人实例中运行多个交易对，需要小心管理
    # 使用字典将交易对映射到其活跃 FVG 列表更安全
    # 例如: {'BTC/USDT': [{'type': 'bearish', ...}, ...], 'ETH/USDT': [...]}
    active_fvgs_storage = {}  # 存储活跃 FVG: {'交易对': [fvg_字典1, fvg_字典2, ...]}
    trade_sl_storage = {}     # 存储活跃交易的止损价格: {'交易ID': sl_price}

    def __init__(self, config: dict) -> None:
        """初始化策略，确保存储字典存在"""
        super().__init__(config)
        # 为每个交易对初始化存储（如果不存在的话），处理重启/重载情况
        if not hasattr(self, 'active_fvgs_storage'):
            self.active_fvgs_storage = {}
        if not hasattr(self, 'trade_sl_storage'):
            self.trade_sl_storage = {}


    # --- 指标填充 ---
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        向给定的 DataFrame 添加各种技术分析指标和识别 FVG
        """
        pair = metadata['pair'] # 获取当前处理的交易对

        # --- 如果当前交易对的存储不存在，则初始化 ---
        if pair not in self.active_fvgs_storage:
            self.active_fvgs_storage[pair] = []

        # --- 管理活跃 FVG ---
        current_index = dataframe.index[-1] # 获取当前 DataFrame 的最后一个索引
        active_fvgs = self.active_fvgs_storage[pair] # 获取当前交易对的活跃 FVG 列表

        # 1. 移除旧的 FVG
        active_fvgs = [
            fvg for fvg in active_fvgs
            if current_index - fvg['index'] <= self.fvg_lookback_period # 只保留在回看周期内的 FVG
        ]

        # 2. 从 *倒数第二根* K 线识别 *新的* FVG
        # 我们检查 K 线 i-1，看它是否利用 i-1, i-2, i-3 的数据完成了 FVG
        if len(dataframe) > 3: # 确保有足够的 K 线进行计算
            last_idx = -2 # 最后一根完整 K 线的索引
            i_high = dataframe['high'].iloc[last_idx]       # K 线 i-1 的最高价
            i_low = dataframe['low'].iloc[last_idx]         # K 线 i-1 的最低价
            i_minus_2_high = dataframe['high'].shift(2).iloc[last_idx] # K 线 i-3 的最高价
            i_minus_2_low = dataframe['low'].shift(2).iloc[last_idx]   # K 线 i-3 的最低价
            fvg_candle_index = dataframe.index[last_idx] # 形成 FVG 的 K 线的索引 (i-1)

            # 检查新的看跌 FVG (High(i-1) < Low(i-3))
            if i_high < i_minus_2_low:
                fvg = {
                    'type': 'bearish',               # 类型：看跌
                    'top': i_minus_2_low,            # 顶部：K 线 i-3 的最低价
                    'bottom': i_high,                # 底部：K 线 i-1 的最高价
                    'mid': (i_minus_2_low + i_high) / 2, # 中点
                    'index': fvg_candle_index,       # 形成时的 K 线索引
                    'used': False                    # 标记：是否已用于触发入场
                }
                # 避免添加重复的 FVG（如果同一个 FVG 在多根 K 线上持续满足条件）
                if not any(f['type'] == 'bearish' and f['index'] == fvg_candle_index for f in active_fvgs):
                    active_fvgs.append(fvg)
                    # print(f"{pair} - {dataframe['date'].iloc[last_idx]}: 发现新的看跌 FVG: 顶部={fvg['top']:.5f}, 底部={fvg['bottom']:.5f}") # 调试输出

            # 检查新的看涨 FVG (Low(i-1) > High(i-3))
            if i_low > i_minus_2_high:
                fvg = {
                    'type': 'bullish',               # 类型：看涨
                    'top': i_low,                    # 顶部：K 线 i-1 的最低价
                    'bottom': i_minus_2_high,        # 底部：K 线 i-3 的最高价
                    'mid': (i_low + i_minus_2_high) / 2, # 中点
                    'index': fvg_candle_index,       # 形成时的 K 线索引
                    'used': False                    # 标记：是否已用于触发入场
                }
                 # 避免添加重复的 FVG
                if not any(f['type'] == 'bullish' and f['index'] == fvg_candle_index for f in active_fvgs):
                    active_fvgs.append(fvg)
                    # print(f"{pair} - {dataframe['date'].iloc[last_idx]}: 发现新的看涨 FVG: 顶部={fvg['top']:.5f}, 底部={fvg['bottom']:.5f}") # 调试输出

        # 更新该交易对的存储
        self.active_fvgs_storage[pair] = active_fvgs

        # --- 如果绘图工具需要，添加用于入场/出场信号的虚拟列 ---
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0

        return dataframe

    # --- 入场趋势填充 ---
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        基于技术指标，填充给定 DataFrame 的入场信号
        """
        pair = metadata['pair'] # 获取当前交易对
        active_fvgs = self.active_fvgs_storage.get(pair, []) # 获取活跃 FVG 列表，如果不存在则返回空列表
        current_close = dataframe['close'].iloc[-1] # 获取最新收盘价
        current_date = dataframe['date'].iloc[-1] # 获取当前时间（用于日志记录）

        # --- 初始化信号 ---
        enter_long_signal = False
        enter_short_signal = False
        sl_price_for_trade = None # 用于存储本次交易的止损价格

        # --- 检查条件 ---
        # 按索引降序排序 FVG，优先检查最新的 FVG
        sorted_fvgs = sorted(active_fvgs, key=lambda x: x['index'], reverse=True)

        for fvg in sorted_fvgs:
            if fvg['used']: # 如果此 FVG 已被用于入场，则跳过
                continue

            # 做多入场: 收盘价上穿 *看跌* FVG 的顶部
            if fvg['type'] == 'bearish' and current_close > fvg['top']:
                enter_long_signal = True
                sl_price_for_trade = fvg['mid'] # 止损设置在看跌 FVG 的中点
                fvg['used'] = True # 标记为已使用
                print(f"{pair} - {current_date}: 触发做多入场。收盘价 {current_close:.5f} 上穿看跌 FVG 顶部 {fvg['top']:.5f} (形成于索引 {fvg['index']})。止损设为 {sl_price_for_trade:.5f}.")
                break # 每根 K 线只触发一次信号

            # 做空入场: 收盘价下穿 *看涨* FVG 的底部
            elif fvg['type'] == 'bullish' and current_close < fvg['bottom']:
                enter_short_signal = True
                sl_price_for_trade = fvg['mid'] # 止损设置在看涨 FVG 的中点
                fvg['used'] = True # 标记为已使用
                print(f"{pair} - {current_date}: 触发做空入场。收盘价 {current_close:.5f} 下穿看涨 FVG 底部 {fvg['bottom']:.5f} (形成于索引 {fvg['index']})。止损设为 {sl_price_for_trade:.5f}.")
                break # 每根 K 线只触发一次信号

        # --- 设置信号并存储止损 ---
        # 仅在最后一根 K 线上设置信号
        if enter_long_signal:
            dataframe.loc[dataframe.index[-1], 'enter_long'] = 1
            # 临时存储止损价；custom_stoploss 将使用交易信息来检索它
            # 在知道 trade_id 之前，我们使用临时的 '交易对_方向' 键
            self.trade_sl_storage[f"{pair}_long"] = sl_price_for_trade

        if enter_short_signal:
            dataframe.loc[dataframe.index[-1], 'enter_short'] = 1
            self.trade_sl_storage[f"{pair}_short"] = sl_price_for_trade

        # 从主存储列表中清理 'used' 的 FVG（可选，但可能是好习惯）
        # self.active_fvgs_storage[pair] = [fvg for fvg in active_fvgs if not fvg['used']]
        # 我们保留它们但标记为已使用，也许能简化调试。它们最终会被时间淘汰。

        return dataframe

    # --- 出场趋势填充 ---
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        基于技术指标，填充给定 DataFrame 的出场信号
        """
        # 这里没有定义基于指标的退出，使用 custom_exit 和 custom_stoploss
        dataframe.loc[:, 'exit_long'] = 0
        dataframe.loc[:, 'exit_short'] = 0
        return dataframe

    # --- 自定义止损 ---
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: 'datetime',
                        current_rate: float, current_profit: float, **kwargs) -> float:
        """
        自定义止损逻辑，返回止损的绝对价格。
        返回 -1 或正值。正值意味着绝对止损价格。
        '-1' 表示禁用自定义止损，并回退到配置文件中的 'stoploss' 值。
        """
        sl_price = None # 初始化止损价格

        # --- 将交易入场与存储的止损价格关联 ---
        # 尝试查找在入场信号生成期间存储的止损价
        sl_key_base = f"{pair}_{trade.direction}" # 例如: 'BTC/USDT_long'

        # 当交易刚开仓时，freqtrade 可能在下一次 populate_entry 之前调用此方法，
        # 所以键可能仍然是临时的那个。一根 K 线后，我们可以使用 trade_id。
        if trade.id in self.trade_sl_storage:
             sl_price = self.trade_sl_storage[trade.id] # 优先使用交易 ID 查找
        elif sl_key_base in self.trade_sl_storage:
            sl_price = self.trade_sl_storage[sl_key_base] # 尝试使用临时键查找
            # 将其与 trade_id 关联存储以备将来调用，并删除临时键
            self.trade_sl_storage[trade.id] = sl_price
            del self.trade_sl_storage[sl_key_base]

        # --- 返回止损价格 ---
        if sl_price is not None:
            # print(f"交易 {trade.id}: 自定义止损激活于 {sl_price:.5f}") # 调试信息
            # 返回正值，表示绝对止损价格
            return sl_price
        else:
            # 后备方案: 如果没有找到止损价格（通常不应发生）
            # 返回一个能有效禁用止损或使用非常宽的默认值的值
            # 返回 -1 会让 freqtrade 使用配置中的 `stoploss` 百分比
            # 为了安全起见，我们返回一个非常宽的绝对止损，主要还是依赖于找到 FVG 中点。
            print(f"警告: 未找到交易 {trade.id} 的止损价格。使用非常宽的后备止损。")
            if trade.direction == 'long':
                return trade.open_rate * 0.1 # 极其宽的止损 (示例)
            else: # short
                 return trade.open_rate * 2.0 # 极其宽的止损 (示例)

        # 如果 sl_price 被找到或使用了后备方案，则不应到达此点
        # return -1 # 如果需要，回退到配置的 stoploss 百分比


    # --- 自定义退出 ---
    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        """
        自定义退出信号逻辑。目前实现固定止盈。
        可以返回一个字符串，作为退出原因记录下来。
        """
        # --- 止盈 ---
        if current_profit >= self.take_profit_pct:
             # 清理此交易的止损存储
            if trade.id in self.trade_sl_storage:
                del self.trade_sl_storage[trade.id]
            # 同时检查临时键，以防止盈发生得非常快
            temp_key = f"{pair}_{trade.direction}"
            if temp_key in self.trade_sl_storage:
                 del self.trade_sl_storage[temp_key]

            return f'take_profit_target_{self.take_profit_pct*100:.1f}%' # 自定义退出原因

        # 如果没有触发自定义退出信号，返回 None
        return None


    # --- 可选: 确认交易入场/出场 ---
    # 对于调试或添加二次检查很有用

    # def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
    #                         time_in_force: str, current_time: datetime, **kwargs) -> bool:
    #     """
    #     在实际下单前进行最后确认。
    #     """
    #     # trade = Trade.get_trades([Trade.pair == pair, Trade.is_open.is_(True),]).first()
    #     # if trade:
    #          # 可以在这里添加逻辑，例如检查成交量、滑点等
    #     #     pass
    #     return True # 批准入场

    # def confirm_trade_exit(self, pair: str, trade: 'Trade', order_type: str, amount: float,
    #                        rate: float, time_in_force: str, exit_reason: str,
    #                        current_time: datetime, **kwargs) -> bool:
    #     """
    #     在实际下单退出前进行最后确认。
    #     """
        # 在退出确认前清理止损存储（双重保险）
        # if trade.id in self.trade_sl_storage:
        #     try:
        #         del self.trade_sl_storage[trade.id]
        #     except KeyError:
        #         pass # 可能已被 custom_exit 删除
        # return True # 批准退出