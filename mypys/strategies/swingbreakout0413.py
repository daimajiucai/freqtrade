# --- 引入必要的库 ---
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame, Series # 引入 Series
import logging
from datetime import datetime, timedelta # 确保 datetime 已引入

# --- Freqtrade 相关引入 ---
from freqtrade.strategy import (IStrategy, DecimalParameter, IntParameter, CategoricalParameter)
from freqtrade.persistence import Trade
from freqtrade.exchange import timeframe_to_prev_date

# --- 类型提示相关引入 ---
from typing import Optional, Any

# --- 可选的库 ---
# (此特定逻辑目前不需要额外的库)

# --- 策略特定导入 ---
# (此特定逻辑目前不需要额外的库)

logger = logging.getLogger(__name__) # 设置日志记录器

# --- 定义策略类 ---
class SwingBreakoutPercentTPPivotSL(IStrategy):
    """
    基于 TradingView "摆动点突破策略 (% TP, Pivot SL)" 的 Freqtrade 策略 (修正版)

    逻辑:
    - 做多入场: 新摆动低点 > 前摆动低点 且 新摆动高点 > 前摆动高点
    - 做空入场: 新摆动高点 < 前摆动高点 且 新摆动低点 < 前摆动低点
    - 止盈: 使用 custom_exit 实现，基于入场价计算的固定百分比 (通过 tp_percent 参数)
    - 止损: 使用 custom_stoploss 实现，前一个摆动低点 (多头) / 前一个摆动高点 (空头) (在入场时固定)
    """

    # 策略接口版本 - 必需
    INTERFACE_VERSION = 3

    # --- 策略配置 ---

    # 此策略是否可以做空?
    can_short: bool = True

    # Minimal ROI: 设置一个非常高的值，让 custom_exit 处理实际的止盈
    # 例如，设置为 1000%，基本上不会被触发
    minimal_roi = {
        "0": 0.01
    }

    # 止损: 设置一个默认的较大止损，实际止损将通过 `custom_stoploss` 动态计算。
    stoploss = -0.99 # 必须定义，但 custom_stoploss 优先。

    # 移动止损: 此策略禁用
    trailing_stop = False

    # 策略的最佳时间框架 (Timeframe)
    timeframe = '1m' # 示例，根据需要调整

    # 仅在新K线上运行 "populate_indicators()"
    process_only_new_candles = True

    # 这些值可以在 config.json 文件中覆盖
    use_exit_signal = True # **必须为 True** 才能让 custom_exit 生效
    exit_profit_only = False # 允许在亏损时由 custom_exit 或 stoploss 退出
    ignore_roi_if_entry_signal = False

    # --- 超参数 / 输入参数 ---
    # 这些取代了 Pine Script 中的 `input.*`

    # 摆动点设置
    pivot_left_bars = IntParameter(1, 15, default=5, space="buy", optimize=True, load=True, description="摆动点左侧周期")
    pivot_right_bars = IntParameter(1, 15, default=5, space="buy", optimize=True, load=True, description="摆动点右侧周期")

    # 止盈百分比 (用于 custom_exit)
    tp_percent = DecimalParameter(0.1, 5.0, default=1.0, decimals=1, space="buy", optimize=True, load=True, description="百分比止盈 (%)")
    use_custom_stoploss = False#不使用自定义止损，使用exit

    # --- 计算指标 (populate_indicators) ---
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        在此方法中计算所有需要的技术指标并添加到 DataFrame 中
        """
        # --- 摆动点计算 ---
        pivot_left = self.pivot_left_bars.value  # 获取左侧 K 线数量
        pivot_right = self.pivot_right_bars.value  # 获取右侧 K 线数量

        # 初始化枢轴点相关的列，确保类型正确
        dataframe['pivot_low_price'] = np.nan  # 存储枢轴低点价格的列
        dataframe['pivot_high_price'] = np.nan  # 存储枢轴高点价格的列
        # 标记是否为枢轴低点/高点的列 (False 表示不是)
        # 使用 pd.Series 初始化以确保索引对齐
        dataframe['is_pivot_low'] = pd.Series(False, index=dataframe.index)
        dataframe['is_pivot_high'] = pd.Series(False, index=dataframe.index)

        # --- 寻找潜在的枢轴点 ---
        # 循环遍历 DataFrame，'i' 代表当前**确认**枢轴点的 K 线索引
        # 'pivot_candle_index' 是潜在的枢轴点 K 线的索引
        # 需要确保潜在枢轴点左右有足够数量的 K 线用于比较
        # 最早可能的枢轴点 K 线索引是 'pivot_left'
        # 因此，最早的确认点索引 'i' 是 'pivot_left + pivot_right'
        for i in range(pivot_left + pivot_right, len(dataframe)):
            # 计算比较窗口的起始索引
            window_start = i - pivot_left - pivot_right
            # 潜在枢轴点 K 线的索引 (确认点索引 - 右侧 K 线数)
            pivot_candle_index = i - pivot_right
            # 对应右侧比较窗口结束的 K 线索引 (即当前确认点 'i')
            right_window_end_index = i

            # --- 检查枢轴低点 (Pivot Low) ---
            potential_pivot_low_val = dataframe['low'].iloc[pivot_candle_index]  # 获取潜在枢轴低点的 'low' 值
            is_pl = True  # 初始化标志，假设当前是枢轴低点

            # 检查左侧 (从 window_start 到 pivot_candle_index - 1)
            for j in range(window_start, pivot_candle_index):
                if dataframe['low'].iloc[j] < potential_pivot_low_val:  # 如果左侧有更低的 'low'
                    is_pl = False  # 则不是枢轴低点
                    break  # 提前退出左侧检查

            # 只有在左侧检查通过时才继续检查右侧
            if is_pl:
                # 检查右侧 (从 pivot_candle_index + 1 到 right_window_end_index)
                for j in range(pivot_candle_index + 1, right_window_end_index + 1):
                    # 注意：严格的 TradingView 比较可能使用 <= (枢轴点必须是严格最低)
                    # 但使用 < 更常见。如果需要精确匹配 TradingView，请仔细检查其行为。
                    # 这里暂时保持和原代码一致，使用 <
                    if dataframe['low'].iloc[j] < potential_pivot_low_val:  # 如果右侧有更低的 'low'
                        is_pl = False  # 则不是枢轴低点
                        break  # 提前退出右侧检查

                # 如果左右两侧检查都通过
                if is_pl:
                    # **** 修正的索引 ****
                    # 将枢轴信息存储在 *实际* 枢轴 K 线的索引处
                    pivot_actual_index = dataframe.index[pivot_candle_index]  # 获取实际枢轴 K 线对应的 DataFrame 索引
                    dataframe.loc[pivot_actual_index, 'pivot_low_price'] = potential_pivot_low_val  # 在实际枢轴位置记录价格
                    dataframe.loc[pivot_actual_index, 'is_pivot_low'] = True  # 在实际枢轴位置标记为枢轴低点

            # --- 检查枢轴高点 (Pivot High) ---
            potential_pivot_high_val = dataframe['high'].iloc[pivot_candle_index]  # 获取潜在枢轴高点的 'high' 值
            is_ph = True  # 初始化标志，假设当前是枢轴高点

            # 检查左侧 (从 window_start 到 pivot_candle_index - 1)
            for j in range(window_start, pivot_candle_index):
                # 注意：严格的 TradingView 比较可能使用 >=
                if dataframe['high'].iloc[j] > potential_pivot_high_val:  # 如果左侧有更高的 'high'
                    is_ph = False  # 则不是枢轴高点
                    break  # 提前退出左侧检查

            # 只有在左侧检查通过时才继续检查右侧
            if is_ph:
                # 检查右侧 (从 pivot_candle_index + 1 到 right_window_end_index)
                for j in range(pivot_candle_index + 1, right_window_end_index + 1):
                    # 注意：严格的 TradingView 比较可能使用 >=
                    if dataframe['high'].iloc[j] > potential_pivot_high_val:  # 如果右侧有更高的 'high'
                        is_ph = False  # 则不是枢轴高点
                        break  # 提前退出右侧检查

                # 如果左右两侧检查都通过
                if is_ph:
                    # **** 修正的索引 ****
                    # 将枢轴信息存储在 *实际* 枢轴 K 线的索引处
                    pivot_actual_index = dataframe.index[pivot_candle_index]  # 获取实际枢轴 K 线对应的 DataFrame 索引
                    dataframe.loc[pivot_actual_index, 'pivot_high_price'] = potential_pivot_high_val  # 在实际枢轴位置记录价格
                    dataframe.loc[pivot_actual_index, 'is_pivot_high'] = True  # 在实际枢轴位置标记为枢轴高点

        # --- 存储枢轴历史 (基于修正后的索引) ---
        # 创建一个 Series，在枢轴点位置有价格，其他位置为 NaN
        dataframe['last_pivot_low_temp'] = np.where(dataframe['is_pivot_low'], dataframe['pivot_low_price'], np.nan)
        dataframe['last_pivot_high_temp'] = np.where(dataframe['is_pivot_high'], dataframe['pivot_high_price'], np.nan)

        # 向前填充 (ffill)，使得每根 K 线都有最近一个已知枢轴点的值
        # 因为 is_pivot_high/low 和 pivot_high/low_price 现在记录在正确的索引上，ffill 会从正确的位置开始填充
        dataframe['last_pivot_low'] = dataframe['last_pivot_low_temp'].ffill()
        dataframe['last_pivot_high'] = dataframe['last_pivot_high_temp'].ffill()

        # 清理临时列
        dataframe.drop(columns=['last_pivot_low_temp', 'last_pivot_high_temp'], inplace=True)

        # --- 计算前一个枢轴点 (修正后此逻辑应该正确) ---
        # 当 is_pivot_low/high 为 True 时 (即当前 K 线是 *新确认* 的枢轴点)，
        # 我们记录下 *这条 K 线之前* 的 'last_pivot_low/high' 的值。
        # .shift(1) 获取上一根 K 线的 'last_pivot' 值，这个值就是新枢轴点出现之前的那个枢轴点的值。
        dataframe['prev_pivot_low'] = np.where(
            dataframe['is_pivot_low'], dataframe['last_pivot_low'].shift(1), np.nan
        )
        dataframe['prev_pivot_high'] = np.where(
            dataframe['is_pivot_high'], dataframe['last_pivot_high'].shift(1), np.nan
        )
        # 向前填充 'previous' 值，直到找到下一个 'previous' (即下一个新枢轴点被确认时)
        dataframe['prev_pivot_low'].ffill(inplace=True)
        dataframe['prev_pivot_high'].ffill(inplace=True)

        # --- 止损价格计算 ---
        # 这里现在应该使用时间正确的最近枢轴点值
        dataframe['entry_sl_price_long'] = dataframe['last_pivot_low']  # 多头入场止损价 (基于最近枢轴低点)
        dataframe['entry_sl_price_short'] = dataframe['last_pivot_high']  # 空头入场止损价 (基于最近枢轴高点)

        # 可选：如果后续不再需要，清理中间计算列
        # dataframe.drop(columns=['pivot_low_price', 'pivot_high_price', 'is_pivot_low', 'is_pivot_high'], inplace=True)

        return dataframe

    # def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
    #                         time_in_force: str, current_time: datetime, entry_tag: Optional[str],
    #                         side: str, **kwargs) -> bool:
    #     """
    #     在下单前确认入场交易。在这里计算并存储自定义止损价格。
    #     """
    #     dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
    #     # 获取信号产生的那根K线的数据 (通常是最后一根可用K线)
    #     signal_candle = dataframe.iloc[-1]
    #
    #     try:
    #         sl_price = None
    #         if side == 'long':
    #             sl_price = signal_candle.get('entry_sl_price_long')
    #             if sl_price is None or pd.isna(sl_price):
    #                 logger.warning(f"{pair} - Entry confirmation: Could not get 'entry_sl_price_long' "
    #                                f"from signal candle {signal_candle['date']}. Cannot set custom SL.")
    #                 # 根据你的策略决定是否要阻止入场，或者允许入场但没有自定义SL
    #                 # return False # 阻止入场
    #                 return True  # 允许入场，将使用默认SL
    #             # 验证止损价逻辑 (可选，但推荐)
    #             if sl_price >= rate:  # 多头止损必须低于入场价 'rate'
    #                 logger.warning(f"{pair} - Entry confirmation: Calculated SL price {sl_price} >= entry rate {rate}. "
    #                                f"Cannot set custom SL.")
    #                 # return False
    #                 return True
    #
    #         elif side == 'short':
    #             sl_price = signal_candle.get('entry_sl_price_short')
    #             if sl_price is None or pd.isna(sl_price):
    #                 logger.warning(f"{pair} - Entry confirmation: Could not get 'entry_sl_price_short' "
    #                                f"from signal candle {signal_candle['date']}. Cannot set custom SL.")
    #                 # return False
    #                 return True
    #             # 验证止损价逻辑 (可选，但推荐)
    #             if sl_price <= rate:  # 空头止损必须高于入场价 'rate'
    #                 logger.warning(f"{pair} - Entry confirmation: Calculated SL price {sl_price} <= entry rate {rate}. "
    #                                f"Cannot set custom SL.")
    #                 # return False
    #                 return True
    #
    #         # 如果获取并验证成功，将止损价格存储在 trade.custom_info 中
    #         # 注意：此时 trade 对象还不存在，我们需要返回 True，
    #         # Freqtrade 会在创建 Trade 对象后调用下面的 populate_entry_trend，
    #         # 或者更稳妥的做法是直接在这里返回，然后在 custom_stoploss 里做最后的补救查找。
    #         # 但是，理论上 confirm_trade_entry 是用来 *修改订单参数* 或 *否决交易* 的。
    #         #
    #         # 更标准的做法是依赖 populate_entry_trend 已经计算好的列，
    #         # 并在 custom_stoploss 里首次访问时，如果 trade.custom_info 为空，
    #         # 则执行一次查找并存储，后续直接读取。
    #
    #         # ---->> 我们将在 custom_stoploss 中实现查找和存储逻辑 <<----

        # except KeyError as e:
        #     logger.error(
        #         f"{pair} - Entry confirmation: KeyError accessing signal candle data: {e}. Columns: {list(signal_candle.index)}")
        #     # return False
        #     return True  # 允许入场但无自定义SL
        # except Exception as e:
        #     logger.error(f"{pair} - Entry confirmation: Unexpected error: {e}")
        #     # return False
        #     return True  # 允许入场但无自定义SL
        #
        # # 如果所有检查都通过，返回 True 以允许交易继续
        # return True

    # --- 入场信号生成 (populate_entry_trend) ---
    # (这部分代码保持不变，和上一个版本一样)
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        基于指标计算入场信号，并过滤掉止损距离过近的信号

        :param dataframe: DataFrame
        :param metadata: 元数据字典，包含货币对信息
        :return: DataFrame with entry signals
        """
        # --- 定义最小止损距离 ---
        # 这个值可以定义在策略类属性中 (self.min_stop_distance_pct)
        # 或者直接在这里定义
        min_stop_distance_pct = 0.002  # 示例：要求止损距离至少为 0.2%

        # --- 做多条件 ---
        # 1. 基本的摆动点趋势条件
        conditions_long_base = (
                (dataframe['last_pivot_low'] > dataframe['prev_pivot_low']) &
                (dataframe['last_pivot_high'] > dataframe['prev_pivot_high']) &
                (dataframe['entry_sl_price_long'].notna())  # 确保止损价有效
        )

        # 2. 止损距离检查 (使用当前 close 作为入场价估计)
        # 距离 = (估计入场价 - 止损价) / 估计入场价
        # 确保分母不为0 (虽然价格通常不会是0) 并且止损价低于收盘价
        # （如果止损价高于当前收盘价，那么这个入场本身就非常危险或无效）
        distance_check_long = (
                (dataframe['close'] > dataframe['entry_sl_price_long']) &  # 基本理智检查
                (((dataframe['close'] - dataframe['entry_sl_price_long']) / dataframe['close'].replace(0,
                                                                                                       np.nan)) > min_stop_distance_pct)
        # 计算距离百分比
        )

        # 3. 合并条件
        conditions_long = conditions_long_base & distance_check_long

        # 4. 设置信号和 tag
        dataframe.loc[conditions_long, 'enter_long'] = 1
        dataframe.loc[conditions_long, 'enter_tag'] = 'sl_long_' + dataframe.loc[
            conditions_long, 'entry_sl_price_long'].astype(str)

        # --- 做空条件 ---
        # 1. 基本的摆动点趋势条件
        conditions_short_base = (
                (dataframe['last_pivot_high'] < dataframe['prev_pivot_high']) &
                (dataframe['last_pivot_low'] < dataframe['prev_pivot_low']) &
                (dataframe['entry_sl_price_short'].notna())  # 确保止损价有效
        )

        # 2. 止损距离检查 (使用当前 close 作为入场价估计)
        # 距离 = (止损价 - 估计入场价) / 估计入场价
        # 确保分母不为0 并且止损价高于收盘价
        distance_check_short = (
                (dataframe['entry_sl_price_short'] > dataframe['close']) &  # 基本理智检查
                (((dataframe['entry_sl_price_short'] - dataframe['close']) / dataframe['close'].replace(0,
                                                                                                        np.nan)) > min_stop_distance_pct)
        # 计算距离百分比
        )

        # 3. 合并条件
        conditions_short = conditions_short_base & distance_check_short

        # 4. 设置信号和 tag
        dataframe.loc[conditions_short, 'enter_short'] = 1
        dataframe.loc[conditions_short, 'enter_tag'] = 'sl_short_' + dataframe.loc[
            conditions_short, 'entry_sl_price_short'].astype(str)

        return dataframe
    # --- 出场信号生成 (populate_exit_trend) ---
    # (这部分代码保持不变，返回 0，因为退出主要由 custom_exit 和 custom_stoploss 控制)
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        根据技术指标，生成退出信号 (此策略中不使用，返回0)
        """
        dataframe.loc[:, 'exit_long'] = 0
        dataframe.loc[:, 'exit_short'] = 0
        return dataframe

    # --- 自定义止损 (custom_stoploss) ---
    # def custom_stoploss(self, pair: str, trade: Trade, current_time: pd.Timestamp,
    #                     current_rate: float, current_profit: float, **kwargs) -> float:
    #     """
    #     自定义止损逻辑。
    #     首次调用时，查找开仓时的止损价并存储为 trade 对象的一个动态属性。
    #     后续调用直接从该动态属性读取。
    #     """
    #     # --- 使用动态附加属性代替 trade.cache ---
    #     # 定义一个属性名 (使用下划线开头以避免潜在冲突)
    #     stop_loss_attr_name = '_stored_stop_loss_price'
    #
    #     # --- 检查 trade 对象是否已经有这个属性 ---
    #     if hasattr(trade, stop_loss_attr_name):
    #         # 如果已存在，直接返回存储的值
    #         stored_sl = getattr(trade, stop_loss_attr_name)
    #         # logger.info(f"DEBUG: Trade {trade.id}: Reading dynamically attached SL price: {stored_sl}")
    #         return stored_sl
    #
    #     # --- 如果没有该属性，执行一次查找和存储 ---
    #     logger.info(f"DEBUG: Trade {trade.id}: Attribute '{stop_loss_attr_name}' not found. Performing lookup...")
    #     dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
    #
    #     try:
    #         # 1. 计算信号K线时间
    #         signal_candle_time = timeframe_to_prev_date(self.timeframe, trade.open_date)
    #         # logger.info(f"DEBUG: Trade {trade.id} open date: {trade.open_date}. Calculated signal candle time: {signal_candle_time}")
    #
    #         # 2. 查找信号K线
    #         open_candle_df = dataframe.loc[dataframe['date'] == signal_candle_time]
    #
    #         # 3. 检查是否找到
    #         if open_candle_df.empty:
    #             logger.warning(
    #                 f"{pair} - Trade {trade.id}: Could not find signal candle data in DataFrame "
    #                 f"for calculated time {signal_candle_time} during initial SL lookup. "
    #                 f"Using default stoploss for this check. DataFrame range: {dataframe['date'].iloc[0]} to {dataframe['date'].iloc[-1]}"
    #             )
    #             return -1
    #
    #         # 4. 获取数据行
    #         open_candle_data: Series = open_candle_df.iloc[0]
    #         # logger.info(f"DEBUG: Trade {trade.id}: Successfully found signal candle data for time {signal_candle_time} during initial lookup.")
    #
    #         # 5. 获取预计算的止损价格
    #         sl_price_raw = None
    #         log_sl_type = ""
    #         if trade.is_short:
    #             sl_price_raw = open_candle_data.get('entry_sl_price_short')
    #             log_sl_type = "Short"
    #         else:
    #             sl_price_raw = open_candle_data.get('entry_sl_price_long')
    #             log_sl_type = "Long"
    #         # logger.info(f"DEBUG: Trade {trade.id} ({log_sl_type}) Raw SL Price from initial lookup: {sl_price_raw}")
    #
    #         # 6. 检查和验证价格
    #         if sl_price_raw is None or pd.isna(sl_price_raw):
    #             logger.warning(f"{pair} - Trade {trade.id}: Signal candle ({signal_candle_time}) "
    #                            f"did not contain a valid pre-calculated SL price during initial lookup. Using default stoploss.")
    #             return -1
    #
    #         sl_price = float(sl_price_raw)
    #
    #         # 7. 验证逻辑
    #         if trade.is_short and sl_price <= trade.open_rate:
    #             logger.warning(
    #                 f"{pair} - Trade {trade.id} ({log_sl_type}) Invalid SL on initial lookup: SL ({sl_price:.8f}) "
    #                 f"<= Open rate ({trade.open_rate:.8f}). Using default stoploss.")
    #             return -1
    #         if not trade.is_short and sl_price >= trade.open_rate:
    #             logger.warning(
    #                 f"{pair} - Trade {trade.id} ({log_sl_type}) Invalid SL on initial lookup: SL ({sl_price:.8f}) "
    #                 f">= Open rate ({trade.open_rate:.8f}). Using default stoploss.")
    #             return -1
    #
    #         # --- 将查找到的有效止损价动态附加到 trade 对象上 ---
    #         setattr(trade, stop_loss_attr_name, sl_price)
    #         logger.info(
    #             f"DEBUG: Trade {trade.id}: Dynamically attached SL price {sl_price:.8f} as attribute '{stop_loss_attr_name}'.")
    #
    #         # 9. 返回本次查找到的有效止损价格
    #         return sl_price
    #
    #     except KeyError as e:
    #         logger.error(
    #             f"{pair} - Trade {trade.id}: KeyError during initial SL lookup for signal candle {signal_candle_time}: {e}. "
    #             f"Available columns: {list(dataframe.columns)}")
    #         return -1
    #     except Exception as e:
    #         logger.error(f"{pair} - Trade {trade.id}: Unexpected error during initial SL lookup: {e}")
    #         return -1


    # --- 自定义退出逻辑 (custom_exit) ---
    # **新增:** 这个函数用于处理基于百分比的止盈
    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs):
        """
        自定义退出逻辑，用于处理基于入场时确定的止损价

        :param pair: 货币对 (e.g. 'BTC/USDT')
        :param trade: trade 对象 (freqtrade.persistence.Trade)
        :param current_time: 当前时间 (datetime)
        :param current_rate: 当前价格 (float)
        :param current_profit: 当前利润率 (float)
        :param **kwargs: 其他可能需要的参数
        :return: Optional[str] - 返回 'stop_loss' 触发止损, 'force_exit' 强制退出, 或 None/不返回保持仓位
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        # last_candle = dataframe.iloc[-1].squeeze() # 如果需要当前蜡烛的其他信息

        tag = trade.enter_tag

        # 检查 tag 是否存在且符合我们定义的格式
        if tag and tag.startswith('sl_'):
            try:
                # 从 tag 中解析出止损价格
                stop_price_str = tag.split('_', 2)[-1]  # 分割 'sl_long_' 或 'sl_short_'
                stop_price = float(stop_price_str)

                # --- 检查止损条件 ---
                if trade.is_short:
                    # 做空止损：当前价格 >= 入场时确定的前摆动高点
                    if current_rate >= stop_price:
                        return 'stop_loss'  # 返回 'stop_loss' 字符串触发止损
                else:
                    # 做多止损：当前价格 <= 入场时确定的前摆动低点
                    if current_rate <= stop_price:
                        return 'stop_loss'  # 返回 'stop_loss' 字符串触发止损

            except ValueError:
                # 如果 tag 格式不正确或无法转换为 float，打印错误信息
                print(f"无法从 tag '{tag}' 解析止损价格。")
                # 或者可以采取其他处理方式，比如强制退出
                # return 'force_exit'

        # 如果没有触发自定义止损，返回 None (或不返回)，让其他退出机制（如止盈）生效
        return None  # 或者直接省略 return 语句
