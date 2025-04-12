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
        "0": 10.0
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

    # --- 计算指标 (populate_indicators) ---
    # (这部分代码保持不变，和上一个版本一样)
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        在此方法中计算所有需要的技术指标并添加到 DataFrame 中
        """
        # --- 摆动点计算 ---
        pivot_left = self.pivot_left_bars.value
        pivot_right = self.pivot_right_bars.value

        dataframe['pivot_low_price'] = np.nan
        dataframe['pivot_high_price'] = np.nan
        dataframe['is_pivot_low'] = False
        dataframe['is_pivot_high'] = False

        for i in range(pivot_left + pivot_right, len(dataframe)):
            window_start = i - pivot_left - pivot_right
            window_end = i # Pine right side includes the pivot candle index + right bars after it
            pivot_candle_index = i - pivot_right

            # Check for Pivot Low
            potential_pivot_low_val = dataframe['low'].iloc[pivot_candle_index]
            is_pl = True
            # Check left side (excluding pivot candle index)
            for j in range(window_start, pivot_candle_index):
                if dataframe['low'].iloc[j] < potential_pivot_low_val:
                    is_pl = False
                    break
            if not is_pl: continue
            # Check right side (up to and including current candle index 'i')
            for j in range(pivot_candle_index + 1, i + 1):
                if dataframe['low'].iloc[j] < potential_pivot_low_val:
                    is_pl = False
                    break
            if is_pl:
                dataframe.loc[dataframe.index[i], 'pivot_low_price'] = potential_pivot_low_val
                dataframe.loc[dataframe.index[i], 'is_pivot_low'] = True

            # Check for Pivot High
            potential_pivot_high_val = dataframe['high'].iloc[pivot_candle_index]
            is_ph = True
            # Check left side
            for j in range(window_start, pivot_candle_index):
                if dataframe['high'].iloc[j] > potential_pivot_high_val:
                    is_ph = False
                    break
            if not is_ph: continue
            # Check right side
            for j in range(pivot_candle_index + 1, i + 1):
                 if dataframe['high'].iloc[j] > potential_pivot_high_val:
                    is_ph = False
                    break
            if is_ph:
                dataframe.loc[dataframe.index[i], 'pivot_high_price'] = potential_pivot_high_val
                dataframe.loc[dataframe.index[i], 'is_pivot_high'] = True

        # --- Store Pivot History ---
        dataframe['last_pivot_low'] = dataframe.loc[dataframe['is_pivot_low'], 'pivot_low_price']
        dataframe['last_pivot_high'] = dataframe.loc[dataframe['is_pivot_high'], 'pivot_high_price']
        dataframe['last_pivot_low'].ffill(inplace=True)
        dataframe['last_pivot_high'].ffill(inplace=True)

        dataframe['prev_pivot_low'] = np.where(
            dataframe['is_pivot_low'], dataframe['last_pivot_low'].shift(1), np.nan
        )
        dataframe['prev_pivot_high'] = np.where(
            dataframe['is_pivot_high'], dataframe['last_pivot_high'].shift(1), np.nan
        )
        dataframe['prev_pivot_low'].ffill(inplace=True)
        dataframe['prev_pivot_high'].ffill(inplace=True)

        # --- Stop Loss Price Calculation ---
        dataframe['entry_sl_price_long'] = dataframe['prev_pivot_low']
        dataframe['entry_sl_price_short'] = dataframe['prev_pivot_high']

        return dataframe

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime, entry_tag: Optional[str],
                            side: str, **kwargs) -> bool:
        """
        在下单前确认入场交易。在这里计算并存储自定义止损价格。
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        # 获取信号产生的那根K线的数据 (通常是最后一根可用K线)
        signal_candle = dataframe.iloc[-1]

        try:
            sl_price = None
            if side == 'long':
                sl_price = signal_candle.get('entry_sl_price_long')
                if sl_price is None or pd.isna(sl_price):
                    logger.warning(f"{pair} - Entry confirmation: Could not get 'entry_sl_price_long' "
                                   f"from signal candle {signal_candle['date']}. Cannot set custom SL.")
                    # 根据你的策略决定是否要阻止入场，或者允许入场但没有自定义SL
                    # return False # 阻止入场
                    return True  # 允许入场，将使用默认SL
                # 验证止损价逻辑 (可选，但推荐)
                if sl_price >= rate:  # 多头止损必须低于入场价 'rate'
                    logger.warning(f"{pair} - Entry confirmation: Calculated SL price {sl_price} >= entry rate {rate}. "
                                   f"Cannot set custom SL.")
                    # return False
                    return True

            elif side == 'short':
                sl_price = signal_candle.get('entry_sl_price_short')
                if sl_price is None or pd.isna(sl_price):
                    logger.warning(f"{pair} - Entry confirmation: Could not get 'entry_sl_price_short' "
                                   f"from signal candle {signal_candle['date']}. Cannot set custom SL.")
                    # return False
                    return True
                # 验证止损价逻辑 (可选，但推荐)
                if sl_price <= rate:  # 空头止损必须高于入场价 'rate'
                    logger.warning(f"{pair} - Entry confirmation: Calculated SL price {sl_price} <= entry rate {rate}. "
                                   f"Cannot set custom SL.")
                    # return False
                    return True

            # 如果获取并验证成功，将止损价格存储在 trade.custom_info 中
            # 注意：此时 trade 对象还不存在，我们需要返回 True，
            # Freqtrade 会在创建 Trade 对象后调用下面的 populate_entry_trend，
            # 或者更稳妥的做法是直接在这里返回，然后在 custom_stoploss 里做最后的补救查找。
            # 但是，理论上 confirm_trade_entry 是用来 *修改订单参数* 或 *否决交易* 的。
            #
            # 更标准的做法是依赖 populate_entry_trend 已经计算好的列，
            # 并在 custom_stoploss 里首次访问时，如果 trade.custom_info 为空，
            # 则执行一次查找并存储，后续直接读取。

            # ---->> 我们将在 custom_stoploss 中实现查找和存储逻辑 <<----

        except KeyError as e:
            logger.error(
                f"{pair} - Entry confirmation: KeyError accessing signal candle data: {e}. Columns: {list(signal_candle.index)}")
            # return False
            return True  # 允许入场但无自定义SL
        except Exception as e:
            logger.error(f"{pair} - Entry confirmation: Unexpected error: {e}")
            # return False
            return True  # 允许入场但无自定义SL

        # 如果所有检查都通过，返回 True 以允许交易继续
        return True

    # --- 入场信号生成 (populate_entry_trend) ---
    # (这部分代码保持不变，和上一个版本一样)
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        根据技术指标，生成入场信号
        """
        long_condition = (
            (dataframe['last_pivot_low'] > dataframe['prev_pivot_low']) &
            (dataframe['last_pivot_high'] > dataframe['prev_pivot_high']) &
            dataframe['last_pivot_low'].notna() &
            dataframe['prev_pivot_low'].notna() &
            dataframe['last_pivot_high'].notna() &
            dataframe['prev_pivot_high'].notna() &
            (dataframe['volume'] > 0)
        )
        short_condition = (
            (dataframe['last_pivot_high'] < dataframe['prev_pivot_high']) &
            (dataframe['last_pivot_low'] < dataframe['prev_pivot_low']) &
            dataframe['last_pivot_low'].notna() &
            dataframe['prev_pivot_low'].notna() &
            dataframe['last_pivot_high'].notna() &
            dataframe['prev_pivot_high'].notna() &
            (dataframe['volume'] > 0)
        )
        dataframe.loc[long_condition, ['enter_long', 'enter_tag']] = (1, 'long_entry_signal')
        dataframe.loc[short_condition, ['enter_short', 'enter_tag']] = (1, 'short_entry_signal')
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
    def custom_stoploss(self, pair: str, trade: Trade, current_time: pd.Timestamp,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        """
        自定义止损逻辑。
        首次调用时，查找开仓时的止损价并存储在 trade.custom_info。
        后续调用直接从 trade.custom_info 读取。
        """
        # --- 检查是否已存储止损价 ---
        if 'stored_stop_loss_price' in trade.custom_info:
            # 如果已存储，直接返回
            # logger.info(f"DEBUG: Trade {trade.id}: Reading stored SL price: {trade.custom_info['stored_stop_loss_price']}")
            return trade.custom_info['stored_stop_loss_price']

        # --- 如果未存储，执行一次查找和存储 ---
        logger.info(f"DEBUG: Trade {trade.id}: First call to custom_stoploss or SL not stored. Performing lookup...")
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        try:
            # 1. 计算信号K线时间 (与之前相同)
            signal_candle_time = timeframe_to_prev_date(self.timeframe, trade.open_date)
            logger.info(
                f"DEBUG: Trade {trade.id} open date: {trade.open_date}. Calculated signal candle time: {signal_candle_time}")

            # 2. 查找信号K线
            open_candle_df = dataframe.loc[dataframe['date'] == signal_candle_time]

            # 3. 检查是否找到
            if open_candle_df.empty:
                # 这个问题仍然可能在首次调用时发生（如果数据窗口刚好不包含）
                logger.warning(
                    f"{pair} - Trade {trade.id}: Could not find signal candle data in DataFrame "
                    f"for calculated time {signal_candle_time} during initial SL lookup. "
                    f"Using default stoploss for this check. DataFrame range: {dataframe['date'].iloc[0]} to {dataframe['date'].iloc[-1]}"
                )
                # 即使这次找不到，也不存储任何东西，下次可能会找到（如果数据窗口变化）
                # 或者，如果认为这种情况不应发生，可以返回一个明确的错误或固定值
                return -1  # 使用默认止损

            # 4. 获取数据行
            open_candle_data: Series = open_candle_df.iloc[0]
            logger.info(
                f"DEBUG: Trade {trade.id}: Successfully found signal candle data for time {signal_candle_time} during initial lookup.")

            # 5. 获取预计算的止损价格
            sl_price_raw = None
            log_sl_type = ""
            if trade.is_short:
                sl_price_raw = open_candle_data.get('entry_sl_price_short')
                log_sl_type = "Short"
            else:
                sl_price_raw = open_candle_data.get('entry_sl_price_long')
                log_sl_type = "Long"
            logger.info(f"DEBUG: Trade {trade.id} ({log_sl_type}) Raw SL Price from initial lookup: {sl_price_raw}")

            # 6. 检查和验证价格
            if sl_price_raw is None or pd.isna(sl_price_raw):
                logger.warning(f"{pair} - Trade {trade.id}: Signal candle ({signal_candle_time}) "
                               f"did not contain a valid pre-calculated SL price during initial lookup. Using default stoploss.")
                return -1

            sl_price = float(sl_price_raw)  # 转换为 float

            # 7. 验证逻辑 (相对于 trade.open_rate!)
            if trade.is_short and sl_price <= trade.open_rate:
                logger.warning(
                    f"{pair} - Trade {trade.id} ({log_sl_type}) Invalid SL on initial lookup: SL ({sl_price:.8f}) "
                    f"<= Open rate ({trade.open_rate:.8f}). Using default stoploss.")
                return -1
            if not trade.is_short and sl_price >= trade.open_rate:
                logger.warning(
                    f"{pair} - Trade {trade.id} ({log_sl_type}) Invalid SL on initial lookup: SL ({sl_price:.8f}) "
                    f">= Open rate ({trade.open_rate:.8f}). Using default stoploss.")
                return -1

            # --- 存储查找到的有效止损价 ---
            trade.custom_info['stored_stop_loss_price'] = sl_price
            logger.info(f"DEBUG: Trade {trade.id}: Stored SL price {sl_price:.8f} in trade.custom_info.")

            # 9. 返回本次查找到的有效止损价格
            return sl_price

        except KeyError as e:
            logger.error(
                f"{pair} - Trade {trade.id}: KeyError during initial SL lookup for signal candle {signal_candle_time}: {e}. "
                f"Available columns: {list(dataframe.columns)}")
            return -1
        except Exception as e:
            logger.error(f"{pair} - Trade {trade.id}: Unexpected error during initial SL lookup: {e}")
            return -1


    # --- 自定义退出逻辑 (custom_exit) ---
    # **新增:** 这个函数用于处理基于百分比的止盈
    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs):
        """
        自定义退出逻辑，用于处理百分比止盈。
        :param pair: Pair open trade relative to (e.g. BTC/USDT)
        :param trade: Trade object.
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Rate, calculated based on pricing settings in config.
        :param current_profit: Current profit (as ratio), calculated based on current_rate.
        :param **kwargs: Ensure to keep this for future compatibility.
        :return: String value indicating the exit reason, or None if no exit should happen.
                 Returning 'force_exit' will execute the trade immediately.
        """
        # 计算目标盈利比例
        target_profit = self.tp_percent.value / 100.0

        # 检查当前利润是否达到或超过目标利润
        if current_profit >= target_profit:
            # 返回一个自定义的退出原因字符串（例如 'take_profit_custom'）
            # Freqtrade 会记录这个原因，并执行市价退出
            # 或者你可以返回 'force_exit' 来强制立即退出
            return 'take_profit_custom' # 或者 'force_exit'

        # 如果未达到止盈条件，则返回 None 或不返回任何东西 (等同于返回 None)
        return None

    # --- 移除了 minimal_roi 的 @property ---
    # 我们不再需要动态计算 minimal_roi 了，因为它现在只是一个占位符