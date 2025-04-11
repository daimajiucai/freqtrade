# --- 策略引用和依赖 ---
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import talib.abstract as ta
import pandas as pd
from pandas import DataFrame, Series
from freqtrade.strategy import IStrategy, CategoricalParameter, DecimalParameter, IntParameter, merge_informative_pair
from freqtrade.persistence import Trade  # 引用Trade对象，用于自定义止损
from datetime import datetime, timedelta, timezone # 用于处理时间

# --- 策略主体 ---
class SmcFvgStrategy0409(IStrategy):
    """
    基于SMC FVG概念的双时间框架策略框架
    主时间框架 (Entry): 1m
    信息时间框架 (Signal): 15m

    策略逻辑概要:
    1. 在15m timeframe上识别看涨/看跌FVG。
    2. 根据相对于前期高/低点的折扣区/溢价区判断15m FVG的有效性。
    3. 存储所有识别到的FVG（待实现细节）。
    4. FVG被触碰一次后失效（待实现细节）。
    5. 当有效的15m FVG出现，准备入场。
    6. 在1m timeframe上寻找精确入场点（此处简化为触碰15m FVG边界，待细化）。
    7. 设置固定比例止盈和基于前期高/低点的止损。
    8. **优化：一个15m FVG只触发一次挂单/入场信号。**
    """

    # --- 策略核心参数 ---
    INTERFACE_VERSION = 3  # Freqtrade接口版本

    # ROI Tabe (止盈设置) - 0分钟后止盈0.5%
    minimal_roi = {
        "0": 0.01  # 0.5%
    }

    # Stoploss (止损设置) - 使用自定义止损 `custom_stoploss`
    stoploss = -0.99 # 设置一个较大的值，因为我们将使用custom_stoploss

    # Trailing Stop (追踪止损) - 暂时不使用
    trailing_stop = False
    # trailing_stop_positive = 0.001
    # trailing_stop_positive_offset = 0.002
    # trailing_only_offset_is_reached = True
    use_custom_stoploss = True # <--- 显式确保启用自定义止损
    # Timeframe (时间框架)
    timeframe = '1m'  # 交易和入场确认时间框架
    informative_timeframe = '15m' # 信号产生时间框架

    # Run "populate_indicators()" only for new candles
    process_only_new_candles = True

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Number of candles the strategy requires before producing valid signals
    # 需要根据指标计算所需的最大lookback期间来设置
    # 例如: swing high/low 需要 lookback, FVG需要3根K线
    startup_candle_count: int = 50 # 初始设置一个值，后续根据需要调整

    # --- 可配置参数 ---
    # 用于定义前期高低点的回看周期
    swing_lookback = IntParameter(low=10, high=100, default=20, space='buy', optimize=True, load=True)

    # --- Informative Pairs 定义 ---
    def informative_pairs(self):
        """
        定义策略需要使用的其他交易对或时间框架的数据
        """
        pair = self.dp.current_whitelist()[0] # 获取当前白名单中的第一个交易对
        # 返回 (交易对, 时间框架) 的列表
        return [(pair, self.informative_timeframe)]

    # --- 指标计算 ---
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        计算策略所需的技术指标

        Args:
            dataframe (DataFrame): 主时间框架 (1m) 的 OHLCV 数据
            metadata (dict): 包含交易对信息的字典

        Returns:
            DataFrame: 增加了指标列的 DataFrame
        """
        # --- 1. 获取信息时间框架 (15m) 的数据 ---
        if not self.dp:
            print(f"数据提供者不可用，无法获取信息数据 not available for {metadata['pair']} at {dataframe['date'].iloc[-1]}") # 添加打印
             # 数据提供者不可用，无法获取信息数据
            return dataframe

        # 获取15m数据
        informative_df = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.informative_timeframe)

        # 添加打印：检查获取到的 informative_df
        if informative_df.empty:
            print(f"Warning: 未获取到 informative_df for {metadata['pair']} at {dataframe['date'].iloc[-1]}")
        else:
            print(f"获取到 informative_df for {metadata['pair']}, shape: {informative_df.shape}, last date: {informative_df['date'].iloc[-1]}")

        # --- 2. 在 15m 数据上计算 FVG 和 Swing High/Low ---
        try:  # 使用 try...except 包裹，以便看到 calculate_fvg 内部的错误
            informative_df = self.calculate_fvg(informative_df.copy(),
                                                prefix=f'inf_{self.informative_timeframe}')  # 使用 .copy() 避免 SettingWithCopyWarning
            # 添加打印：检查 calculate_fvg 是否添加了列
            print(
                f"Columns in informative_df after calculate_fvg for {metadata['pair']}: {informative_df.columns.tolist()}")
            if f'inf_{self.informative_timeframe}_bull_fvg_detected' not in informative_df.columns:
                print(
                    f"ERROR: 'inf_15m_bull_fvg_detected' column MISSING in informative_df after calculate_fvg for {metadata['pair']}!")

            informative_df = self.calculate_swing_high_low(informative_df.copy(), lookback=self.swing_lookback.value,
                                                           prefix=f'inf_{self.informative_timeframe}')
            # 添加打印：检查 calculate_swing_high_low 是否添加了列
            print(
                f"Columns in informative_df after calculate_swing_high_low for {metadata['pair']}: {informative_df.columns.tolist()}")

        except Exception as e:
            print(f"ERROR 指标计算出错 informative_df for {metadata['pair']}: {e}")
            # 如果指标计算出错，也可能导致列缺失，直接返回避免后续错误
            return dataframe

        # --- 3. 将 15m 指标合并到 1m DataFrame ---
        # 添加打印：检查合并前主 dataframe 的列
        print(f"合并前的列 {metadata['pair']}: {dataframe.columns.tolist()}")
        print(f"合并前的列2 {metadata['pair']}: {informative_df.columns.tolist()}")
        # 使用 merge_informative_pair 函数进行合并，合并后会在15m框架的列名后增加后缀 _15m
        try: # 使用 try...except 包裹合并操作
            dataframe = merge_informative_pair(dataframe, informative_df, self.timeframe, self.informative_timeframe, ffill=True)
            # 添加打印：检查合并后主 dataframe 的列
            print(f"合并后的列 {metadata['pair']}: {dataframe.columns.tolist()}")
            if f'inf_{self.informative_timeframe}_bull_fvg_detected_{self.informative_timeframe}' not in dataframe.columns:
                print(f"ERROR: 'inf_15m_bull_fvg_detected' column MISSING in main dataframe AFTER merge for {metadata['pair']}!")

        except Exception as e:
            print(f"ERROR 合并操作失败 {metadata['pair']}: {e}")
            # 如果合并出错，后续代码会失败，直接返回
            return dataframe

        # --- 4. 在 1m 数据上计算 FVG (如果需要用于精确入场) ---
        # 目前框架简化，暂时不在1m上强制计算FVG用于入场，但保留计算框架
        # dataframe = self.calculate_fvg(dataframe, prefix='tf_1m')
        # dataframe = self.calculate_swing_high_low(dataframe, lookback=self.swing_lookback.value, prefix='tf_1m') # 如果需要1m的swing点
        # --- 检查点：在尝试访问之前，再次确认列是否存在 ---
        # --- 5. 检查关键列是否存在 ---
        required_columns = [
            f'inf_{self.informative_timeframe}_bull_fvg_detected_{self.informative_timeframe}',
            f'inf_{self.informative_timeframe}_fvg_bull_top_{self.informative_timeframe}',
            f'inf_{self.informative_timeframe}_fvg_bull_bottom_{self.informative_timeframe}', # Added for completeness
            f'inf_{self.informative_timeframe}_bear_fvg_detected_{self.informative_timeframe}',
            f'inf_{self.informative_timeframe}_fvg_bear_top_{self.informative_timeframe}',   # Added for completeness
            f'inf_{self.informative_timeframe}_fvg_bear_bottom_{self.informative_timeframe}',
            f'inf_{self.informative_timeframe}_swing_high_{self.informative_timeframe}',   # Needed for stoploss
            f'inf_{self.informative_timeframe}_swing_low_{self.informative_timeframe}',    # Needed for stoploss
            f'inf_{self.informative_timeframe}_swing_midpoint_{self.informative_timeframe}',
            f'inf_{self.informative_timeframe}_fvg_candle_time_{self.informative_timeframe}' # 新增：FVG形成时间戳
        ]
        missing_cols = [col for col in required_columns if col not in dataframe.columns]
        if missing_cols:
            print(f"FATAL: Missing required columns after merge for {metadata['pair']} at {dataframe['date'].iloc[-1]}: {missing_cols}. Returning original dataframe.")
            return dataframe

        # --- 5. 判断 15m FVG 的有效性 (折扣区/溢价区) ---
        dataframe[f'inf_{self.informative_timeframe}_bull_fvg_valid_{self.informative_timeframe}'] = (
            (dataframe[f'inf_{self.informative_timeframe}_bull_fvg_detected_{self.informative_timeframe}']) &
            (dataframe[f'inf_{self.informative_timeframe}_fvg_bull_top_{self.informative_timeframe}'] < dataframe[f'inf_{self.informative_timeframe}_swing_midpoint_{self.informative_timeframe}']) # FVG顶部低于50%
        )
        dataframe[f'inf_{self.informative_timeframe}_bear_fvg_valid_{self.informative_timeframe}'] = (
            (dataframe[f'inf_{self.informative_timeframe}_bear_fvg_detected_{self.informative_timeframe}']) &
            (dataframe[f'inf_{self.informative_timeframe}_fvg_bear_bottom_{self.informative_timeframe}'] > dataframe[f'inf_{self.informative_timeframe}_swing_midpoint_{self.informative_timeframe}']) # FVG底部高于50%
        )

        # --- 待实现: FVG 存储和失效逻辑 ---
        # TODO: 实现一个机制来存储所有FVG及其状态（活跃/失效）
        # TODO: 实现检测价格触碰FVG并将其标记为失效的逻辑

        # 打印一些调试信息 (可选)
        # print(f"Pair: {metadata['pair']}")
        # print("Last 5 rows of dataframe with indicators:")
        # print(dataframe[['date', 'close', 'inf_15m_bull_fvg_valid', 'inf_15m_bear_fvg_valid', 'inf_15m_fvg_bull_top', 'inf_15m_fvg_bear_bottom', 'inf_15m_swing_high', 'inf_15m_swing_low', 'inf_15m_swing_midpoint']].tail())

        return dataframe

    # --- FVG 计算辅助函数 ---
    def calculate_fvg(self, df: DataFrame, prefix: str) -> DataFrame:
        """
        计算给定DataFrame的FVG (Fair Value Gap)

        Args:
            df (DataFrame): 输入的OHLCV数据
            prefix (str): 添加到结果列名的前缀 (e.g., 'inf_15m' or 'tf_1m')

        Returns:
            DataFrame: 增加了FVG相关列的DataFrame
        """
        # FVG需要比较 candle i, i-1, i-2
        df[f'{prefix}_high_i_minus_2'] = df['high'].shift(2)
        df[f'{prefix}_low_i_minus_2'] = df['low'].shift(2)
        df[f'{prefix}_high_i'] = df['high'] # 当前 high
        df[f'{prefix}_low_i'] = df['low']   # 当前 low

        # 看涨 FVG (Bullish FVG): candle i 的 low 高于 candle i-2 的 high
        # FVG 区域: [candle i-2 high, candle i low]
        # FVG 被 candle i-1 标识
        df[f'{prefix}_bull_fvg_detected'] = (df[f'{prefix}_low_i'] > df[f'{prefix}_high_i_minus_2'])
        # FVG 的顶部 = candle i 的 low
        df[f'{prefix}_fvg_bull_top'] = df[f'{prefix}_low_i']
        # FVG 的底部 = candle i-2 的 high
        df[f'{prefix}_fvg_bull_bottom'] = df[f'{prefix}_high_i_minus_2']
        # 将非FVG行的区域设为NaN
        df.loc[~df[f'{prefix}_bull_fvg_detected'], [f'{prefix}_fvg_bull_top', f'{prefix}_fvg_bull_bottom']] = np.nan
        # FVG 识别K线的时间 (candle i-1 的时间) - 注意shift后索引对齐
        # 我们将FVG信息附加到产生这个缺口的第三根K线（即当前K线i）上，shift(1)可以获取i-1的信息

        # 看跌 FVG (Bearish FVG): candle i 的 high 低于 candle i-2 的 low
        # FVG 区域: [candle i high, candle i-2 low]
        # FVG 被 candle i-1 标识
        df[f'{prefix}_bear_fvg_detected'] = (df[f'{prefix}_high_i'] < df[f'{prefix}_low_i_minus_2'])
        # FVG 的顶部 = candle i-2 的 low
        df[f'{prefix}_fvg_bear_top'] = df[f'{prefix}_low_i_minus_2']
        # FVG 的底部 = candle i 的 high
        df[f'{prefix}_fvg_bear_bottom'] = df[f'{prefix}_high_i']
        # 将非FVG行的区域设为NaN
        df.loc[~df[f'{prefix}_bear_fvg_detected'], [f'{prefix}_fvg_bear_top', f'{prefix}_fvg_bear_bottom']] = np.nan

        # 我们将时间戳记录在 FVG 被 *确认* 的那根 K 线 (candle i) 上
        df[f'{prefix}_fvg_candle_time'] = pd.NaT # 初始化为 Not-a-Time
        # 仅在 FVG *首次* 被检测到的那根K线上记录时间戳 (避免在连续FVG中重复记录)
        # is_new_bull_fvg = df[f'{prefix}_bull_fvg_detected'] & ~df[f'{prefix}_bull_fvg_detected'].shift(1).fillna(False)
        # is_new_bear_fvg = df[f'{prefix}_bear_fvg_detected'] & ~df[f'{prefix}_bear_fvg_detected'].shift(1).fillna(False)
        # df.loc[is_new_bull_fvg | is_new_bear_fvg, f'{prefix}_fvg_candle_time'] = df['date']
        # 简化逻辑：只要当前K线检测到FVG，就记录当前K线时间。merge_informative_pair的ffill会处理后续。
        # 这意味着同一个15m FVG，其关联的 `fvg_candle_time` 会是形成它的那根15m K线的时间。
        df.loc[df[f'{prefix}_bull_fvg_detected'] | df[f'{prefix}_bear_fvg_detected'], f'{prefix}_fvg_candle_time'] = df['date']

        # 清理辅助列 (可选)
        # df.drop(columns=[f'{prefix}_high_i_minus_2', f'{prefix}_low_i_minus_2', f'{prefix}_high_i', f'{prefix}_low_i'], inplace=True)

        # 将 FVG 状态向前填充，使得在FVG形成后的K线上也能知道最近的FVG信息（简化处理，后续应替换为存储机制）
        # 注意：这只是权宜之计，真正的状态管理需要更复杂的逻辑
        # df[f'{prefix}_bull_fvg_detected'] = df[f'{prefix}_bull_fvg_detected'].fillna(method='ffill', limit=5) # 限制填充K线数
        # df[f'{prefix}_bear_fvg_detected'] = df[f'{prefix}_bear_fvg_detected'].fillna(method='ffill', limit=5)
        # df[f'{prefix}_fvg_bull_top'] = df[f'{prefix}_fvg_bull_top'].fillna(method='ffill', limit=5)
        # df[f'{prefix}_fvg_bull_bottom'] = df[f'{prefix}_fvg_bull_bottom'].fillna(method='ffill', limit=5)
        # df[f'{prefix}_fvg_bear_top'] = df[f'{prefix}_fvg_bear_top'].fillna(method='ffill', limit=5)
        # df[f'{prefix}_fvg_bear_bottom'] = df[f'{prefix}_fvg_bear_bottom'].fillna(method='ffill', limit=5)

        return df

    # --- Swing High/Low 计算辅助函数 ---
    def calculate_swing_high_low(self, df: DataFrame, lookback: int, prefix: str) -> DataFrame:
        """
        计算给定DataFrame的前期高点 (Swing High) 和前期低点 (Swing Low)

        Args:
            df (DataFrame): 输入的OHLCV数据
            lookback (int): 回看周期
            prefix (str): 添加到结果列名的前缀

        Returns:
            DataFrame: 增加了Swing High/Low相关列的DataFrame
        """
        # 计算回看期内的高点和低点
        # 使用 shift(1) 是为了确保我们寻找的是 "之前" 的高低点，不包含当前K线
        df[f'{prefix}_swing_high'] = df['high'].shift(1).rolling(window=lookback, min_periods=min(lookback, 5)).max()
        df[f'{prefix}_swing_low'] = df['low'].shift(1).rolling(window=lookback, min_periods=min(lookback, 5)).min()
        # 计算 Swing Range 的中点 (50% level)
        df[f'{prefix}_swing_midpoint'] = (df[f'{prefix}_swing_high'] + df[f'{prefix}_swing_low']) / 2

        return df

    # --- 入场信号 ---
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        根据指标生成入场信号 (long / short)

        Args:
            dataframe (DataFrame): 带有指标的 DataFrame
            metadata (dict): 交易对信息

        Returns:
            DataFrame: 增加了 entry_long 和 entry_short 列的 DataFrame
        """

        # --- 入场条件 ---
        # 做多条件 (Long Entry):
        # 1. 检测到有效的 15m 看涨 FVG (inf_15m_bull_fvg_valid == True)
        # 2. 当前 1m K线的最低价触碰或进入了该 15m FVG 的上边界 (low <= inf_15m_fvg_bull_top)
        #    注意：这里简化了入场逻辑，实际可能需要更精细的1m确认
        # 3. (可选) 增加其他过滤条件，例如波动性、成交量等
        # dataframe.loc[
        #     (
        #         (dataframe[f'inf_{self.informative_timeframe}_bull_fvg_valid_{self.informative_timeframe}'] == True) &
        #         (dataframe['low'] <= dataframe[f'inf_{self.informative_timeframe}_fvg_bull_top_{self.informative_timeframe}']) & # 价格触及或进入15m FVG上沿
        #         # (dataframe['low'] >= dataframe['inf_15m_fvg_bull_bottom']) & # (可选) 价格仍在FVG内部
        #          (dataframe['volume'] > 0) # 基础的成交量过滤
        #     ),
        #     ['enter_long', 'enter_tag']] = (1, f'{self.informative_timeframe}_bull_fvg_entry') # 设置信号和标签

        # # 做空条件 (Short Entry):
        # # 1. 检测到有效的 15m 看跌 FVG (inf_15m_bear_fvg_valid == True)
        # # 2. 当前 1m K线的最高价触碰或进入了该 15m FVG 的下边界 (high >= inf_15m_fvg_bear_bottom)
        # # 3. (可选) 增加其他过滤条件
        # dataframe.loc[
        #     (
        #         (dataframe[f'inf_{self.informative_timeframe}_bear_fvg_valid_{self.informative_timeframe}'] == True) &
        #         (dataframe['high'] >= dataframe[f'inf_{self.informative_timeframe}_fvg_bear_bottom_{self.informative_timeframe}']) & # 价格触及或进入15m FVG下沿
        #         # (dataframe['high'] <= dataframe['inf_15m_fvg_bear_top']) & # (可选) 价格仍在FVG内部
        #         (dataframe['volume'] > 0) # 基础的成交量过滤
        #     ),
        #     ['enter_short', 'enter_tag']] = (1, f'{self.informative_timeframe}_bear_fvg_entry') # 设置信号和标签

        # # --- 待实现: 防止在失效FVG处入场 ---
        # # TODO: 确保入场信号不会在已经被标记为失效的FVG上触发
        # --- 先初始化输出列 ---
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        dataframe['enter_tag'] = None

        # --- 获取必要的列名 ---
        inf_timeframe = self.informative_timeframe
        bull_fvg_valid_col = f'inf_{inf_timeframe}_bull_fvg_valid_{inf_timeframe}'
        bull_fvg_top_col = f'inf_{inf_timeframe}_fvg_bull_top_{inf_timeframe}'
        bear_fvg_valid_col = f'inf_{inf_timeframe}_bear_fvg_valid_{inf_timeframe}'
        bear_fvg_bottom_col = f'inf_{inf_timeframe}_fvg_bear_bottom_{inf_timeframe}'
        fvg_time_col = f'inf_{inf_timeframe}_fvg_candle_time_{inf_timeframe}' # FVG形成的时间戳列

        # 检查时间戳列是否存在
        if fvg_time_col not in dataframe.columns:
            print(f"Warning: FVG timestamp column '{fvg_time_col}' not found for {metadata['pair']}. Skipping entry signal generation.")
            return dataframe

        # --- 1. 计算原始触发条件 (Raw Trigger) ---
        # 做多原始触发: 有效15m看涨FVG + 1m low触碰FVG上边界
        dataframe['raw_trigger_long'] = (
            (dataframe[bull_fvg_valid_col] == True) &
            (dataframe['low'] <= dataframe[bull_fvg_top_col]) &
            # (dataframe['low'] >= dataframe[f'inf_{inf_timeframe}_fvg_bull_bottom_{inf_timeframe}']) & # Optional: price inside FVG
            (dataframe['volume'] > 0)
        ).astype(int) # 转为 0 或 1

        # 做空原始触发: 有效15m看跌FVG + 1m high触碰FVG下边界
        dataframe['raw_trigger_short'] = (
            (dataframe[bear_fvg_valid_col] == True) &
            (dataframe['high'] >= dataframe[bear_fvg_bottom_col]) &
            # (dataframe['high'] <= dataframe[f'inf_{inf_timeframe}_fvg_bear_top_{inf_timeframe}']) & # Optional: price inside FVG
            (dataframe['volume'] > 0)
        ).astype(int) # 转为 0 或 1

        # --- 2. 过滤重复信号：每个15m FVG只触发一次 ---
        # 我们使用 FVG 形成的时间戳 (fvg_time_col) 来唯一标识一个 FVG 事件。
        # 对每个 FVG 时间戳分组，计算原始触发信号的累积和 (cumsum)。
        # 只有 cumsum == 1 的那根 1m K 线才是有效的第一次触发。

        # 按 FVG 时间戳分组计算累积触发次数 (需要处理 NaN 时间戳，代表没有活跃 FVG)
        # fillna(pd.NaT) 可能不足以让groupby工作，用一个不可能的时间或特殊值代替?
        # 或者只在有效时间戳上计算 groupby
        dataframe['fvg_entry_count_long'] = dataframe.loc[dataframe[fvg_time_col].notna()].groupby(fvg_time_col)['raw_trigger_long'].cumsum()
        dataframe['fvg_entry_count_short'] = dataframe.loc[dataframe[fvg_time_col].notna()].groupby(fvg_time_col)['raw_trigger_short'].cumsum()

        # 将计算结果填回整个 dataframe (NaNs in count columns where timestamp was NaN)
        # The groupby result index aligns with the original index where timestamp was notna.

        # --- 3. 生成最终入场信号 ---
        # 做多条件: 原始触发为1 且 是该FVG的第一次触发 (cumsum == 1)
        dataframe.loc[
            (dataframe['raw_trigger_long'] == 1) &
            (dataframe['fvg_entry_count_long'] == 1),
            ['enter_long', 'enter_tag']
        ] = (1, f'{inf_timeframe}_bull_fvg_entry')

        # 做空条件: 原始触发为1 且 是该FVG的第一次触发 (cumsum == 1)
        dataframe.loc[
            (dataframe['raw_trigger_short'] == 1) &
            (dataframe['fvg_entry_count_short'] == 1),
            ['enter_short', 'enter_tag']
        ] = (1, f'{inf_timeframe}_bear_fvg_entry')

        # --- 清理辅助列 (可选) ---
        # dataframe.drop(columns=['raw_trigger_long', 'raw_trigger_short', 'fvg_entry_count_long', 'fvg_entry_count_short'], inplace=True, errors='ignore')

        return dataframe

    # --- 出场信号 ---
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        生成自定义出场信号 (非必须，因为我们主要使用ROI和自定义止损)

        Args:
            dataframe (DataFrame): 带有指标的 DataFrame
            metadata (dict): 交易对信息

        Returns:
            DataFrame: 增加了 exit_long 和 exit_short 列的 DataFrame
        """
        # 默认情况下，让 ROI 和止损处理退出
        # 如果需要基于指标的退出逻辑，可以在这里添加
        # dataframe.loc[:, ['exit_long', 'exit_short']] = (0, 0) # 明确置零
        return dataframe

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        """
        自定义止损逻辑，基于入场时的前期高/低点，并包含详细的调试日志。

        Args:
            pair (str): 交易对
            trade (Trade): Freqtrade 的 Trade 对象
            current_time (datetime): 当前时间 (通常是 UTC)
            current_rate (float): 当前价格
            current_profit (float): 当前利润率
            **kwargs: 其他可能需要的参数

        Returns:
            float: 需要设置的绝对止损价格。 返回 -1.0 表示使用配置文件中的默认止损。
        """
        # --- 日志: 函数开始 ---
        # print(f"【止损调试 {trade.id}】: 时间={current_time}, 交易对={pair}, 当前价格={current_rate}, 当前利润={current_profit:.2%}")
        # return -0.01 # 返回一个小的负数，表示使用自定义止损价
        # --- 1. 获取分析后的数据 ---
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        if dataframe.empty:
            print(f"【止损调试 {trade.id}】: 错误 - 无法获取分析数据框(dataframe)。返回-1。")
            return -1.0

        # --- 2. 处理开仓时间 (确保是 UTC) ---
        # Freqtrade 通常内部处理好时区，但以防万一做个检查和转换
        if trade.open_date.tzinfo is None:
             open_date_utc = trade.open_date.replace(tzinfo=timezone.utc)
             print(f"【止损调试 {trade.id}】: 警告 - 交易开仓时间无时区，已强制转换为UTC: {open_date_utc}")
        else:
             # 确保转换为 UTC 标准时区进行比较
             open_date_utc = trade.open_date.astimezone(timezone.utc)
        # print(f"【止损调试 {trade.id}】: 交易开仓时间 (UTC): {open_date_utc}")

        # --- 3. 查找入场 K 线 ---
        # 找到严格在开仓时间之前的最后一根 K 线
        # 注意：如果开仓发生在K线的早期，这可能会选到前一根K线的数据，这通常是期望的行为（基于已完成K线计算指标）
        entry_candle = dataframe.loc[dataframe['date'] < open_date_utc]
        if entry_candle.empty:
            print(f"【止损调试 {trade.id}】: 错误 - 在 {open_date_utc} 之前找不到入场K线。数据可能不足或时间戳有问题。返回-1。")
            return -1.0 # 无法确定入场K线，使用默认止损

        # 获取入场前的最后一行数据
        entry_candle_row = entry_candle.iloc[-1]
        # print(f"【止损调试 {trade.id}】: 找到入场参考K线: 时间={entry_candle_row['date']}, 收盘价={entry_candle_row['close']}")

        # --- 4. 计算潜在止损价格 ---
        potential_stop_price = -1.0 # 初始化潜在止损价
        stop_col_name = "" # 用于日志记录的列名
        inf_tf = self.informative_timeframe # 简化后续使用

        if trade.is_short:
            # --- 4.1 做空止损: 基于前期高点 ---
            stop_col_name = f'inf_{inf_tf}_swing_high_{inf_tf}'
            # print(f"【止损调试 {trade.id}】: (做空) 尝试获取止损依据列: '{stop_col_name}'")
            if stop_col_name in entry_candle_row.index:
                raw_stop_val = entry_candle_row[stop_col_name]
                # print(f"【止损调试 {trade.id}】: 原始前期高点值: {raw_stop_val}")
                if not pd.isna(raw_stop_val):
                    # --- 添加缓冲 (可选但推荐) ---
                    # 做空止损应略高于前期高点，避免被精确扫描
                    buffer = trade.open_rate * 0.001 # 例如开仓价的 0.1% 作为缓冲
                    potential_stop_price = raw_stop_val + buffer
                    # print(f"【止损调试 {trade.id}】: 添加缓冲 ({buffer:.5f}) 后，潜在止损价: {potential_stop_price}")
                else:
                    print(f"【止损调试 {trade.id}】: 错误 - 前期高点值为 NaN。无法设置止损。")
                    potential_stop_price = -1.0 # 明确标记为无效
            else:
                print(f"【止损调试 {trade.id}】: 错误 - 数据框中缺少列 '{stop_col_name}'。无法设置止损。")
                potential_stop_price = -1.0
        else:
            # --- 4.2 做多止损: 基于前期低点 ---
            stop_col_name = f'inf_{inf_tf}_swing_low_{inf_tf}'
            # print(f"【止损调试 {trade.id}】: (做多) 尝试获取止损依据列: '{stop_col_name}'")
            if stop_col_name in entry_candle_row.index:
                raw_stop_val = entry_candle_row[stop_col_name]
                # print(f"【止损调试 {trade.id}】: 原始前期低点值: {raw_stop_val}")
                if not pd.isna(raw_stop_val):
                    # --- 添加缓冲 (可选但推荐) ---
                    # 做多止损应略低于前期低点
                    buffer = trade.open_rate * 0.001 # 例如开仓价的 0.1% 作为缓冲
                    potential_stop_price = raw_stop_val - buffer
                    # print(f"【止损调试 {trade.id}】: 添加缓冲 ({buffer:.5f}) 后，潜在止损价: {potential_stop_price}")
                    # 检查缓冲后是否变为负数或零
                    if potential_stop_price <= 0:
                         print(f"【止损调试 {trade.id}】: 警告 - 缓冲后止损价 <= 0 ({potential_stop_price})。视为无效。")
                         potential_stop_price = -1.0
                else:
                    print(f"【止损调试 {trade.id}】: 错误 - 前期低点值为 NaN。无法设置止损。")
                    potential_stop_price = -1.0
            else:
                print(f"【止损调试 {trade.id}】: 错误 - 数据框中缺少列 '{stop_col_name}'。无法设置止损。")
                potential_stop_price = -1.0

        # --- 5. 验证计算出的止损价 ---
        # print(f"【止损调试 {trade.id}】: 最终计算的潜在止损价: {potential_stop_price}")

        # 检查1: 止损价是否有效 (大于0)
        if potential_stop_price <= 0:
            print(f"【止损调试 {trade.id}】: 验证失败 - 潜在止损价无效 ({potential_stop_price} <= 0)。返回-1。")
            return -1.0

        # 检查2: 止损价是否与当前价格相同 (避免无效操作或被立即触发)
        # 使用 qtpylib.crossed_below 或 crossed_above 可能更稳健，但这里用简单比较
        # if abs(potential_stop_price - current_rate) < trade. FRACTURE? # 需要一个小的容忍度
        # 简化：如果完全相同，则可能无法设置，先跳过
        if potential_stop_price == current_rate:
             print(f"【止损调试 {trade.id}】: 验证警告 - 潜在止损价 ({potential_stop_price}) 等于当前价 ({current_rate})。暂时返回-1，等待价格变化。")
             return -1.0 # Freqtrade 通常在下一个 tick 会再次调用 custom_stoploss

        # 检查3: 止损价相对于当前价格的位置是否合理
        stop_loss_valid = False
        if not trade.is_short: # 做多
            if potential_stop_price < current_rate:
                stop_loss_valid = True
                # print(f"【止损调试 {trade.id}】: (做多) 验证通过 - 止损价 ({potential_stop_price}) < 当前价 ({current_rate})。")
            #else:
                # print(f"【止损调试 {trade.id}】: (做多) 验证失败 - 止损价 ({potential_stop_price}) >= 当前价 ({current_rate})。返回-1。")
        else: # 做空
            if potential_stop_price > current_rate:
                stop_loss_valid = True
                # print(f"【止损调试 {trade.id}】: (做空) 验证通过 - 止损价 ({potential_stop_price}) > 当前价 ({current_rate})。")
            #else:
                # print(f"【止损调试 {trade.id}】: (做空) 验证失败 - 止损价 ({potential_stop_price}) <= 当前价 ({current_rate})。返回-1。")

        # --- 6. 返回结果 ---
        if stop_loss_valid:
            print(f"【止损调试 {trade.id}】: >>> 成功设置自定义止损价: {potential_stop_price}")
            return potential_stop_price
            
        else:
            # 如果上面的逻辑没错，这里理论上不应该执行到，但作为保险
            print(f"【止损调试 {trade.id}】: 验证逻辑判断后仍未通过，返回-1。")
            return -1.0

# --- (可选) 策略优化器设置 ---
# class SmcFvgStrategy_Optimize(SmcFvgStrategy):
#     # 可以继承基础策略并覆盖参数范围用于优化
#     pass