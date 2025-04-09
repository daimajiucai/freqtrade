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
    """

    # --- 策略核心参数 ---
    INTERFACE_VERSION = 3  # Freqtrade接口版本

    # ROI Tabe (止盈设置) - 0分钟后止盈0.5%
    minimal_roi = {
        "0": 0.005  # 0.5%
    }

    # Stoploss (止损设置) - 使用自定义止损 `custom_stoploss`
    stoploss = -0.04 # 设置一个较大的值，因为我们将使用custom_stoploss

    # Trailing Stop (追踪止损) - 暂时不使用
    trailing_stop = False
    # trailing_stop_positive = 0.001
    # trailing_stop_positive_offset = 0.002
    # trailing_only_offset_is_reached = True

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
                                                prefix='inf_15m')  # 使用 .copy() 避免 SettingWithCopyWarning
            # 添加打印：检查 calculate_fvg 是否添加了列
            print(
                f"Columns in informative_df after calculate_fvg for {metadata['pair']}: {informative_df.columns.tolist()}")
            if 'inf_15m_bull_fvg_detected' not in informative_df.columns:
                print(
                    f"ERROR: 'inf_15m_bull_fvg_detected' column MISSING in informative_df after calculate_fvg for {metadata['pair']}!")

            informative_df = self.calculate_swing_high_low(informative_df.copy(), lookback=self.swing_lookback.value,
                                                           prefix='inf_15m')
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
            if 'inf_15m_bull_fvg_detected_15m' not in dataframe.columns:
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
        if 'inf_15m_bull_fvg_detected_15m' not in dataframe.columns:
            print(f"FATAL: 'inf_15m_bull_fvg_detected' is STILL missing before access for {metadata['pair']} at {dataframe['date'].iloc[-1]}. Returning original dataframe.")
            # 可以选择填充默认值或直接返回，避免 KeyErorr
            # dataframe['inf_15m_bull_fvg_detected'] = False # 填充默认值示例
            # dataframe['inf_15m_fvg_bull_top'] = np.nan
            # dataframe['inf_15m_swing_midpoint'] = np.nan
            # return dataframe # 或者直接返回
            # 目前选择直接返回，因为缺少关键列无法继续计算
            return dataframe
        if 'inf_15m_fvg_bull_top_15m' not in dataframe.columns:
             print(f"FATAL: 'inf_15m_fvg_bull_top_15m' is missing...")
             return dataframe
        if 'inf_15m_swing_midpoint_15m' not in dataframe.columns:
             print(f"FATAL: 'inf_15m_swing_midpoint_15m' is missing...")
             return dataframe
        if 'inf_15m_bear_fvg_detected_15m' not in dataframe.columns: # 检查看跌FVG的列
            print(f"FATAL: 'inf_15m_bear_fvg_detected_15m' is missing...")
            return dataframe
        if 'inf_15m_fvg_bear_bottom_15m' not in dataframe.columns:
            print(f"FATAL: 'inf_15m_fvg_bear_bottom_15m' is missing...")
            return dataframe

        # --- 5. 判断 15m FVG 的有效性 (折扣区/溢价区) ---
        dataframe['inf_15m_bull_fvg_valid_15m'] = (
            (dataframe['inf_15m_bull_fvg_detected_15m']) &
            (dataframe['inf_15m_fvg_bull_top_15m'] < dataframe['inf_15m_swing_midpoint_15m']) # FVG顶部低于50%
        )
        dataframe['inf_15m_bear_fvg_valid_15m'] = (
            (dataframe['inf_15m_bear_fvg_detected_15m']) &
            (dataframe['inf_15m_fvg_bear_bottom_15m'] > dataframe['inf_15m_swing_midpoint_15m']) # FVG底部高于50%
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
        dataframe.loc[
            (
                (dataframe['inf_15m_bull_fvg_valid_15m'] == True) &
                (dataframe['low'] <= dataframe['inf_15m_fvg_bull_top_15m']) & # 价格触及或进入15m FVG上沿
                # (dataframe['low'] >= dataframe['inf_15m_fvg_bull_bottom']) & # (可选) 价格仍在FVG内部
                 (dataframe['volume'] > 0) # 基础的成交量过滤
            ),
            ['enter_long', 'enter_tag']] = (1, f'15m_bull_fvg_entry') # 设置信号和标签

        # 做空条件 (Short Entry):
        # 1. 检测到有效的 15m 看跌 FVG (inf_15m_bear_fvg_valid == True)
        # 2. 当前 1m K线的最高价触碰或进入了该 15m FVG 的下边界 (high >= inf_15m_fvg_bear_bottom)
        # 3. (可选) 增加其他过滤条件
        dataframe.loc[
            (
                (dataframe['inf_15m_bear_fvg_valid_15m'] == True) &
                (dataframe['high'] >= dataframe['inf_15m_fvg_bear_bottom_15m']) & # 价格触及或进入15m FVG下沿
                # (dataframe['high'] <= dataframe['inf_15m_fvg_bear_top']) & # (可选) 价格仍在FVG内部
                (dataframe['volume'] > 0) # 基础的成交量过滤
            ),
            ['enter_short', 'enter_tag']] = (1, f'15m_bear_fvg_entry') # 设置信号和标签

        # --- 待实现: 防止在失效FVG处入场 ---
        # TODO: 确保入场信号不会在已经被标记为失效的FVG上触发

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

    # --- 自定义止损 ---
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        """
        自定义止损逻辑，基于入场时的前期高/低点

        Args:
            pair (str): 交易对
            trade (Trade): Freqtrade 的 Trade 对象
            current_time (datetime): 当前时间
            current_rate (float): 当前价格
            current_profit (float): 当前利润率
            **kwargs: 其他可能需要的参数

        Returns:
            float: 需要设置的绝对止损价格. 返回 -1 表示使用配置文件中的默认止损
        """
        # 获取入场时的DataFrame行信息 (注意数据可能变化，需要谨慎处理)
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        # 找到与交易开仓时间最接近的K线索引 (需要开仓时间是UTC)
        # trade.open_date 是 aware datetime object (UTC)
        entry_candle = dataframe.loc[dataframe['date'] < trade.open_date]
        if entry_candle.empty:
            # print(f"Warning: Could not find entry candle for trade {trade.id} opened at {trade.open_date}")
            return -1 # 无法确定入场K线，使用默认止损

        entry_candle_row = entry_candle.iloc[-1]

        # 获取入场时使用的 15m Swing High/Low (需要确保这些列存在)
        # 注意：这里的 swing high/low 是基于 entry_candle 之前的数据计算的
        stop_price = -1 # 默认值

        if trade.is_short:
            # 做空止损: 入场时的前期高点 (inf_15m_swing_high)
            swing_high_col = 'inf_15m_swing_high_15m'
            if swing_high_col in entry_candle_row.index and not pd.isna(entry_candle_row[swing_high_col]):
                stop_price = entry_candle_row[swing_high_col]
                # print(f"Trade {trade.id} (Short): Setting stoploss to Swing High {stop_price} based on candle at {entry_candle_row['date']}")
            else:
                # print(f"Warning: Swing high not available for trade {trade.id} at entry candle {entry_candle_row['date']}")
                pass # stop_price 保持 -1
        else:
            # 做多止损: 入场时的前期低点 (inf_15m_swing_low)
            swing_low_col = 'inf_15m_swing_low_15m'
            if swing_low_col in entry_candle_row.index and not pd.isna(entry_candle_row[swing_low_col]):
                stop_price = entry_candle_row[swing_low_col]
                # print(f"Trade {trade.id} (Long): Setting stoploss to Swing Low {stop_price} based on candle at {entry_candle_row['date']}")
            else:
                # print(f"Warning: Swing low not available for trade {trade.id} at entry candle {entry_candle_row['date']}")
                pass # stop_price 保持 -1

        # 确保止损价格不是无效值 (e.g., 0 or NaN) 且与当前价格有一定距离以避免立即触发
        if stop_price > 0 and stop_price != current_rate:
             # 对于多单，止损价必须低于当前价；对于空单，止损价必须高于当前价
            if (not trade.is_short and stop_price < current_rate) or \
               (trade.is_short and stop_price > current_rate):
                return stop_price
            else:
                # print(f"Warning: Calculated stop price {stop_price} is invalid relative to current rate {current_rate} for trade {trade.id}. Using default stoploss.")
                return -1 # 返回-1使用默认止损
        else:
            # print(f"Warning: Invalid stop price {stop_price} calculated for trade {trade.id}. Using default stoploss.")
            return -1 # 返回-1使用默认止损

# --- (可选) 策略优化器设置 ---
# class SmcFvgStrategy_Optimize(SmcFvgStrategy):
#     # 可以继承基础策略并覆盖参数范围用于优化
#     pass