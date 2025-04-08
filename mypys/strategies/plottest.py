from freqtrade.strategy import IStrategy, IntParameter
from pandas import DataFrame
import numpy as np
import talib.abstract as ta

# Freqtrade 接口版本 - 不要轻易修改
INTERFACE_VERSION = 3

class FvgPlotStrategy(IStrategy):
    """
    一个仅用于绘制公允价值缺口 (FVG) 的 Freqtrade 策略 Demo。
    - 绿色区域表示看涨 FVG (Bullish FVG)
    - 红色区域表示看跌 FVG (Bearish FVG)
    FVG 特性:
    1. 当价格触及 FVG 区域时，该 FVG 停止延申 (变为 NaN)。
       - 看涨 FVG: low <= FVG bottom
       - 看跌 FVG: high >= FVG top
    2. FVG 最多延申 50 根 K 线，如果未被触及，50 根后自动失效 (变为 NaN)。
    本策略不进行任何买卖操作。
    """

    # -- 策略核心配置 --

    # 时间周期 (e.g., '1h', '4h', '1d', '15m')
    timeframe = '1m'

    # 必要的 ROI 表 (即使不交易也要有)
    # key 是持仓时间(分钟), value 是期望利润率
    # {0: 1} 表示 ROI 永不触发卖出 (利润目标 100%)
    minimal_roi = {
        "0": 1.0
    }

    # 必要的止损设置 (即使不交易也要有)
    # -0.99 表示几乎不止损 (99% 亏损)
    stoploss = -0.99

    # 可选: 是否使用追踪止损
    trailing_stop = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.02
    # trailing_only_offset_is_reached = False

    # 运行策略需要的最少蜡烛数据量
    # FVG 需要当前蜡烛、前一根、再前一根 (共3根) 来判断
    # 所以至少需要 2 根历史蜡烛 + 1 根当前蜡烛 = 3
    startup_candle_count: int = 55 # 需要足够的数据来计算年龄 (50 + FVG形成所需3根)

     # --- FVG 超时设置 ---
    fvg_timeout_limit: int = 50

    # --- 绘图配置 ---
    plot_config = {
        'main_plot': {
            # --- 看涨 FVG ---
            # 绘制 FVG 的上边界线 (candle[-2] 的 low)
            'bullish_fvg_top': {
                'name': 'Bullish FVG Top',
                'type': 'line',
                'color': 'rgba(0, 255, 0, 0.5)', # 半透明绿线
                'plotly': {
                    'fill': 'tonexty',  # 填充到下一条线 (bullish_fvg_bottom)
                    'fillcolor': 'rgba(0, 255, 0, 0.15)', # 浅绿色填充区域
                }
            },
            # 绘制 FVG 的下边界线 (candle[0] 的 high)
            'bullish_fvg_bottom': {
                'name': 'Bullish FVG Bottom',
                'type': 'line',
                'color': 'rgba(0, 0, 0, 0)', # 完全透明，不显示这条线本身
                # 'plotly': {'visible': 'legendonly'} # 或者只在图例显示
            },

            # --- 看跌 FVG ---
             # 绘制 FVG 的上边界线 (candle[0] 的 low)
            'bearish_fvg_top': {
                'name': 'Bearish FVG Top',
                'type': 'line',
                'color': 'rgba(255, 0, 0, 0.5)', # 半透明红线
                 'plotly': {
                    'fill': 'tonexty', # 填充到下一条线 (bearish_fvg_bottom)
                    'fillcolor': 'rgba(255, 0, 0, 0.15)', # 浅红色填充区域
                }
            },
            # 绘制 FVG 的下边界线 (candle[-2] 的 high)
            'bearish_fvg_bottom': {
                'name': 'Bearish FVG Bottom',
                'type': 'line',
                'color': 'rgba(0, 0, 0, 0)', # 完全透明
                # 'plotly': {'visible': 'legendonly'}
            },
        },
        'subplots': {} # 没有子图
    }

    # --- 指标计算 ---
    # def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    #     """
    #     计算指标，包括 FVG 的上下边界。

    #     FVG 定义 (基于3根蜡烛):
    #     - Index 0: 当前蜡烛
    #     - Index -1: 前一根蜡烛
    #     - Index -2: 再前一根蜡烛

    #     看涨 FVG (Bullish FVG):
    #       - 条件: candle[-2].low > candle[0].high
    #       - 区域: 从 candle[0].high 到 candle[-2].low

    #     看跌 FVG (Bearish FVG):
    #       - 条件: candle[-2].high < candle[0].low
    #       - 区域: 从 candle[-2].high 到 candle[0].low
    #     """

    #     # 获取前一根和再前一根蜡烛的数据
    #     dataframe['prev_high'] = dataframe['high'].shift(1)
    #     dataframe['prev_low'] = dataframe['low'].shift(1)
    #     dataframe['prev_prev_high'] = dataframe['high'].shift(2)
    #     dataframe['prev_prev_low'] = dataframe['low'].shift(2)

    #     # --- 计算看涨 FVG ---
    #     # 条件: 再前一根的 low > 当前的 high
    #     bullish_fvg_condition = (dataframe['prev_prev_low'] > dataframe['high'])

    #     # 如果满足条件，记录 FVG 的上下边界，否则为 NaN
    #     dataframe['bullish_fvg_top'] = np.where(
    #         bullish_fvg_condition,
    #         dataframe['prev_prev_low'], # FVG 上边界是 candle[-2] 的 low
    #         np.nan
    #     )
    #     dataframe['bullish_fvg_bottom'] = np.where(
    #         bullish_fvg_condition,
    #         dataframe['high'],           # FVG 下边界是 candle[0] 的 high
    #         np.nan
    #     )

    #     # --- 计算看跌 FVG ---
    #     # 条件: 再前一根的 high < 当前的 low
    #     bearish_fvg_condition = (dataframe['prev_prev_high'] < dataframe['low'])

    #     # 如果满足条件，记录 FVG 的上下边界，否则为 NaN
    #     dataframe['bearish_fvg_top'] = np.where(
    #         bearish_fvg_condition,
    #         dataframe['low'],           # FVG 上边界是 candle[0] 的 low
    #         np.nan
    #     )
    #     dataframe['bearish_fvg_bottom'] = np.where(
    #         bearish_fvg_condition,
    #         dataframe['prev_prev_high'], # FVG 下边界是 candle[-2] 的 high
    #         np.nan
    #     )

    #     # 清理临时列（可选）
    #     # dataframe.drop(['prev_high', 'prev_low', 'prev_prev_high', 'prev_prev_low'], axis=1, inplace=True)

    #     print(dataframe[['date', 'open', 'high', 'low', 'close', 'bullish_fvg_top', 'bullish_fvg_bottom', 'bearish_fvg_top', 'bearish_fvg_bottom']].tail(10)) # Debugging

    #     return dataframe
    """
    一个仅用于绘制公允价值缺口 (FVG) 的 Freqtrade 策略 Demo。
    - 绿色区域表示看涨 FVG (Bullish FVG)
    - 红色区域表示看跌 FVG (Bearish FVG)
    FVG 特性:
    1. 当价格触及 FVG 区域时，该 FVG 停止延申 (变为 NaN)。
       - 看涨 FVG: low <= FVG bottom
       - 看跌 FVG: high >= FVG top
    2. FVG 最多延申 50 根 K 线，如果未被触及，50 根后自动失效 (变为 NaN)。
    本策略不进行任何买卖操作。
    """
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        计算 FVG 指标，并应用触碰失效和超时失效逻辑。
        """
        # Freqtrade 推荐直接修改传入的 dataframe
        # df = dataframe.copy() # 如果你绝对想用副本，记得最后处理 KeyErorr 或同步回去

        # 1. 获取必要的历史数据 (基于修正后的 FVG 定义)
        # shift(1) 是前一根蜡烛 (candle -1)
        # shift(2) 是前两根蜡烛 (candle -2), 即 FVG 定义中的 "第一根蜡烛"
        dataframe['low_shift2'] = dataframe['low'].shift(2)
        dataframe['high_shift2'] = dataframe['high'].shift(2)

        # --- 计算看涨 FVG (BISI) ---
        # 条件: 当前 low > 前两根的 high (low[0] > high[-2])
        bullish_fvg_condition_raw = (dataframe['low'] > dataframe['high_shift2'])
        # FVG 顶部: 当前 low (low[0])
        # FVG 底部: 前两根 high (high[-2])
        dataframe['bullish_fvg_top_raw'] = np.where(bullish_fvg_condition_raw, dataframe['low'], np.nan)
        dataframe['bullish_fvg_bottom_raw'] = np.where(bullish_fvg_condition_raw, dataframe['high_shift2'], np.nan)

        # --- 计算看跌 FVG (SIBI) ---
        # 条件: 当前 high < 前两根的 low (high[0] < low[-2])
        bearish_fvg_condition_raw = (dataframe['high'] < dataframe['low_shift2'])
        # FVG 顶部: 前两根 low (low[-2])
        # FVG 底部: 当前 high (high[0])
        dataframe['bearish_fvg_top_raw'] = np.where(bearish_fvg_condition_raw, dataframe['low_shift2'], np.nan)
        dataframe['bearish_fvg_bottom_raw'] = np.where(bearish_fvg_condition_raw, dataframe['high'], np.nan)

        # 3. 识别每个 FVG 区块的开始
        dataframe['bullish_fvg_start'] = dataframe['bullish_fvg_top_raw'].notna() & dataframe['bullish_fvg_top_raw'].shift(1).isna() # 更精确的开始点
        dataframe['bearish_fvg_start'] = dataframe['bearish_fvg_top_raw'].notna() & dataframe['bearish_fvg_top_raw'].shift(1).isna() # 更精确的开始点


        # 4. 计算 FVG 年龄 (持续时间)
        bullish_group_id = dataframe['bullish_fvg_start'].cumsum()
        bearish_group_id = dataframe['bearish_fvg_start'].cumsum()

        dataframe['bullish_fvg_age'] = dataframe.groupby(bullish_group_id).cumcount() # cumcount 从 0 开始，代表形成后的 K 线数
        dataframe['bearish_fvg_age'] = dataframe.groupby(bearish_group_id).cumcount()

        # 将非 FVG 周期的年龄重置 (可以用 ffill 后的结果判断)

        # 5. 第一次向前填充 (ffill) - 获取潜在的 FVG 区域
        dataframe['bullish_fvg_top_ffill'] = dataframe['bullish_fvg_top_raw'].ffill()
        dataframe['bullish_fvg_bottom_ffill'] = dataframe['bullish_fvg_bottom_raw'].ffill()
        dataframe['bearish_fvg_top_ffill'] = dataframe['bearish_fvg_top_raw'].ffill()
        dataframe['bearish_fvg_bottom_ffill'] = dataframe['bearish_fvg_bottom_raw'].ffill()

        # 修正年龄计算：只有在 ffill 后 FVG 确实存在时，年龄才有意义
        dataframe.loc[dataframe['bullish_fvg_top_ffill'].isna(), 'bullish_fvg_age'] = np.nan
        dataframe.loc[dataframe['bearish_fvg_top_ffill'].isna(), 'bearish_fvg_age'] = np.nan

        # 6. 计算失效条件
        # 条件1: 价格触碰 (Mitigation)
        # 看涨 FVG (BISI) 被触碰: low <= FVG 底部 (high[-2])
        bullish_mitigated = (dataframe['low'] <= dataframe['bullish_fvg_bottom_ffill'])
        # 看跌 FVG (SIBI) 被触碰: high >= FVG 顶部 (low[-2])
        bearish_mitigated = (dataframe['high'] >= dataframe['bearish_fvg_top_ffill'])

        # 条件2: 超时 (Timeout) - 年龄从0开始，所以比较 >= limit - 1 (第0根到第49根共50根)
        # 或者让 cumcount+1，然后比较 >= limit
        # 这里用 cumcount() 从0开始计数，所以第50根 K 线时 age 是 49
        bullish_timed_out = (dataframe['bullish_fvg_age'] >= self.fvg_timeout_limit)
        bearish_timed_out = (dataframe['bearish_fvg_age'] >= self.fvg_timeout_limit)

        # 合并失效条件
        invalidate_bullish = bullish_mitigated | bullish_timed_out
        invalidate_bearish = bearish_mitigated | bearish_timed_out

        # 7. 应用失效逻辑 (使用中间列，避免影响下一步的 ffill)
        dataframe['bullish_fvg_top_intermediate'] = dataframe['bullish_fvg_top_ffill']
        dataframe['bullish_fvg_bottom_intermediate'] = dataframe['bullish_fvg_bottom_ffill']
        dataframe['bearish_fvg_top_intermediate'] = dataframe['bearish_fvg_top_ffill']
        dataframe['bearish_fvg_bottom_intermediate'] = dataframe['bearish_fvg_bottom_ffill']

        # 找到需要失效的行的索引
        # 失效应该从 *下一根* K线开始，或者说，失效条件满足的 *当前* K线是最后有效的一根
        # 如果希望 FVG 在触碰/超时的 *那根 K 线* 就消失，需要 shift 失效条件
        # 按照你的描述 "当价格触及 FVG 区域时，该 FVG 停止延申 (变为 NaN)"，意味着触及 K 线 *开始* 变 NaN
        # 使用 .loc[index, column] 更安全
        dataframe.loc[invalidate_bullish, ['bullish_fvg_top_intermediate', 'bullish_fvg_bottom_intermediate']] = np.nan
        dataframe.loc[invalidate_bearish, ['bearish_fvg_top_intermediate', 'bearish_fvg_bottom_intermediate']] = np.nan


        # 8. 第二次向前填充 (Final ffill) - 这步是关键
        # 再次 ffill 会将 NaN 点之后的值也填上，这不是我们想要的
        # 我们需要在失效点之后保持 NaN
        # 正确的方法是：使用原始 FVG 值，结合未失效的 ffill 值

        # 方法一： 使用 where 保留原始 NaN
        # dataframe['bullish_fvg_top'] = np.where(invalidate_bullish.shift(1).fillna(False), np.nan, dataframe['bullish_fvg_top_raw'].ffill(limit=self.fvg_timeout_limit))
        # dataframe['bullish_fvg_bottom'] = np.where(invalidate_bullish.shift(1).fillna(False), np.nan, dataframe['bullish_fvg_bottom_raw'].ffill(limit=self.fvg_timeout_limit))
        # dataframe['bearish_fvg_top'] = np.where(invalidate_bearish.shift(1).fillna(False), np.nan, dataframe['bearish_fvg_top_raw'].ffill(limit=self.fvg_timeout_limit))
        # dataframe['bearish_fvg_bottom'] = np.where(invalidate_bearish.shift(1).fillna(False), np.nan, dataframe['bearish_fvg_bottom_raw'].ffill(limit=self.fvg_timeout_limit))

        # 方法二： 使用你原来的双 ffill 逻辑 (更接近你的原始代码)
        # 这确保 FVG 从形成点延申，直到失效点 *之前* 的那根 K 线。
        # 失效点及其之后，值将保持为 NaN (因为 intermediate 在失效点是 NaN，ffill 不会填充它)
        dataframe['bullish_fvg_top'] = dataframe['bullish_fvg_top_intermediate'].ffill()
        dataframe['bullish_fvg_bottom'] = dataframe['bullish_fvg_bottom_intermediate'].ffill()
        dataframe['bearish_fvg_top'] = dataframe['bearish_fvg_top_intermediate'].ffill()
        dataframe['bearish_fvg_bottom'] = dataframe['bearish_fvg_bottom_intermediate'].ffill()

        # 再进行一次检查，确保超时的 FVG 确实被清除了
        # 因为 ffill 可能会跨越超时的界限，如果中间没有触碰的话
        dataframe.loc[dataframe['bullish_fvg_age'] >= self.fvg_timeout_limit, ['bullish_fvg_top', 'bullish_fvg_bottom']] = np.nan
        dataframe.loc[dataframe['bearish_fvg_age'] >= self.fvg_timeout_limit, ['bearish_fvg_top', 'bearish_fvg_bottom']] = np.nan


        # # --- 清理中间列 (可选) ---
        # dataframe.drop(columns=[
        #     'low_shift2', 'high_shift2',
        #     'bullish_fvg_top_raw', 'bullish_fvg_bottom_raw',
        #     'bearish_fvg_top_raw', 'bearish_fvg_bottom_raw',
        #     'bullish_fvg_start', 'bearish_fvg_start',
        #     'bullish_fvg_age', 'bearish_fvg_age',
        #     'bullish_fvg_top_ffill', 'bullish_fvg_bottom_ffill',
        #     'bearish_fvg_top_ffill', 'bearish_fvg_bottom_ffill',
        #     'bullish_fvg_top_intermediate', 'bullish_fvg_bottom_intermediate',
        #     'bearish_fvg_top_intermediate', 'bearish_fvg_bottom_intermediate'
        # ], inplace=True, errors='ignore')

        # --- 调试打印 (现在使用 dataframe) ---
        print("Bullish FVG Data:")
        print(dataframe[['date', 'low', 'high', 'bullish_fvg_top', 'bullish_fvg_bottom']].tail(60))
        print("\nBearish FVG Data:")
        print(dataframe[['date', 'low', 'high', 'bearish_fvg_top', 'bearish_fvg_bottom']].tail(60))

        return dataframe
    # --- 买入信号 (空实现) ---
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        定义买入信号逻辑。
        由于本策略只绘图，这里不产生任何信号。
        """
        # 确保这些列存在，且全为 0
        dataframe.loc[:, 'enter_long'] = 0
        dataframe.loc[:, 'enter_short'] = 0
        return dataframe

    # --- 卖出信号 (空实现) ---
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        定义卖出信号逻辑。
        由于本策略只绘图，这里不产生任何信号。
        """
        # 确保这些列存在，且全为 0
        dataframe.loc[:, 'exit_long'] = 0
        dataframe.loc[:, 'exit_short'] = 0
        return dataframe

# --- Freqtrade 需要的策略参数范围 (可选, 用于超参数优化) ---
# class FvgPlotStrategy_Optimize(FvgPlotStrategy):
#     pass # 这里可以添加用于优化的参数，但对于纯绘图策略意义不大