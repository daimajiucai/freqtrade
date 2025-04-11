# --- 导入必要的库 ---
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import pandas as pd
from pandas import DataFrame
# --- !!! 导入 informative 装饰器 !!! ---
from freqtrade.strategy import informative
from freqtrade.strategy import IStrategy, IntParameter


# --- FVG 数据结构 (示例) ---
from datetime import datetime, timedelta
class FVG:
    def __init__(self, formation_time, direction, top, bottom, timeframe='15m'):
        self.formation_time: datetime = formation_time # FVG形成时的15min K线开始时间
        self.timeframe: str = timeframe
        self.direction: str = direction # 'bullish' 或 'bearish'
        self.top: float = top
        self.bottom: float = bottom
        self.midpoint: float = (top + bottom) / 2
        self.status: str = 'active' # 状态: 'active', 'touched_mid', 'mitigated', 'expired' (可选)
        self.last_updated: datetime = formation_time # 最后更新状态的时间 (1min K线时间)
        self.id: str = f"{timeframe}_{direction}_{formation_time.isoformat()}" # 唯一ID

    def __repr__(self):
        return (f"FVG(id={self.id}, status={self.status}, "
                f"top={self.top:.5f}, bottom={self.bottom:.5f})")

    def update_status(self, current_time: datetime, current_price_high: float, current_price_low: float):
        """根据当前K线价格更新FVG状态"""
        self.last_updated = current_time

        if self.status == 'mitigated' or self.status == 'expired':
            return # 最终状态，不再更新

        price_crossed_bottom = current_price_low < self.bottom
        price_crossed_top = current_price_high > self.top
        price_touched_mid = (current_price_low <= self.midpoint <= current_price_high)

        if self.direction == 'bullish': # 看涨FVG (价格预期从下方进入)
            if price_crossed_bottom:
                self.status = 'mitigated' # 价格跌破底部，FVG被完全填充/失效
            elif self.status == 'active' and (current_price_low <= self.top): # 首次触及
                if price_touched_mid:
                   self.status = 'touched_mid'
                # 可以添加更细致的状态，如 'entered'
            elif self.status == 'touched_mid' and price_crossed_bottom: # 如果之前只触及中点，现在完全跌破
                 self.status = 'mitigated'
            # 可以添加过期逻辑，例如 FVG 形成后 N 根 K 线仍未被触及则 expired

        elif self.direction == 'bearish': # 看跌FVG (价格预期从上方进入)
            if price_crossed_top:
                self.status = 'mitigated' # 价格涨破顶部，FVG被完全填充/失效
            elif self.status == 'active' and (current_price_high >= self.bottom): # 首次触及
                 if price_touched_mid:
                    self.status = 'touched_mid'
            elif self.status == 'touched_mid' and price_crossed_top:
                 self.status = 'mitigated'
            # 可以添加过期逻辑

# --- 策略类定义 ---
class SmcFvgStrategy0412(IStrategy):
    """
    计算不对称摆动高点 (Swing High - SH) 和摆动低点 (Swing Low - SL) 的策略框架。
    - 摆动点左侧需要 N 根 K 线满足条件。
    - 摆动点右侧需要 M 根 K 线满足条件。
    - N 和 M 是可配置的超参数。
    """
    # --- 策略核心参数 ---
    timeframe = '1m'  # 示例时间框架

    # 定义需要的信息数据对和时间框架
    informative_timeframe = '15m'

    # --- Freqtrade 必须的配置 ---
    minimal_roi = {"0": 100}
    stoploss = -0.99
    trailing_stop = False

    # --- 超参数定义 ---
    # N: 摆动点左侧需要比较的 K 线数量
    swing_left_n = IntParameter(low=1, high=10, default=8, space="indicator", optimize=True, load=True)
    # M: 摆动点右侧需要比较的 K 线数量 (决定确认延迟)
    swing_right_m = IntParameter(low=1, high=10, default=2, space="indicator", optimize=True, load=True)

    # 启动时需要的最少 K 线数量
    # 窗口大小 = n (左侧) + 1 (自身) + m (右侧)
    # 确认需要 m 根 K 线之后发生
    # 所以至少需要 n + 1 + m 根 K 线的数据
    # 使用参数的最大值来计算，再加点缓冲
    # (使用 property 确保它总是基于最新的超参数值)

    # --- FVG 管理 ---
    # 使用类属性存储 FVG 列表。注意：这在回测和实时中需要小心处理状态一致性
    # 更好的方式可能是将 FVG 列表存储在 metadata 或 cache 中，但这更复杂
    # 我们先用简单的类属性示例
    fvg_list: list[FVG] = []


    @informative('15m') # 使用装饰器
    def populate_indicators_15m(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """计算 15 分钟时间框架的指标"""
        # dataframe 参数已经是 15 分钟的 K 线数据了
        print(f"计算 15m 指标 for {metadata['pair']} @ {dataframe['date'].iloc[-1]}")
        # 在这里计算 15min FVG 或其他指标
        # ... 计算 FVG 逻辑 ...
        # --- 1. 计算 15min FVG 并更新 FVG 列表 ---
        have_new=self.update_fvg_list(dataframe)
        # FVG 计算结果会自动合并回主时间框架 (1m) 的 DataFrame
        # Freqtrade 会处理时间对齐和填充 (通常是向前填充 ffill)
        # TODO:判断各种情况，更新标记，然后更新FVG对象状态
        dataframe['is_bull_fvg'] = ...  # 当前k线是否涉及看涨fvg（与前面形成的fvg的关系）
        dataframe['is_bear_fvg'] = ...  # 当前k线是否涉及看跌fvg（与前面形成的fvg的关系）
        #若有关系，无非以下几种
        dataframe['is_in_fvg'] = ... # 结算价格是否在fvg内部
        dataframe['is_out_fvg'] = ...# 结算价格是否穿过fvg（实体刺穿）
        dataframe['is_pushback_fvg'] = ...#结算价格被推回fvg外部（价格反转）
        #TODO:根据上述判断情况，更新FVG对象状态


        return dataframe[['date', 'fvg_15m_bull_top', 'fvg_15m_bull_bottom']] # 只返回需要的列

    @property
    def startup_candle_count(self) -> int:
        n_max = self.swing_left_n.high if self.config['runmode'] in ('hyperopt',) else self.swing_left_n.value
        m_max = self.swing_right_m.high if self.config['runmode'] in ('hyperopt',) else self.swing_right_m.value
        # 在回测/交易模式下，使用实际值；在优化模式下，使用最大值确保覆盖所有组合
        required_candles = n_max + 1 + m_max
        return required_candles + 5  # 加 5 作为缓冲

    # --- 指标计算 ---
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        计算并添加不对称摆动高点和低点指标到 DataFrame
        """
        # 从超参数获取 N 和 M 的值
        n = self.swing_left_n.value
        m = self.swing_right_m.value

        print(f"计算不对称摆动点指标 - {metadata['pair']} - 使用 N(左) = {n}, M(右) = {m}")

        if not isinstance(n, int) or n <= 0:
            raise ValueError(f"摆动点左侧回顾期 N ('swing_left_n'={n}) 必须是正整数")
        if not isinstance(m, int) or m <= 0:
            raise ValueError(f"摆动点右侧回顾期 M ('swing_right_m'={m}) 必须是正整数")

        # 确认延迟：确认发生在摆动点之后的第 M 根 K 线完成时
        # 因此，我们在当前索引 i 处确认索引 i-m 处的点是否为摆动点
        delay = m

        # --- 计算摆动高点 (Swing High - SH) ---
        # 条件: high[i-m] > high[i-m-k] 对于 k=1..n (左侧)
        #   AND high[i-m] > high[i-m+k] 对于 k=1..m (右侧)
        # 使用 shift(delay) 来获取索引 i-m 处的值
        target_high = dataframe['high'].shift(delay)

        # 检查左侧 N 根 K 线 (索引从 i-m-1 到 i-m-n)
        is_higher_left = pd.Series(True, index=dataframe.index)
        for k in range(1, n + 1):
            # shift(delay + k) 对应于索引 i-m-k
            is_higher_left &= (target_high > dataframe['high'].shift(delay + k))

        # 检查右侧 M 根 K 线 (索引从 i-m+1 到 i)
        is_higher_right = pd.Series(True, index=dataframe.index)
        for k in range(1, m + 1):
            # shift(delay - k) 对应于索引 i-m+k
            is_higher_right &= (target_high > dataframe['high'].shift(delay - k))

        # 综合条件：当左右条件都满足时，索引 i-m 处是一个摆动高点
        is_swing_high = is_higher_left & is_higher_right

        # 记录确认的摆动高点价格。
        # 这个价格是 m 根 K 线之前的价格 (即 dataframe['high'].shift(delay))
        # 这个值只在确认发生的 K 线索引 i 上非 NaN
        col_name_sh_point = f'sh_price_{n}_{m}'  # 列名包含 n 和 m 值
        dataframe[col_name_sh_point] = np.where(
            is_swing_high,
            target_high,  # 记录当时的高点价格
            np.nan
        )

        # --- 计算摆动低点 (Swing Low - SL) ---
        # 条件: low[i-m] < low[i-m-k] 对于 k=1..n (左侧)
        #   AND low[i-m] < low[i-m+k] 对于 k=1..m (右侧)
        target_low = dataframe['low'].shift(delay)

        # 检查左侧 N 根 K 线
        is_lower_left = pd.Series(True, index=dataframe.index)
        for k in range(1, n + 1):
            is_lower_left &= (target_low < dataframe['low'].shift(delay + k))

        # 检查右侧 M 根 K 线
        is_lower_right = pd.Series(True, index=dataframe.index)
        for k in range(1, m + 1):
            is_lower_right &= (target_low < dataframe['low'].shift(delay - k))

        # 综合条件：当左右条件都满足时，索引 i-m 处是一个摆动低点
        is_swing_low = is_lower_left & is_lower_right

        # 记录确认的摆动低点价格。
        # 这个价格是 m 根 K 线之前的价格 (即 dataframe['low'].shift(delay))
        col_name_sl_point = f'sl_price_{n}_{m}'  # 列名包含 n 和 m 值
        dataframe[col_name_sl_point] = np.where(
            is_swing_low,
            target_low,  # 记录当时的低点价格
            np.nan
        )

        # --- 创建 "最近已确认" 摆动点列 ---
        # 使用 ffill() 向前填充
        col_name_last_sh = f'last_sh_{n}_{m}'
        col_name_last_sl = f'last_sl_{n}_{m}'
        dataframe[col_name_last_sh] = dataframe[col_name_sh_point].ffill()
        dataframe[col_name_last_sl] = dataframe[col_name_sl_point].ffill()

        # 打印最后一行的数据，用于验证
        if not dataframe.empty:
            last_idx = dataframe.index[-1]
            last_date = dataframe['date'].iloc[-1]
            last_sh_val = dataframe[col_name_last_sh].iloc[-1]
            last_sl_val = dataframe[col_name_last_sl].iloc[-1]
            sh_confirm_idx = dataframe[col_name_sh_point].last_valid_index()
            sl_confirm_idx = dataframe[col_name_sl_point].last_valid_index()

            print(f"  最新 K 线时间: {last_date} (索引 {last_idx})")
            print(
                f"  最近确认 SH ({n},{m}): {last_sh_val:.5f} (在索引 {sh_confirm_idx if sh_confirm_idx is not None else 'N/A'} 确认)")
            print(
                f"  最近确认 SL ({n},{m}): {last_sl_val:.5f} (在索引 {sl_confirm_idx if sl_confirm_idx is not None else 'N/A'} 确认)")



        return dataframe

    def update_fvg_list(self, df_15m: DataFrame):
        """检测新的 15min FVG 并添加到 self.fvg_list"""
        if df_15m.empty or len(df_15m) < 3:
            return # 需要至少3根K线来识别FVG

        # 计算 FVG 条件 (同之前的代码)
        # 注意 shift 的用法，我们看 k 线 i, i-1, i-2
        df_15m['low_i'] = df_15m['low']
        df_15m['high_i'] = df_15m['high']
        df_15m['low_i-1'] = df_15m['low'].shift(1)
        df_15m['high_i-1'] = df_15m['high'].shift(1)
        df_15m['low_i-2'] = df_15m['low'].shift(2)
        df_15m['high_i-2'] = df_15m['high'].shift(2)

        # 看涨 FVG: low[i] > high[i-2]
        # FVG 区域: (high[i-2], low[i-1])  <-- 这是一个常见的定义，请确认是否符合你的要求
        # 另一种定义: (high[i-2], low[i])
        bull_cond = df_15m['low_i'] > df_15m['high_i-2']
        # 看跌 FVG: high[i] < low[i-2]
        # FVG 区域: (high[i-1], low[i-2]) <-- 常见定义
        # 另一种定义: (high[i], low[i-2])
        bear_cond = df_15m['high_i'] < df_15m['low_i-2']

        # 找出新形成的 FVG (只处理最近的几根 K 线以提高效率)
        # 假设 df_15m 是递增的
        lookback = 3 # 只检查最后几根15min K线是否有新FVG
        for i in range(max(3, len(df_15m) - lookback), len(df_15m)):
            fvg_time = df_15m['date'].iloc[i] # FVG 由第 i 根 K 线确认，基于 i, i-1, i-2
            fvg_id_base = f"15m_{fvg_time.isoformat()}"

            # 检查是否已存在
            if any(fvg.formation_time == fvg_time for fvg in self.fvg_list):
                continue # 跳过已处理的 K 线

            # 创建看涨 FVG
            if bull_cond.iloc[i]:
                bottom = df_15m['high_i-2'].iloc[i]
                top = df_15m['low_i-1'].iloc[i] # 使用定义1
                # top = df_15m['low_i'].iloc[i] # 使用定义2
                if top > bottom: # 确保是有效的缺口
                    new_fvg = FVG(fvg_time, 'bullish', top, bottom, '15m')
                    print(f"发现新的看涨 FVG: {new_fvg}")
                    self.fvg_list.append(new_fvg)
                    return 1

            # 创建看跌 FVG
            if bear_cond.iloc[i]:
                top = df_15m['low_i-2'].iloc[i]
                bottom = df_15m['high_i-1'].iloc[i] # 使用定义1
                # bottom = df_15m['high_i'].iloc[i] # 使用定义2
                if top > bottom: # 确保是有效的缺口
                    new_fvg = FVG(fvg_time, 'bearish', top, bottom, '15m')
                    print(f"发现新的看跌 FVG: {new_fvg}")
                    self.fvg_list.append(new_fvg)
                    return 1
        return 0 # 没有新 FVG

        # 可选：移除过旧的、已失效的 FVG 以节省内存
        # cutoff_time = df_15m['date'].iloc[-1] - timedelta(days=...) # 例如只保留最近几天的
        # self.fvg_list = [fvg for fvg in self.fvg_list if fvg.formation_time > cutoff_time or fvg.status not in ['mitigated', 'expired']]


    def update_all_fvg_statuses(self, current_time: datetime, current_high: float, current_low: float):
        """使用最新的1分钟K线数据更新所有非最终状态FVG的状态"""
        updated_count = 0
        for fvg in self.fvg_list:
            if fvg.status not in ['mitigated', 'expired']:
                original_status = fvg.status
                fvg.update_status(current_time, current_high, current_low)
                if fvg.status != original_status:
                    updated_count += 1
                    # print(f"  FVG 状态更新: {fvg.id} 从 {original_status} -> {fvg.status}")
        if updated_count > 0:
            print(f"  {updated_count} 个 FVG 状态被更新 @ {current_time}")


    def vectorized_mark_dataframe(self, dataframe: DataFrame) -> DataFrame:
        """
        (较优方法) 在 DataFrame 上向量化地标记与当前活跃 FVG 相关的信息。
        这仍然需要访问 self.fvg_list 来获取活跃 FVG 的边界。
        """
        # 初始化标记列
        dataframe['fvg_bull_active'] = 0 # 是否存在活跃的看涨FVG？(1/0)
        dataframe['fvg_bear_active'] = 0 # 是否存在活跃的看跌FVG？(1/0)
        dataframe['fvg_in_bull'] = 0     # 当前价格是否在任何活跃看涨FVG内？
        dataframe['fvg_in_bear'] = 0     # 当前价格是否在任何活跃看跌FVG内？
        dataframe['fvg_bull_top'] = np.nan # 最近的活跃看涨FVG顶部
        dataframe['fvg_bull_bottom'] = np.nan
        dataframe['fvg_bull_mid'] = np.nan
        dataframe['fvg_bear_top'] = np.nan
        dataframe['fvg_bear_bottom'] = np.nan
        dataframe['fvg_bear_mid'] = np.nan
        # 你可以添加更多标记，例如 FVG 被触及中点、FVG 被突破等信号

        # 获取当前所有活跃（包括刚被触及）的 FVG
        active_bull_fvgs = sorted([fvg for fvg in self.fvg_list if fvg.direction == 'bullish' and fvg.status in ['active', 'touched_mid']], key=lambda f: f.formation_time)
        active_bear_fvgs = sorted([fvg for fvg in self.fvg_list if fvg.direction == 'bearish' and fvg.status in ['active', 'touched_mid']], key=lambda f: f.formation_time)

        # 简单的标记：标记是否存在活跃FVG，并记录最新一个的边界
        # 注意：这里只记录了 *最新发现* 的活跃 FVG 信息，更复杂的逻辑可以记录 *最近价格* 的 FVG
        if active_bull_fvgs:
            dataframe['fvg_bull_active'] = 1
            last_bull_fvg = active_bull_fvgs[-1]
            dataframe['fvg_bull_top'] = last_bull_fvg.top
            dataframe['fvg_bull_bottom'] = last_bull_fvg.bottom
            dataframe['fvg_bull_mid'] = last_bull_fvg.midpoint
            # 检查价格是否在 *任何一个* 活跃看涨 FVG 内
            for fvg in active_bull_fvgs:
                 dataframe['fvg_in_bull'] |= ((dataframe['low'] < fvg.top) & (dataframe['high'] > fvg.bottom)).astype(int) # 简化为触及

        if active_bear_fvgs:
            dataframe['fvg_bear_active'] = 1
            last_bear_fvg = active_bear_fvgs[-1]
            dataframe['fvg_bear_top'] = last_bear_fvg.top
            dataframe['fvg_bear_bottom'] = last_bear_fvg.bottom
            dataframe['fvg_bear_mid'] = last_bear_fvg.midpoint
            for fvg in active_bear_fvgs:
                 dataframe['fvg_in_bear'] |= ((dataframe['low'] < fvg.top) & (dataframe['high'] > fvg.bottom)).astype(int) # 简化为触及

        # 填充 NaN 值 (例如，如果没有活跃的 FVG，边界就是 NaN)
        # 可以选择向前填充 ffill() 如果你希望一直携带最近的 FVG 信息，但这取决于策略逻辑

        return dataframe

    # --- 交易逻辑 (占位符) ---
    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        n = self.swing_left_n.value
        m = self.swing_right_m.value
        col_name_last_sh = f'last_sh_{n}_{m}'
        col_name_last_sl = f'last_sl_{n}_{m}'

        # 示例：
        # dataframe.loc[
        #     (qtpylib.crossed_above(dataframe['close'], dataframe[col_name_last_sh])) &
        #     (dataframe['volume'] > 0),
        #     'buy'] = 1
        dataframe['buy'] = 0
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        n = self.swing_left_n.value
        m = self.swing_right_m.value
        col_name_last_sh = f'last_sh_{n}_{m}'
        col_name_last_sl = f'last_sl_{n}_{m}'

        # 示例：
        # dataframe.loc[
        #     (qtpylib.crossed_below(dataframe['close'], dataframe[col_name_last_sl])) &
        #     (dataframe['volume'] > 0),
        #     'sell'] = 1
        dataframe['sell'] = 0
        return dataframe


# --- End of Strategy Class ---

# 注意：这是一个策略类的框架，你需要将其保存为 .py 文件放在 user_data/strategies 目录下才能在 freqtrade 中使用。
# 你可以运行 `freqtrade backtesting --strategy SwingPointIndicatorStrategy ...` 来测试指标计算。
# 在实际策略中，你需要填充 `populate_buy_trend` 和 `populate_sell_trend` 方法。