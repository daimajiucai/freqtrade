import os
import ccxt
import pandas as pd
from datetime import datetime, timedelta
import time

# 从环境变量读取代理配置（示例：export HTTP_PROXY="http://user:pass@192.168.1.100:8080"）
PROXY_CONFIG = {
    'http':"http://127.0.0.1:7890",  # 示例: http://username:password@proxy_ip:port
    'https': "http://127.0.0.1:7890"
}

def get_okx_earliest_timestamp(symbol='SOL/USDT:USDT', timeframe='1m', max_retries=3):
    """获取 OKX 期货某交易对的最早可用 K 线时间戳（带代理和重试）"""
    exchange = ccxt.okx({
        'enableRateLimit': True,
        'proxies': {
            'http': PROXY_CONFIG['http'],
            'https': PROXY_CONFIG['https']
        },
        'options': {
            'defaultType': 'future'  # 必须指定期货类型
        }
    })
    
    attempt = 0
    while attempt < max_retries:
        try:
            # 动态探测起点：从 2020 年开始，每次后移 1 年直至找到数据
            since_date = datetime(2000, 1, 1)
            found = False
            
            while not found:
                since = exchange.parse8601(since_date.isoformat() + 'Z')
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1)
                
                if ohlcv:
                    earliest_ts = ohlcv[0][0]
                    print(f"✅ 成功获取数据 | 最早 {timeframe} K线时间: {pd.to_datetime(earliest_ts, unit='ms')}")
                    return earliest_ts
                else:
                    print(f"⏳ 未找到 {since_date.year} 年数据，向后搜索...")
                    since_date += timedelta(days=365)
                    
                # 防止无限循环（假设交易所数据不早于 2010）
                if since_date.year > 2030:
                    raise ValueError("超出历史数据搜索范围")
                    
        except (ccxt.RequestTimeout, ccxt.NetworkError) as e:
            print(f"⚠️ 网络错误: {e} | 重试 {attempt+1}/{max_retries}")
            attempt += 1
            time.sleep(5)
        except ccxt.ExchangeError as e:
            print(f"❌ 交易所错误: {e}")
            break
            
    return None

if __name__ == "__main__":
    earliest_ts = get_okx_earliest_timestamp()
    if earliest_ts:
        print(f"OKX 期货最早有效时间戳: {earliest_ts}")
    else:
        print("无法获取数据，请检查网络或代理配置")