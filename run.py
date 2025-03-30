# run.py
import sys
from freqtrade.main import main

def run_robot():
    # 模拟命令行参数
    sys.argv = [
        "freqtrade",
        "trade",
        "--config", "user_data/config.json",
        "--strategy", "SampleStrategy"
    ]
    main()
    # import ccxt
    # exchange = ccxt.okx({
    #     'options': {'defaultType': 'future'},
    #     'proxies': {'http': 'http://127.0.0.1:7890', 'https': 'http://127.0.0.1:7890'}
    # })
    # markets = exchange.fetch_tickers()  # 尝试获取全市场数据
    # print(len(markets))  # 如果返回空或少量数据，说明不支持



if __name__ == "__main__":
    run_robot()