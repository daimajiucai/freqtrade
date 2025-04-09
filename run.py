# -*- coding: utf-8 -*-
# run.py
import sys
from freqtrade.main import main
import os
from datetime import datetime, timedelta

def run_robot():
    os.environ["PYTHONUTF8"] = "1" # 设置环境变量，确保 Python 使用 UTF-8 编码
    # 设置工作目录（可选）
    #os.chdir(os.path.dirname(os.path.abspath(__file__)))

        # 设置全局代理环境变量
    os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"     # 例如 http://192.168.1.100:1080
    os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"    # 如果有用户名密码：http://user:pass@192.168.1.100:1080

    
    # 配置路径（根据系统选择）  
    # Windows UNC 路径示例
    config_path = r"//192.168.123.62/share/user_data/config.json"
    # Linux/macOS 路径示例ewewe
    # config_path = "/mnt/知识中心/user_data/config.json"
    # 模拟命令行参数
    # 验证路径是否存在
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    else:
        print(f"找到配置文件: {config_path}")
    sys.argv = [
        "freqtrade",
        "trade",
        "--config", config_path,
        "--strategy", "SmcFvgStrategy0409"
    ]
    main()



if __name__ == "__main__":
    run_robot()