# -*- coding: utf-8 -*-
import sys
from freqtrade.main import main
import os
from datetime import datetime, timedelta

def run_collect():
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
    '''
    #数据下载
    '''
    # 数据下载参数
    start_date_str = "20191227"  # 起始日期
    start_date = datetime.strptime(start_date_str, "%Y%m%d")
    end_date_str="20240926"
    end_date = datetime.strptime(end_date_str, "%Y%m%d")    # 结束日期
    current_end = end_date  # 从当前日期开始逆序下载
    while current_end >= start_date:
        # 计算当前块的开始日期（10天跨度）
        current_start = current_end - timedelta(days=9)
        current_end = current_end - timedelta(days=9)
        if current_start < start_date:
            current_start = start_date  # 
        # 生成时间段参数
        timerange_arg = (
            f"{current_start.strftime('%Y%m%d')}-"
        )
        print(f"\n下载时间段: {timerange_arg}")
        sys.argv = [
            "freqtrade",
            "download-data",
            "--config", config_path,
            "--timeframes", "1m","5m", "15m", "1h", "4h","1d",
            "--prepend",
            "--timerange", timerange_arg
        ]
                # 执行下载
        try:
            main()
        except SystemExit as e:
            if e.code != None:  # 处理正常退出
                print(f"错误退出码: {e.code}，终止下载")
                break
        except Exception as e:
            print(f"下载失败: {str(e)}")

def update_latest_data():
    """仅下载最新的缺失数据以补全现有数据。"""
    print("开始更新最新数据...")
    os.environ["PYTHONUTF8"] = "1"
    # 设置代理 (如果需要)
    os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
    os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

    # 配置文件路径 (根据你的系统调整)
    config_path = r"//192.168.123.62/share/user_data/config.json"
    # config_path = "/mnt/知识中心/user_data/config.json" # Linux/macOS 示例

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    else:
        print(f"找到配置文件: {config_path}")

    # 准备 Freqtrade 命令参数
    # **关键：省略 --timerange 参数**
    # 使用 --append 更符合更新最新数据的语义，但 Freqtrade 通常也能正确处理
    sys.argv = [
        "freqtrade",
        "download-data",
        "--config", config_path,
        "--timeframes", "1m", "5m", "15m", "1h", "4h", "1d",
        # "--append", # 可以考虑用 --append 替换 --prepend，语义更清晰
    ]

    print(f"执行命令: {' '.join(sys.argv)}")

    # 执行下载
    try:
        main()
        print("数据更新成功完成。")
    except SystemExit as e:
        # SystemExit(0) 是正常退出，非 0 通常表示错误
        if e.code is not None and e.code != 0:
            print(f"Freqtrade 异常退出，退出码: {e.code}。更新可能未完成。")
        elif e.code == 0:
            print("Freqtrade 正常退出 (code 0)。")
        else:
            print("Freqtrade 退出，但未提供明确的退出码。")
    except Exception as e:
        print(f"数据更新过程中发生错误: {str(e)}")


if __name__ == "__main__":
    # run_collect()
    update_latest_data()