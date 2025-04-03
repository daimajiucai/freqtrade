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


if __name__ == "__main__":
    run_collect()