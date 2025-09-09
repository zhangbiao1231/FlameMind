import collections
from pathlib import Path
import os
import math
import argparse
import matplotlib.pyplot as plt
import numpy as np
import sys
import shutil

import pandas as pd
import time

# from plots import plot_df

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # flameMind root directory: /Users/zebulonzhang/deeplearning/FlameMind
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # .

def load_files(src_folder):
    files = [f for f in os.listdir(src_folder)
             if f.endswith('.csv') and not f.startswith('.')]
    return files
def read_csv(fname):
    return pd.read_csv(fname)
def process_df(df, Threshold=90):
    # 切片
    # df = df.loc[range1]
    df["rpm"] = df.iloc[:, 1]
    df["delta_T"] = (df.iloc[:, 9:11].mean(axis=1) - df.iloc[:, 4:6].mean(axis=1)).round(2)
    df["H_amp"] = df.iloc[:, 12:16].mean(axis=1).round(2)
    df["M_amp"] = df.iloc[:, 16:20].mean(axis=1).round(2)
    df["L_amp"] = df.iloc[:, 20:22].mean(axis=1).round(2)
    # df["flame_density"] = df.iloc[:, 22:50].mean(axis=1).round(2) # 取均值
    df["flame_density"] = df.iloc[:, 22:50].max(axis=1).round(2) # 取最大值
    df['flame_active'] = (df['flame_density'] > Threshold).astype(int)
    return df

# 横琴数据
def process_df_hengqin(df,Threshold=50):
    # 切片
    # df = df.loc[range1]
    df["rpm"] = df.iloc[:, 2]
    df["delta_T"] = (df.iloc[:, 1] - df.iloc[:, 3]).round(2)


    # df["H_amp"] = df.iloc[:, [16, 18]].mean(axis=1).round(2)  # H 16 18
    # df["M_amp"] = df.iloc[:, [10, 12, 14]].mean(axis=1).round(2)  # M 10, 12, 14
    # df["L_amp"] = df.iloc[:, [8]].round(2)  # L 8

    # 归一化函数
    def normalize(series, min_val, max_val):
        return (series - min_val) / (max_val - min_val)

    # 低频
    df["L_amp"] = normalize(df.iloc[:, 8], 2,4)

    # 中频
    df["M_amp"] = ((normalize(df.iloc[:, 10], 3,4)+
                   normalize(df.iloc[:, 12], 3,4)+
                   normalize(df.iloc[:, 14], 3,4))
                   / 3.0)

    # 高频
    df["H_amp"] = ((normalize(df.iloc[:, 16], 0.3,0.5)+
                   normalize(df.iloc[:, 18], 0.3,0.5))
                   / 2.0)

    df["flame_density"] = df.iloc[:, 4:8].max(axis=1).round(2)
    df['flame_active'] = (df['flame_density'] > Threshold).astype(int)
    return df

def save_df_hengqin(df,file,save_folder):
    # 选择你要保存的列
    selected_cols = [
        "rpm",
        "delta_T",
        "H_amp", "M_amp", "L_amp",
        "flame_active",
    ]
    # 新 DataFrame
    df_selected = df[selected_cols]
    print(df_selected)
    # 保存为新 CSV 文件
    df_selected.to_csv(Path(save_folder)/"{}_final.csv".format(file.split('.')[0]), index=False)
# hengqin
    old_name = file
    new_name = None
    try:
        new_name = rename_file(old_name)
    except ValueError as e:
        print(e)
    print(f"原文件名: {old_name} -> 新文件名: {new_name}")
    df_selected.to_csv(Path(save_folder)/"{}_final.csv".format(new_name), index=False)
# 修改文件名
# “启动 2025.3.15 2345-0045.csv" -> "startup_20250315_final.csv"
import os
import re
def rename_file(old_filename):
    """
    将原始文件名重命名为标准格式：
    启动/停机 2025.3.15 2345-0045.csv -> startup/shutdown_20250315_final.csv
    """
    # 中文前缀映射
    prefix_map = {
        "启动": "startup",
        "停机": "shutdown"
    }
    # 识别前缀
    prefix = None
    for cn_prefix, en_prefix in prefix_map.items():
        if old_filename.startswith(cn_prefix):
            prefix = en_prefix
            break

    if not prefix:
        raise ValueError(f"无法识别前缀：{old_filename}")

    # 提取日期
    match = re.search(r"(\d{4})\.(\d{1,2})\.(\d{1,2})", old_filename)
    if not match:
        raise ValueError(f"无法提取日期：{old_filename}")

    year, month, day = match.groups()
    month = month.zfill(2)
    day = day.zfill(2)
    date_part = f"{year}{month}{day}"

    # 生成新文件名
    new_filename = f"{prefix}_{date_part}"
    return new_filename
def save_df(df,file,save_folder):
    # 选择你要保存的列
    selected_cols = [
        "rpm",
        "delta_T",
        "H_amp", "M_amp", "L_amp",
        # "flame_density",
        "flame_active",
    ]
    # 新 DataFrame
    df_selected = df[selected_cols]
    print(df_selected)
    # 保存为新 CSV 文件
    df_selected.to_csv(Path(save_folder)/"{}_final.csv".format(file.split('.')[0]), index=False)
def parse_opt():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--src-folder", type=str, default=ROOT / "datasets/source_data", help="dataset path")
    # parser.add_argument("--tgt-folder", type=str, default=ROOT / "datasets/target_data", help="dataset path")
    # parser.add_argument("--threshold", type=float, default=90, help="flame density threshold")
    # 横琴数据
    parser.add_argument("--src-folder", type=str, default=ROOT / "datasets/Hengqin-newdata", help="dataset path")
    parser.add_argument("--tgt-folder", type=str, default=ROOT / "datasets/Hengqin_target_newdata", help="dataset path")
    parser.add_argument("--Hengqin-threshold", type=float, default=50, help="flame density threshold")
    opt = parser.parse_args()
    return opt
def main(opt):
    src_folder = opt.src_folder
    tgt_folder = opt.tgt_folder
    print(f"src_folder:{src_folder}")
    print(f"tgt_folder:{tgt_folder}")
    # for f in os.listdir(src_folder):
    #     print(f)
    files = load_files(src_folder)
    for file in files:
        fname = Path(src_folder)/file
        # ⏱️ 开始计时
        start_time = time.time()
        df = read_csv(fname)
        # df_prime = process_df(df=df,Threshold=opt.threshold)
        # save_df(df_prime,file,tgt_folder)
        df_prime = process_df_hengqin(df=df,Threshold=opt.Hengqin_threshold)
        save_df_hengqin(df_prime,file,tgt_folder)

        # ⏱️ 结束计时
        end_time = time.time()
        elapsed = end_time - start_time

        # 单位时间预处理速度
        time_per_sample = elapsed / len(df) * 1000.

        # 输出
        print(f"文件 {file} 预处理耗时：{elapsed:.2f} 秒")
        print(f"数据量：{len(df)} 条，每条数据平均预处理时间：{time_per_sample:.6f} ms")
if __name__ == '__main__':
    opt = parse_opt()
    main(opt)




