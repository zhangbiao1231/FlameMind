# flameMind 🔥, 1.0.0 license
"""Plotting utils."""
import contextlib
import math
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torch
import numpy as np

from utils.general import LOGGER, increment_path

# Settings
RANK = int(os.getenv("RANK", -1))
matplotlib.rc("font", **{"size": 14})
matplotlib.use("Agg")  # for writing to files only
import pandas as pd
def plot_df(df,range1):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    # ax.plot(df.loc[range1]["ZRTCS_11MBR01BT105XQ01"]-df.loc[range1]["ZRTCS_11MBA12BT103XQ01"],color="red",label=r"$\delta t$")
    #==============温差特征==============#
    t_45 = (df.loc[range1]["ZRTCS_11MBR01BT104XQ01"] + df.loc[range1]["ZRTCS_11MBR01BT105XQ01"]) / 2.
    t_23 = (df.loc[range1]["ZRTCS_11MBA12BT102XQ01"] + \
            df.loc[range1]["ZRTCS_11MBA12BT103XQ01"]) / 2.
    ax.plot(t_45 - t_23, color="red", label=r"$\Delta t$")

    plt.legend(loc="upper left")

    ax1 = ax.twinx()
    #==============低频特征==============#
    L_mean = (df.loc[range1]["ZRTCS_11PCVB_LFD1_AMP"] + df.loc[range1]["ZRTCS_11PCVB_LFD2_AMP"]) / 2.
    ax1.plot(L_mean, lw=1.5, label=r"$L mean AMP$")
    # ==============中频==============#
    M_mean = (df.loc[range1]["ZRTCS_11PCVB_MFD1_AMP"] + df.loc[range1]["ZRTCS_11PCVB_MFD2_AMP"] + \
              df.loc[range1]["ZRTCS_11PCVB_MFD3_AMP"] + df.loc[range1]["ZRTCS_11PCVB_MFD4_AMP"]) / 4.
    ax1.plot(M_mean, lw=1.5, label=r"$M mean AMP$")
    # ==============高频特征==============#
    H_mean = (df.loc[range1]["ZRTCS_11PCVB_HFD1_AMP"] + df.loc[range1]["ZRTCS_11PCVB_HFD2_AMP"] + \
              df.loc[range1]["ZRTCS_11PCVB_HFD3_AMP"] + df.loc[range1]["ZRTCS_11PCVB_HFD4_AMP"]) / 4.
    ax1.plot(H_mean, lw=1.5, label=r"$H mean AMP$")
    plt.legend(loc="best")
    # plt.grid()
    # plt.show()
    plt.savefig("test.png", dpi=300, bbox_inches="tight")
    plt.close()
# 测试集验证
def plot_comparison(fname):
    df =  pd.read_csv(fname)
    file_stem = Path(fname).stem.replace("_result", "")
    save_path = Path(fname).parent / f"{file_stem}_plot.png"
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # 准确率计算
    acc = np.mean(df["true_label"].values == df["prediction"].values)

    # 绘制真实值（绿线）
    plt.plot(df['time_index'], df['true_label'], color='green', label='True')

    # 绘制预测值（红点）
    plt.scatter(df['time_index'], df['prediction'], color='red', label='Pred', s=10)

    # 添加图例和标签
    # plt.title(f"Flame State Prediction - {file_stem}")
    plt.title(f"{file_stem} - Accuracy: {acc * 100:.2f}%", fontsize=13)# 绘制准确率信息，后续补充延迟时间
    plt.xlabel('Time (s)')
    plt.ylabel('Flame State')
    plt.legend(loc="upper left")
    plt.grid(True)

    # 显示图像
    plt.tight_layout()
    plt.show()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"图像已保存：{save_path}")
# 训练过程
def plot_train(fname):
    df =  pd.read_csv(fname)
    df.columns = df.columns.str.strip()
    # 设置画布大小
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    ax.plot(df['epoch'], df['train/loss'], 'b-', label=r'Train Loss')
    ax.plot(df['epoch'], df['valid/loss'], 'r--', label='Valid Loss')

    # 添加图例和标签
    ax.set_xlabel('epoch',fontsize=14)
    ax.set_ylabel('Loss',fontsize=14)
    ax.set_title('Train Loss & Valid Accuracy vs. Epoch')
    plt.legend(loc="upper left")
    plt.grid(True)

    ax1 = ax.twinx()
    ax1.plot(df['epoch'], df['metrics/accuracy'], 'g', label='Valid Accuracy')
    ax1.set_ylabel('Accuracy',fontsize=14)
    plt.legend(loc="upper right")
    # plt.grid(True)

    # 显示图像
    plt.tight_layout()
    plt.show()

    plt.savefig("train.png", dpi=300, bbox_inches="tight")
    plt.close()
if __name__ == '__main__':
    # fname = "/Users/zebulonzhang/deeplearning/FlameMind/hengqin_3_comparison.csv"
    # plot_comparison(fname)
    fname1 = "/Users/zebulonzhang/deeplearning/FlameMind/runs/train-cls/exp18/results.csv"
    plot_train(fname1)