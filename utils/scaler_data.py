# FlameMind 🔥, 1.0.0 license
"""Scaler utils."""

from pathlib import Path
import os
import pandas as pd

import argparse
import sys

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # backfire root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from sklearn.preprocessing import StandardScaler


def scaler_data(df):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    return scaled_data,scaler

def get_features_labels(df):
    return df.iloc[:,:-1].values, df.iloc[:,-1].values

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-folder", type=str, default=ROOT / "datasets/train", help="train dataset path")
    parser.add_argument("--valid-folder", type=str, default=ROOT / "datasets/valid", help="valid dataset path")
    parser.add_argument("--test-folder", type=str, default=ROOT / "datasets/test", help="test dataset path")
    # hengqin
    parser.add_argument("--hengqin-test-folder", type=str, default=ROOT / "datasets/Hengqin_target_data", help="train dataset path")
    opt = parser.parse_args()
    return opt

def main(opt):
    hengqin_test_folder = opt.hengqin_test_folder
    for filename in os.listdir(hengqin_test_folder):
        file_path = os.path.join(hengqin_test_folder, filename)
        if filename == ".DS_Store":
            continue  # 过滤掉 .DS_Store 文件
        print(file_path)
        df = pd.read_csv(file_path)
        features, label = get_features_labels(df)
        print(f"after scaler:\n{scaler_data(features)[:5]}")

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)


