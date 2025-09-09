# flame-mind 🔥, 1.0.0 license

import argparse
import os
import sys
from pathlib import Path
import time

import torch

import numpy as np
import pandas as pd

from collections import deque
from tqdm import tqdm

import matplotlib.pyplot as plt



plt.ioff()


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from utils.plots import plot_comparison
from utils.general import (
    LOGGER,
    Profile,
    colorstr,
    intersect_dicts,
    increment_path,
    print_args,
)
from utils.dataProcess import load_files, read_csv
from utils.general import try_gpu

RANK = int(os.getenv("RANK", -1))

def run(
        source="",  # dataset dir
        weights="",  # model.pt path(s)
        project="",  # save to project/name
        name="",  # save to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=False,  # use FP16 half-precision inference
        model=None,
        dataloader=None,
        criterion=None,
        maxlen=10,
):
    device = try_gpu()

    # Directories
    # project = ROOT / "runs/inference-cls"
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir
    # csv = save_dir / "results.csv"

    # Load model
    ckpt = torch.load(weights, map_location="cpu")  # 直接加载model
    model = ckpt["model"]
    LOGGER.info(f"Loaded full model from {weights}")  # report

    # 恢复scaler
    import pickle
    scaler = pickle.loads(ckpt['scaler'])

    # Dataloader-Buffer
    class Predictor:
        def __init__(self, model, maxlen=10, dims=5, scaler=None, device=device):
            self.device = device
            self.model = model.to(self.device)
            self.maxlen = maxlen
            self.dims = dims
            self.scaler = scaler
            self.buffer = deque([torch.zeros(self.dims) for _ in range(maxlen)], maxlen=maxlen)
            self.last_y = None
            self.results = []
        def update_buffer(self, new_data):
            """
           添加新数据到buffer（长度固定，自动滑动）
            参数:
            new_data: np.ndarray (dims,) 当前时刻的特征数据
            """
            self.buffer.append(new_data)

        def predict_next(self):
            """
            执行单步预测：
            1. 使用scaler标准化缓存数据
            2. encoder提取时序特征
            3. decoder进行一步预测
            """
            X_input = np.array(self.buffer)
            X_scaled = self.scaler.transform(X_input)
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0).to(self.device)  # (1, maxlen, dims)

            # 编码器处理时序数据
            encoder_output = self.model.encoder(X_tensor)  # Y (10,1,32) and state (2,1,32)
            state = self.model.decoder.init_state(encoder_output, None)  # state (2,1,32)

            # 解码器输入初始化
            if self.last_y is None:
                # 第一个 buffer，用 0 启动
                decoder_input = torch.zeros((1, 1), dtype=torch.long)  # (1, 1) # 单步预测 0启动
            else:
                # 后续 buffer 用上一次的预测
                decoder_input = self.last_y.unsqueeze(1)  # (1, 1)

            # 执行解码器预测
            output, _ = self.model.decoder(decoder_input, state)  # (1,10)
            y_hat = output.argmax(dim=-1)[:, -1]  # #(1,1)

            # 记录结果
            self.last_y = y_hat # 更新 decoder 启动输入
            self.results.append(y_hat.item())
            return self.results

    # Run inference
    # Inference on data
    def run_inference(df, predictor,file, maxlen):
        # ⏱️ 开始计时
        start_time = time.time()

        for t in tqdm(range(len(df)), desc=f"{file} 推理中", ncols=80):
            current_data = df.iloc[t, :-1].values  # shape = (5,),
            predictor.update_buffer(current_data)
            if t >= maxlen - 1:
                results = predictor.predict_next()
                # if t % 1000 == 0:
                #     print(f"第 {t} 秒预测火焰状态: {results[-1]}")
        end_time = time.time()  # ⏱️ 结束计时
        elapsed = end_time - start_time
        return predictor.results,elapsed

    def evaluate_inference(file, df, results, maxlen, save_dir):
        # 提取真实标签
        true_labels = df['flame_active'].iloc[maxlen:].tolist()
        pred_labels = results[:-1]

        assert len(true_labels) == len(pred_labels), "预测数量与真实标签数量不一致！"

        # 推理准确率
        accuracy = np.mean(np.array(true_labels) == np.array(pred_labels))

        # 假设你已在 run_inference 中返回了耗时（秒）
        elapsed = df.attrs.get("elapsed", 1e-6)  # 容错（确保非零）
        time_per_sample = elapsed / len(true_labels) * 1000.0

        # 打印结果
        # print("文件名  数量  平均推理时间 准确率")
        print(f"文件: {file:<18}数量: {len(true_labels):<12} 单位推理时间: {time_per_sample:<8.2f} ms     准确率: {accuracy * 100:<7.2f}%")

        # 保存预测结果
        df_compare = pd.DataFrame({
            'time_index': list(range(maxlen, maxlen + len(true_labels))),
            'true_label': true_labels,
            'prediction': pred_labels,
        })
        df_compare.to_csv(save_dir / f"{Path(file).stem}_result.csv", index=False)

        return accuracy, time_per_sample, len(true_labels)
    # Load data
    files = load_files(source)

    all_results = []
    total_elapsed_time = 0.
    for file in files:
        print(f"\n开始推理文件: {file}")

        # ⏱️ 开始计时
        # start_time = time.time()

        df = read_csv(Path(source) / file)
        predictor = Predictor(model, maxlen=maxlen, dims=5, scaler=scaler, device="cpu")  # 每个文件重新初始化预测器
        results, elapsed= run_inference(df, predictor,file, maxlen)

        df.attrs["elapsed"] = elapsed  # 传入评估函数

        accuracy, time_per_sample,length = evaluate_inference(file, df, results, maxlen, save_dir)

        all_results.append({
            "file": file,
            "count": length,
            "time_per_sample": time_per_sample,
            "accuracy": accuracy * 100.,
            "elapsed": elapsed
        })
        total_elapsed_time += elapsed

    # 表头
    print(f"\n{'FileName':<30}{'Quantities':<20}{'UnitTime':<12}{'Accuracy':<12}")
    print("-" * 75)

    # 每行数据
    for r in all_results:
        print(f"{r['file']:<30}{r['count']:<20}{r['time_per_sample']:<12.2f} {r['accuracy']:<12.2f}")

    # 汇总
    mean_acc = sum(r['accuracy'] for r in all_results) / len(all_results)
    total_elapsed = sum(r['elapsed'] for r in all_results)

    print("-" * 75)
    print(f"{'Avg Acc:':<60}{mean_acc:>10.2f}%")
    print(f"{'Total Time:':<60}{total_elapsed:>10.2f}s")

    if RANK in {-1, 0}:
        LOGGER.info(
            f"\nResults saved to {colorstr('bold', save_dir)}",
        )

    # 绘制图像并保存
    for csv_file in save_dir.glob("*_result.csv"):
        plot_comparison(fname=csv_file)


def parse_opt():
    """Parses command line arguments for YOLOv5 inference settings including model, source, device, and image size."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default=ROOT / "datasets/test-hengqin",
                        help="dataset path")
    # parser.add_argument("--model", type=str, default=None, help="initial weights path")
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "runs/train-cls-hengqin/exp/weights/best.pt",
                        help="model.pt path(s)")
    parser.add_argument("--project", default=ROOT / "runs/inference-cls-hengqin", help="save to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--maxlen", default=10, type=int, help="max length of buffer")
    opt = parser.parse_args()
    print_args(vars(opt))
    return opt

def main(opt):
    """Executes YOLOv5 model inference with options for ONNX DNN and video frame-rate stride adjustments."""
    # check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)








