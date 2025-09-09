# Ultralytics YOLOv5 🚀, AGPL-3.0 license
"""
Validate a trained YOLOv5 classification model on a classification dataset.

Usage:
    $ bash data/scripts/get_imagenet.sh --val  # download ImageNet val split (6.3G, 50000 images)
    $ python classify/val.py --weights yolov5m-cls.pt --data ../datasets/imagenet --img 224  # validate ImageNet

Usage - formats:
    $ python3 val/val.py --weights best.pt                 # PyTorch

"""

import argparse
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm

from utils.dataLoader import SeqDataLoader, concat_data

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from utils.general import (
    LOGGER,
    TQDM_BAR_FORMAT,
    Profile,
    colorstr,
    intersect_dicts,
    increment_path,
    print_args,
)

from utils.general import try_gpu

# @smart_inference_mode()
def run(
        data="",  # dataset dir
        weights="",  # model.pt path(s)
        project=ROOT / "runs/val-cls-hengqin",   # save to project/name
        name="exp",  # save to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=False,  # use FP16 half-precision inference
        model=None,
        dataloader=None,
        criterion=None,
        pbar=None,
):
    """Validates a Seq2Seq model on a dataset, computing metrics like top1 accuracy."""
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device, pt, jit, engine = next(model.parameters()).device, True, False, False  # get model device, PyTorch model
        half &= device.type != "cpu"  # half precision only supported on CUDA
        model.half() if half else model.float()
    else:  # called directly
        device = try_gpu()

        # Directories
        # project = ROOT / "runs/valid-cls"
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        save_dir.mkdir(parents=True, exist_ok=True)  # make dir
        csv = save_dir/"results.csv"

        # Load model
        ckpt = torch.load(weights, map_location="cpu")  # 直接加载model
        model = ckpt["model"]
        LOGGER.info(f"Loaded full model from {weights}") # report

        # Dataloader
        valid_dir = Path(data) / "valid_hengqin" # datasets/valid
        valid_features, valid_labels,_ = concat_data(data_folder=valid_dir, use_random=False)
        dataloader = SeqDataLoader(
            valid_features,
            valid_labels,
            batch_size=64,
            num_steps=10,
            use_random_iter=False, )

    model.eval()
    pred, targets, loss, dt = [], [], 0, (Profile(device=device), Profile(device=device), Profile(device=device))
    n = len(dataloader)  # number of batches
    action = "validating"
    desc = f"{pbar.desc[:-36]}{action:>36}" if pbar else f"{action}"

    bar = tqdm(dataloader, desc, total=n, bar_format=TQDM_BAR_FORMAT, position=0,leave=False if training else True, disable=False)
    with torch.amp.autocast(device_type=device.type, enabled=(device.type != "cpu")):
        for X, y in bar:
            m = y.numel()
            with dt[0]:
                X, y = X.to(device, non_blocking=True), y.to(device)
            # Inference
            with dt[1]:
                y_hat, _ = model(X.to(torch.float32), y.long())

            with dt[2]:
                pred.append(y_hat.argmax(dim=-1))
                targets.append(y)
                if criterion:
                    loss += criterion(y_hat.permute(0, 2, 1), y).sum()/y.shape[0]
    loss /= n
    pred, targets = torch.cat(pred), torch.cat(targets)
    acc =  (targets == pred).float().sum()
        # print(acc/n/320)
    acc /= (n * m)
    if pbar:
        pbar.set_description (
            f"{pbar.desc[:-36]}{loss:>12.3g}{acc:>12.3g}"
                                   )
    return acc, loss

def parse_opt():
    """Parses and returns command line arguments for YOLOv5 model evaluation and inference settings."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=ROOT / "datasets",
                        help="dataset path")
    # parser.add_argument("--model", type=str, default=None, help="initial weights path")
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "runs/train-cls-hengqin/exp/weights/best.pt",
                        help="model.pt path(s)")
    parser.add_argument("--project", default=ROOT / "runs/val-cls-hengqin", help="save to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    opt = parser.parse_args()
    print_args(vars(opt))
    return opt


def main(opt):
    """Executes the YOLOv5 model prediction workflow, handling argument parsing and requirement checks."""
    # check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
