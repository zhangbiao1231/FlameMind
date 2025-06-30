# flameMind 🔥, 1.0.0 license
"""
Train a flame-mind classifier model on a classification dataset.

Usage - Single-GPU training:
    $ python classify/train.py --model yolov5s-cls.pt --data imagenette160 --epochs 5 --img 224

Datasets:           --data , tiny-dog, cifar10, or 'path/to/data'
dong-greed-cls models:  --model resnet34-cls.pt
Torchvision models: --model resnet34, vgg19, etc. See https://pytorch.org/vision/stable/models.html
"""
import argparse
import os
import subprocess
import math
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # flame root directory#
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import val as validate

from utils.general import (
    DATASETS_DIR,
    LOGGER,
    TQDM_BAR_FORMAT,
    colorstr,
    increment_path,
    init_seeds,
    yaml_save,
    print_args,
    try_gpu,
)
from utils.loggers import GenericLogger,SummaryWriter
from utils.dataLoader import SeqDataLoader, concat_data
from models.Seq2Seq import Seq2SeqEncoder, Seq2SeqAttentionDecoder
from models.rnn_layer import get_rnn_layer
from models.EncoderDecoder import EncoderDecoder
from utils.torch_utils import (
    grad_clipping,
    smart_optimizer,
    smart_resume,
    EarlyStopping,
)

LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))
RANK = int(os.getenv("RANK", -1))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))

def train(opt,device):
    """Trains a flame-Mind model, managing datasets, model optimization, logging, and saving checkpoints."""
    init_seeds(opt.seed + 1 + RANK, deterministic=True)
    save_dir,weights,bs,ns,epochs,resume,nw,pretrained,is_train,freeze= (
        opt.save_dir,
        opt.weights,
        opt.batch_size,
        opt.num_steps,
        opt.epochs,
        opt.resume,
        min(os.cpu_count() - 1, opt.workers),
        str(opt.pretrained).lower() == "true",
        opt.is_train,
        opt.freeze,
    )
    cuda = device.type != "cpu"

    # Directories
    wdir = save_dir / "weights"
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    last, best = wdir / "last.pt", wdir / "best.pt"

    # Save run settings
    yaml_save(save_dir / "opt.yaml", vars(opt))

    # Logger
    logger = GenericLogger(opt=opt, console_logger=LOGGER) if RANK in {-1, 0} else None

    # tensorboard
    writer = SummaryWriter(log_dir=str(save_dir))

    # Dataloaders
    train_dir = DATASETS_DIR / "train"
    train_features, train_labels ,scaler= concat_data(data_folder=train_dir, use_random=False)

    import pickle
    # 先保存 scaler 到内存 (不写文件)
    scaler_bytes = pickle.dumps(scaler)

    trainloader = SeqDataLoader(
        train_features,
        train_labels,
        batch_size=bs,
        num_steps=ns,
        use_random_iter=True,
    )

    valid_dir = DATASETS_DIR / "valid"
    if RANK in {-1, 0}:
        valid_features, valid_labels,_ = concat_data(data_folder=valid_dir, use_random=False)
        validloader = SeqDataLoader(
            valid_features,
            valid_labels,
            batch_size=bs,
            num_steps=ns,
            use_random_iter=False,)

    # model
    # create model
    encoder = Seq2SeqEncoder(dims=opt.dims,
                             num_hiddens=opt.num_hiddens,
                             num_layers=opt.num_layers,
                             get_rnn_layer=get_rnn_layer,
                             selected_model=opt.encoder_model)
    decoder = Seq2SeqAttentionDecoder(dims=opt.dims,
                                      embed_size=opt.embed_size,
                                      num_hiddens=opt.num_hiddens,
                                      num_layers=opt.num_layers,
                                      get_rnn_layer=get_rnn_layer,
                                      selected_model=opt.decoder_model)
    model = EncoderDecoder(encoder, decoder).to(torch.float32)

    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    model.apply(xavier_init_weights)
    model.to(device)
    model.train()
    pretrained = str(weights).endswith(".pt") and pretrained
    if pretrained:
        # 直接加载model
        ckpt = torch.load(weights, map_location="cpu")
        model = ckpt["model"]
        LOGGER.info(f"Loaded full model from {weights}") # report
    else:  # create
        model = model.to(device)
    # Info
    if RANK in {-1, 0}:
        if opt.verbose:
           LOGGER.info(model)

    # Optimizer
    optimizer = smart_optimizer(model=model,name=opt.optimizer,
                                lr=opt.lr0,momentum=0.9,
                                decay=opt.decay)
    # Scheduler
    lrf = opt.lrf  # final lr (fraction of lr0)
    lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf  # cosine
    def lf(x):
        """Linear learning rate scheduler function, scaling learning rate from initial value to `lrf` over `epochs`."""
        return (1 - x / epochs) * (1 - lrf) + lrf  # linear
    # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf) #余弦退火
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, opt.lr_period, opt.lr_decay)

    # Resume
    best_fitness, start_epoch = 0.0, 0
    if pretrained:
        if resume:
            best_fitness, start_epoch, epochs = smart_resume(ckpt, optimizer,None,weights, epochs, resume)
        del ckpt

    # Train
    t0 = time.time()
    scheduler.last_epoch = start_epoch-1
    criterion = nn.CrossEntropyLoss(reduction="none")
    best_fitness = 0.0
    stopper, stop = EarlyStopping(patience=opt.patience, min_delta=opt.min_delta), False
    val = valid_dir.stem  # 'valid' or 'test'
    LOGGER.info(
        f'data iter {len(trainloader)} train, {len(validloader)} valid\n'
        f"Logging results to {colorstr('bold', save_dir)}\n"
        f'Starting training flame-mind model for {epochs} epochs...\n\n'
        f"{'Epoch':>10}{'GPU_mem':>10}{'train_loss':>12}{f'{val}_loss':>14}{f'{val}_acc':>13}"
    )

    for epoch in range(start_epoch,epochs):  # loop over the dataset multiple times
        tloss, vloss, fitness = 0.0, 0.0, 0.0  # train loss, val loss, fitness

        pbar = enumerate(trainloader)
        if RANK in {-1, 0}:
            pbar = tqdm(enumerate(trainloader), total=len(trainloader), bar_format=TQDM_BAR_FORMAT)
        for i, (X, y) in pbar:  # progress bar
            optimizer.zero_grad()
            X, y = X.to(torch.float32), y.long()
            y_hat, _ = model(X, y)
            # Forward
            l = criterion(y_hat.permute(0, 2, 1), y).sum()
            # Backward
            l.backward()
            # Optimize
            grad_clipping(net = model, theta = 1) # clip gradients
            optimizer.step()
            if RANK in {-1, 0}:
                # Print
                tloss = (tloss * i + l.item()/y.shape[0]) / (i + 1)  # update mean losses
                mem = "%.3gG" % (torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0)  # (GB)
                pbar.set_description(
                    f"{f'{epoch + 1}/{epochs}':>10}{mem:>10}{tloss:>12.3g}" + " " * 36
                )

                # # validate #
                if i == len(pbar) - 1:  # last batch
                    acc, vloss = validate.run(
                        model=model, dataloader=validloader, criterion=criterion, pbar=pbar
                    )
                    fitness = acc  # define fitness as top1 accuracy
        # Scheduler
        scheduler.step()
        stop = stopper(epoch=epoch, fitness=fitness)  # early stop check

        # Log metrics
        if RANK in {-1, 0}:
            # Best fitness
            if fitness > best_fitness:
                best_fitness = fitness

            # Log
            # acc = 0.0
            metrics = {
                "train/loss": tloss,
                f"{val}/loss": vloss,
                "metrics/accuracy": acc,
                "lr/0": optimizer.param_groups[0]["lr"],
            }  # learning rate
            logger.log_metrics(metrics, epoch+1)
            # scalars方法会创建三个目录存放日志，tb中勾选可以叠加图像
            writer.add_scalars(main_tag="training over epoch",
                              tag_scalar_dict={"train/loss": tloss,
                                               f"{val}/loss": vloss,
                                               "metrics/accuracy": acc},
                              global_step=epoch,)
            for k ,v in metrics.items():
                writer.add_scalar(tag=k,
                                  scalar_value=v,
                                  global_step=epoch)
            # Save model
            final_epoch = epoch + 1 == epochs
            if (not opt.nosave) or final_epoch:
                ckpt = {
                    "epoch": epoch,
                    "best_fitness": best_fitness,
                    "model_state_dict": model.state_dict(), # 保存模型权重
                    "model":model, # 保存整个模型对象（含权重）
                    "optimizer": optimizer.state_dict(),
                    "opt": vars(opt),
                    "date": datetime.now().isoformat(),
                    "scaler": scaler_bytes}
                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fitness:
                    torch.save(ckpt, best)
                if opt.save_period > 0 and epoch % opt.save_period == 0:
                    torch.save(ckpt, wdir / f"epoch{epoch}.pt")
                del ckpt
        # EarlyStopping
        if stop:
            break

    # Train complete
    if RANK in {-1, 0}:
        LOGGER.info(
            f"\n{epochs - start_epoch} epochs completed in {(time.time() - t0) / 3600:.3f} hours."
            f"\nResults saved to {colorstr('bold', save_dir)}"
            f'\nPredict:         python3 predict.py --weights {best} --source example.csv'
            f'\nValidate:        python3 val.py --weights {best}'
        )

def parse_opt(known=False):
    """Parses command line arguments for YOLOv5 training including model path, dataset, epochs, and more, returning
    parsed arguments.
    """
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model", type=str, default= "", help="initial weights path")
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "runs/train-cls/exp18/weights/best.pt",
                        help="model.pt path(s)")
    # ==============================================================about model===================================================================#
    parser.add_argument("--num-hiddens", type=int, default=32, help="number of hiddens")
    parser.add_argument("--num-layers", type=int, default=2, help="number of layers")
    parser.add_argument("--embed-size", type=int, default=8, help="embedding size")
    parser.add_argument("--dims", type=int, default=5, help="input size")
    parser.add_argument("--encoder-model", type=str, default="GRU", help="select rnn model, i.e. RNN, GRU, LSTM et.al")
    parser.add_argument("--decoder-model", type=str, default="GRU", help="select rnn model, i.e. RNN, GRU, LSTM et.al")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout (fraction)")
    # ==============================================================about data===================================================================#
    parser.add_argument("--batch-size", type=int, default=32, help="total batch size for all GPUs")
    parser.add_argument("--num-steps", type=int, default=10, help="number of steps")
    parser.add_argument("--use-random-iter", type=bool, default=False, help="use random iter")
    # ==============================================================about train===================================================================#
    parser.add_argument("--epochs", type=int, default=600, help="total training epochs")
    parser.add_argument("--resume", nargs="?", const=True, default=False, help="resume most recent training")
    parser.add_argument("--nosave", action="store_true", help="only save final checkpoint")
    parser.add_argument("--cache", type=str, nargs="?", const="ram", help='--cache images in "ram" (default) or "disk"')
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--workers", type=int, default=8, help="max dataloader workers (per RANK in DDP mode)")
    parser.add_argument("--project", default=ROOT / "runs/train-cls", help="save to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--pretrained", nargs="?", const=True, default=False, help="start from i.e. --pretrained False")
    parser.add_argument("--optimizer", choices=["SGD", "Adam", "AdamW", "RMSProp"], default="Adam", help="optimizer")
    parser.add_argument("--lr0", type=float, default=1e-7, help="initial learning rate")
    parser.add_argument("--lrf", type=float, default=1e-2, help="terminal learning rate")
    parser.add_argument("--lr-period", type=int, default=20, help="learning rate period")
    parser.add_argument("--lr-decay", type=float, default=0.9, help="learning rate * decay over period per")
    parser.add_argument("--decay", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--label-smoothing", type=float, default=0.1, help="Label smoothing epsilon")
    parser.add_argument("--verbose", action="store_true", help="Verbose mode")
    parser.add_argument("--seed", type=int, default=0, help="Global training seed")
    parser.add_argument("--local-rank", type=int, default=-1, help="Automatic DDP Multi-GPU argument, do not modify")
    parser.add_argument("--is-train", default=False, help="")
    parser.add_argument("--patience", type=int, default=20, help="EarlyStopping patience (epochs without improvement)")
    parser.add_argument("--min-delta", type=float, default=0.001,
                        help="EarlyStopping Minimum Delta (epochs without improvement)")
    parser.add_argument("--save-period", type=int, default=-1, help="Save checkpoint every x epochs (disabled if < 1)")
    parser.add_argument("--freeze", nargs="+", type=int, default=[1], help="Freeze layers: backbone=10, first3=0 1 2")

    return parser.parse_known_args()[0] if known else parser.parse_args()

def main(opt):
    """Executes YOLOv5 training with given options, handling device setup and DDP mode; includes pre-training checks."""
    if RANK in {-1, 0}:
        print_args(vars(opt))
        # check_requirements(ROOT / "requirements.txt")
    device = try_gpu()
    # Parameters
    opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run

    # Train
    train(opt, device)
def run(**kwargs):
    """
    Executes YOLOv5 model training or inference with specified parameters, returning updated options.

    Example: from yolov5 import classify; classify.train.run(data=mnist, imgsz=320, model='yolov5m')
    """
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)
    return opt

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)


