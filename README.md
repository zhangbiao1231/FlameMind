# 🔥 FlameMind: Deep Learning-Based Flame Detection for Gas Turbines
---
**FlameMind** is a high-performance flame detection system for heavy-duty gas turbines based on deep learning. 
It replaces traditional optical flame detectors by leveraging existing physical sensor data (temperature, pressure fluctuation, etc.) to predict flame 
presence in real time, achieving high accuracy(**>99%**) and low latency (**<1ms**).
---
## 🧠 Features
- 🔍 Real-time flame classification based on time-series features  
- 🔁 Encoder–Decoder architecture with GRU/RNN support  
- 📉 Accurate prediction using only physical sensor data (no optical probes)  
- 📦 Exportable models for deployment in edge/plant systems  
- 📊 Integrated visualization for training and inference results
---
## 🛠️ Installation
---
Clone the repository and set up the environment:
```
git clone git@github.com:zhangbiao1231/FlameMind.git
cd FlameMind
conda env create -f environment.yaml
conda activate flame_env
```
If you prefer pip, install from requirements.txt:
```
pip install -r requirements.txt
```
## 🚀 Getting Started
---
### 🔧 Train the model
```
python train.py --epochs 600 
```
### 🔍 Run inference on test data
```
python predict.py --weights runs/train-cls/exp18/best.pt --source datasets/test
```
## 📁 Project Structure
```
FlameMind/
├── train.py               # Script for training the flame prediction model
├── val.py                 # Script for model validation (evaluation on validation set)
├── predict.py             # Script for model inference on test data
├── d2l/                   # D2L core package
├── models/                # Model definitions (e.g., Encoder, Decoder, Seq2Seq, etc.)
├── utils/                 # Helper modules and utility functions
│   ├── dataLoader.py      # Sequence data loader (random/sequential partitioning)
│   ├── dataProcess.py     # Data pre-processing utilities
│   ├── loss.py            # Custom loss functions
│   ├── plots.py           # Visualization functions (training/inference curves)
│   ├── torch_utils.py     # PyTorch-specific helper functions (e.g., early-stopping)
│   ├── general.py         # DATASETS_DIR / datasets
│   └── scaler_data.py     # Data normalization/scaling utilities
├── datasets/              # Input dataset directory (structured into subfolders)
│   ├── train/             # Training CSV files
│   ├── test/              # Test CSV files
│   └── valid/             # Validation CSV files
├── runs/                  # Output directory for logs and results
│   ├── train-cls/         # Training results: weights, logs, charts
│   ├── val-cls/           # Validation results
│   └── inference-cls/     # Inference outputs: result CSVs, prediction plots
├── environment.yaml       # Conda environment configuration file
├── requirements.txt       # pip install dependency list (optional for pip users)
├── README.md              # Project introduction and usage instructions
└── setup.py               # Installation and packaging configuration
```
## 🧪 Example Inference Output
---
### 🔁 Inference Log (Console Output)
```
Start processing file: XXX.csv
XXX.csv Inference: 100%|████████████| 57600/57600 [00:27<00:00, 2059.22it/s]
FileName       Quantities     UnitTime    Accuracy
---------------------------------------------------
XX.csv         57590          0.52ms        99.74%
XX.csv         57590          0.51ms        99.79%
---------------------------------------------------
Avg Acc:                                  99.76%
Total Time:                               56.13s

Results saved to runs/inference-cls/exp18
```
### 📊 Inference Visualization
---
The figure below shows the comparison between predicted flame states and ground truth over time. 
The prediction accuracy is annotated directly on the plot.

<div align="center">
  <img src="runs/inference-cls/exp6/0313_final_plot.png" width="45%" />
  <img src="runs/inference-cls/exp6/1119_final_plot.png" width="45%" />
</div>

<p align="center">
  <b>Figure:</b> Inference results on <code>0313_final.csv</code> and <code>1119_final.csv</code> <br>
  Accuracy: <b>99.74%</b> and <b>99.79%</b> respectively.
</p>

> 🔍 Inference Speed: **< 1ms** on all files.

##  📃 Citing
---
```
@article{Zebulon,
  title={XXX},
  author={Zebulon,...},
  journal={XXX},
  volume={-},
  pages={-},
  year={XXX},
  publisher={XXX}
}
```