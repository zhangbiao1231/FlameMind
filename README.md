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
python predict.py --weights runs/train-cls-hengqin/exp2/best.pt --source datasets/test_hengqin
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
│   ├── reog.py            # Reognize train/valid/test directionary
│   └── scaler_data.py     # Data normalization/scaling utilities
├── datasets/              # Input dataset directory (structured into subfolders)
│   ├── train/             # Training CSV files
│   ├── valid/             # Validation CSV files
│   ├── test/              # Test CSV files
│   ├── train_hengqin/     # Training CSV files for hengqin
│   ├── valid_hengqin/     # Validation CSV files for hengqin
│   └── test_hengqin/      # Test CSV files for hengqin
├── runs/                  # Output directory for logs and results
│   ├── train-cls/         # Training results: weights, logs, charts
│   ├── val-cls/           # Validation results
│   ├── inference-cls/     # Inference outputs: result CSVs, prediction plots
│   ├── train-cls-henqin/  # ... for hengqin
│   ├── val-cls-hengqin/   # ... for hengqin
│   └── inference-cls-hengqin/ # ... for hengqin
├── environment.yaml       # Conda environment configuration file
├── requirements.txt       # pip install dependency list (optional for pip users)
├── README.md              # Project introduction and usage instructions
└── setup.py               # Installation and packaging configuration
```

## 📚 Technical Details

---
### 📁 1. Raw Data Overview
- **Source**: Logged sensor data from gas turbine testbed (or simulation outputs)  
- **Sampling Rate**: 1 min (or project-specific)  
- **Features (base)**:
  - `load`, `diffusion_valve_feedback`, `exhaust_temp`, 
  - `compressor_inlet_temp`, `turbine_exhaust_temp_10B`, 
  - `amb_temperature`,`NG_inlet_temp`, `exhaust_temp`, etc.
- **Labels**:
  - `NOx_in_flue_gas` (i.e. NOx)
- **Size**:
  - N test runs × T time steps × D input features

> _Note: Missing values interpolated; outliers removed using IQR filtering._
---
### 🧹 2. Data Preprocessing
- **Cleaning**:
  - Remove Nulls and duplicates
  - Clip or mask physically invalid values (e.g., NOx < 0.0 ppm)
- **Normalization**:
  - Per-feature Min-Max scaling or StandardScaler
- **Segmentation**:
  - Split with `time_col=Time`, `freq = 1min`
- **Label Strategy**:
  - Predict next-step emission level (regression)
  - Optional: multi-step average prediction
- **Feature Engineering**:
  - valve_share, etc.
  - $\Delta$T, $\Delta$P,
  - Rolling statistics (e.g., mean, std, gradient)

---
### 🧠 3. Model Architecture
- **Base Model**: GRU-based sequence regression
- **Input Format**: `[batch_size, num-steps, num_features]`
- **Architecture**:``` Input → GRU → FC → Digital Prediction```
- **Variants to Explore**:
  - GRU, LSTM, or RNN
  - Temporal Convolutional Network (TCN)
  - Hybrid CNN + RNN for early feature extraction

---
### ⚙️ 4. Hyperparameters & Training Settings
| Parameter           | Value                             |
|---------------------|-----------------------------------|
| Batch Size          | 32                                |
| Number Steps        | 10/20/30                          |
| Number Hiddens      | 32/64/96                          |
| RNN Layers          | 2                                 |
| Dropout             | 0.5                               |
| Optimizer           | Adam                              |
| Learning Rate       | 1e-3                              |
| Epochs              | 1000                              |
| Loss Function       | MSE (per output)                  |
| Early Stopping      | Patience = 20                     |
| Learning Rate Decay | StepLR(lr_period=50,lr_decay=0.9) |

> _All training runs are logged under `runs/`, including checkpoints and loss/metric plots._

## 🧪 Example Inference Output

---
### 🔁 Inference Log (Console Output)
```
Start processing file: XXX.csv
XXX.csv Inference: 100%|████████████| 86391/86391 [00:44<00:00, 1940.77it/s]
FileName       Quantities     UnitTime    Accuracy
---------------------------------------------------
XX.csv         86391          0.50ms        99.75%
XX.csv         86391          0.52ms        99.73%
XX.csv         86391          0.53ms        99.75%
---------------------------------------------------
Avg Acc:                                  99.73%
Total Time:                               134.09s

Results saved to runs/inference-cls-hengqin/exp2
```
### 📊 Inference Visualization

---
The figure below shows the comparison between predicted flame states and ground truth over time. 
The prediction accuracy is annotated directly on the plot.

<div align="center">
  <img src="ExportData%20-6_final_plot_0.png" width="45%" />
  <img src="ExportData%20-6_final_plot_2.png" width="45%" />
  <img src="ExportData%20-6_final_plot_1.png" width="45%" />
</div>

<p align="center">
  <b>Figure:</b> Inference results on <code>ExportData -6_final.csv</code> by 
different weights (i.e. raw/pretrain/train)Accuracy: <b>59.42%</b> 、<b>67.76%</b> and <b>99.756%</b> respectively.
</p>

> 🔍 Inference Speed: **< 1 ms** on all files.

## 🌈 Conclusion

------------------------
- In this work, we used limited sensor data and carefully engineered derived 
features to establish a reliable dataset for emission prediction. 
Based on this foundation, we trained an efficient prediction system that performs 
well on both validation and test datasets. The model accurately captures emission 
trends with promising metrics and low inference latency(<0.5 ms), making it suitable for
real-time deployment in operation and maintenance diagnostic systems.

- Through extensive experiments with different hyperparameters and 
neural network architectures, we identified [GRU]() as the most effective model. 
The optimal configuration — with [num_steps=20]() and [num_hiddens=64]() — achieved 
the best performance, reaching an [Acc >= 0.99]() at valid datasets.
These results provide a $solid baseline$ for future fine-tuning and improvements.

- We hope this work can contribute to the development of gas turbine emission prediction. 
If you have better ideas or suggestions, we welcome collaboration and discussion.

  
## 📃 Citing

---
```
@article{Zebulon,
  title={XXX},
  author={Zebulon,...},
  journal={XXX},
  volume={-},
  pages={-},
  year={XXX},
  publisher={2025}
}
```

##  📬 Contact

---
Questions, suggestions, or collaboration inquiries?
> 📧 Email: 18856307989@163.com.cn