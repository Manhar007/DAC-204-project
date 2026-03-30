# 🔍 Object Detection: R-CNN vs DETR vs YOLOv8

> _A comparative study of three modern object detection architectures — R-CNN, DETR, and YOLOv8 — trained and evaluated on a common benchmark dataset with an interactive inference app._

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)
![Streamlit](https://img.shields.io/badge/App-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Course](https://img.shields.io/badge/Course-DAC--204%20IIT%20Roorkee-blue)

---

## 📋 Table of Contents

- [About the Project](#-about-the-project)
- [Problem Statement](#-problem-statement)
- [Tech Stack](#-tech-stack)
- [Features](#-features)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Performance](#-model-performance)
- [Screenshots / Demo](#-screenshots--demo)
- [Future Improvements](#-future-improvements)
- [Acknowledgements](#-acknowledgements)
- [Author](#-author)

---

## 📌 About the Project

This project presents a **comparative analysis of three state-of-the-art object detection architectures**:

| Architecture | Paradigm | Key Idea |
|---|---|---|
| **R-CNN** | Region Proposal + CNN | Two-stage: propose regions, then classify |
| **DETR** | Transformer-based | End-to-end detection with attention mechanism |
| **YOLOv8** | Single-shot | Real-time detection in one forward pass |

All three models are trained and evaluated on the same dataset for a fair comparison. The best-performing model weights are saved as `best.pt` and served through an interactive **Streamlit web app** (`app.py`) for live inference.

---

## 🎯 Problem Statement

Object detection architectures make fundamentally different trade-offs between **accuracy**, **inference speed**, and **architectural complexity**. This project investigates:

- How do two-stage (R-CNN), transformer-based (DETR), and single-shot (YOLOv8) detectors compare under identical training conditions?
- Which architecture achieves the best mAP on the chosen dataset?
- Which model is most practical for real-world deployment given GPU/CPU constraints?

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| Language | Python 3.8+ |
| Deep Learning | PyTorch, Torchvision |
| Object Detection | Ultralytics YOLOv8, HuggingFace Transformers |
| Data Processing | NumPy, Pandas, OpenCV, Pillow |
| Visualization | Matplotlib, Seaborn |
| Web App | Streamlit |
| Environment | Jupyter Notebook / Google Colab / Local GPU |
| Version Control | Git, GitHub |

---

## ✨ Features

- ✅ **Three complete model implementations** — R-CNN, DETR, and YOLOv8 in clean, self-contained folders
- ✅ **Unified evaluation protocol** — same dataset splits and metrics across all models
- ✅ **Best weights included** — `best.pt` (YOLOv8) ready for zero-setup inference
- ✅ **Interactive Streamlit app** — upload any image via `app.py` and see live bounding boxes
- ✅ **Comprehensive metrics** — mAP@50, mAP@50-95, precision, recall, and inference latency
- ✅ **Side-by-side comparison** — qualitative and quantitative results across architectures

---

## 📦 Dataset

| Property | Details |
|---|---|
| **Name** | [Dataset name — e.g., COCO / Pascal VOC / Custom] |
| **Source** | [URL — e.g., https://cocodataset.org] |
| **Total Images** | [e.g., ~60,000 images] |
| **Classes** | [e.g., 12 classes: person, car, tuktuk, …] |
| **Annotation Format** | [e.g., YOLO .txt / COCO JSON / Pascal VOC XML] |
| **Split** | Train: 70% · Validation: 15% · Test: 15% |

> 📁 The dataset is **not** included in this repository. Download it from the source above and place it in the `data/` directory as described in the Installation section.

---

## 🗂️ Project Structure

```
📦 DAC-204-project/
│
├── 📁 R-CNN/                    # Two-stage detector implementation
│   ├── train.py                 # Training script
│   ├── evaluate.py              # Evaluation & mAP computation
│   ├── model.py                 # Model architecture
│   └── utils.py
│
├── 📁 detr/                     # Transformer-based detector
│   ├── train.py
│   ├── evaluate.py
│   ├── model.py
│   └── utils.py
│
├── 📁 yolo/                     # YOLOv8 single-shot detector
│   ├── train.py
│   ├── evaluate.py
│   └── data.yaml                # Dataset config (Ultralytics format)
│
├── 📄 app.py                    # Streamlit inference web app
├── 📄 best.pt                   # Best-performing YOLOv8 weights
├── 📄 .gitignore
└── 📄 README.md
```

---

## ⚙️ Installation

### Prerequisites

- Python **3.8+**
- `pip` package manager
- NVIDIA GPU + CUDA *(strongly recommended for training; CPU sufficient for inference)*

### Step 1 — Clone the Repository

```bash
git clone https://github.com/Manhar007/DAC-204-project.git
cd DAC-204-project
```

### Step 2 — Create a Virtual Environment *(recommended)*

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

> If `requirements.txt` is not present, install manually:
> ```bash
> pip install torch torchvision ultralytics transformers \
>             streamlit opencv-python matplotlib seaborn pillow
> ```

### Step 4 — Prepare the Dataset

```bash
mkdir -p data/train data/val data/test
# Download the dataset and place images + labels here
```

---

## 🚦 Usage

### 🌐 Streamlit App — Quickest Demo

```bash
streamlit run app.py
```

Open `http://localhost:8501`, upload an image, and get instant detection results powered by `best.pt`.

---

### 🔵 R-CNN

```bash
cd R-CNN

# Train
python train.py --data ../data/ --epochs 50 --batch_size 4

# Evaluate
python evaluate.py --weights rcnn_best.pth --data ../data/test/
```

---

### 🟣 DETR

```bash
cd detr

# Train
python train.py --data_path ../data/ --epochs 50 --lr 1e-4

# Evaluate
python evaluate.py --checkpoint detr_checkpoint.pth --data_path ../data/test/
```

---

### 🟡 YOLOv8

```bash
cd yolo

# Train
yolo detect train data=data.yaml model=yolov8n.pt epochs=100 imgsz=640

# Evaluate on test set
yolo detect val data=data.yaml model=../best.pt

# Predict on a single image
yolo detect predict model=../best.pt source=path/to/your/image.jpg
```

---

## 📊 Model Performance

> All results measured on the **held-out test split** under identical conditions.

### 🏆 Benchmark Comparison

| Model | mAP@50 ↑ | mAP@50-95 ↑ | Precision ↑ | Recall ↑ | 
| **YOLOv8** ⭐ | **84.52 %** | **64.39%** | **83.42%** | **78.03%** |



## 🔮 Future Improvements

- [ ] **Backbone scaling** — experiment with YOLOv8m/l/x and DETR with ResNet-101
- [ ] **Advanced augmentation** — mosaic, mixup, and random affine transforms
- [ ] **Hyperparameter search** — automated tuning with Optuna or W&B sweeps
- [ ] **Model ensembling** — combine predictions from multiple models via WBF
- [ ] **Video inference** — extend `app.py` to process uploaded video files in real-time
- [ ] **Edge deployment** — export YOLOv8 to ONNX / TensorRT for faster inference
- [ ] **Docker container** — package the Streamlit app for one-command cloud deployment

---

## 🙏 Acknowledgements

- **DAC-204 Course Instructors & TAs**, IIT Roorkee — for the project framework and guidance
- [**Ultralytics**](https://github.com/ultralytics/ultralytics) — for the YOLOv8 framework and pre-trained weights
- [**Facebook Research**](https://github.com/facebookresearch/detr) — for the original DETR implementation
- [**HuggingFace Transformers**](https://huggingface.co/docs/transformers) — for DETR model support
- [** https://www.kaggle.com/datasets/mohamedra9ab/object-detection-for-autonomous-cars-egypt/data**] — the benchmark dataset

---



_DAC-204 · IIT Roorkee · [Year]_

</div>

