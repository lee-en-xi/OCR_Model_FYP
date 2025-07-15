# OCR Pipeline with PaddleOCR

This repository outlines an Optical Character Recognition (OCR) pipeline built using **PaddleOCR v3.1.0**. It covers installation, basic inference, finetuning, evaluation, and model export for both text detection and recognition tasks.

---

## 📦 Installation

To get started, follow these steps to set up your environment and install the necessary libraries:

### 1\. Create and Activate a Conda Environment

It's recommended to use a dedicated Conda environment for this project.

```bash
conda create -n paddle python=3.9
conda activate paddle
```

### 2\. Install PaddlePaddle (CPU Version)

Install the CPU version of PaddlePaddle.

```bash
python -m pip install paddlepaddle==3.1.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
```

> 💡 **Need a different version (e.g., GPU support)?** Refer to the [official PaddlePaddle installation guide](https://www.paddlepaddle.org.cn/en/install/quick?docurl=undefined) for instructions.

### 3\. Install PaddleOCR

Once PaddlePaddle is installed, install the PaddleOCR library.

```bash
pip install paddleocr
```

---

## 🔍 Inference

To run a basic OCR inference using the pre-trained models:

```bash
python inference.py
```

---

## 🧠 Finetuning

This section guides you through finetuning PaddleOCR models for custom text detection and recognition tasks.

### 1\. Clone the PaddleOCR Repository

First, clone the official PaddleOCR repository and navigate into its directory:

```bash
git clone https://github.com/PaddlePaddle/PaddleOCR
cd PaddleOCR
```

### 2\. Install Dependencies

Install all required Python packages for finetuning.

```bash
pip install -r requirements.txt
```

### 3\. Prepare Your Dataset

Organize your dataset according to the specified structure for text detection and recognition.

#### Text Detection Dataset Structure

For text detection, your dataset should have the following structure:

```
ocr_det_dataset_examples/
├── images/
├── train.txt
└── val.txt
```

**`train.txt` Sample Entry:**

Each line in `train.txt` (and `val.txt`) should contain the image path relative to the `images/` directory, followed by a JSON string describing the transcriptions and bounding box points.

```
images/train_img_61.jpg [{"transcription": "Ave", "points": [[655, 287], [698, 287], [696, 309], [652, 309]]}]
images/train_img_62.jpg [{"transcription": "Hey", "points": [[655, 287], [698, 287], [696, 309], [652, 309]]}]
```

#### Text Recognition Dataset Structure

For text recognition, your dataset should have this structure:

```
ocr_rec_dataset_examples/
├── dict.txt
├── images/
├── train.txt
└── val.txt
```

**`train.txt` Sample Entry:**

Each line in `train.txt` (and `val.txt`) should contain the image path relative to the `images/` directory, followed by the corresponding transcription.

```
images/train_word_1.png Genaxis Theatre
images/train_word_2.png Zeus
```

### 4\. Start Training

Before training, download the necessary pre-trained models.

**Download Pre-trained Detection Models:**

```bash
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_det_pretrained.pdparams
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_mobile_det_pretrained.pdparams
```

#### Text Detection Training

To start training a text detection model, edit data path in config file and run:

```bash
python tools/train.py -c configs/det/PP-OCRv5/PP-OCRv5_mobile_det.yml
```

**Download Pre-trained Recognition Models:**

```bash
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_rec_pretrained.pdparams
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_mobile_rec_pretrained.pdparams
```

#### Text Recognition Training

To start training a text recognition model, edit data path in config file and run:

```bash
python tools/train.py -c configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml -o Global.use_gpu=true
```

---

### 5\. Evaluation

After training, evaluate your models using the following commands. Ensure you replace paths and model names as per your setup.

#### Text Detection Evaluation

Evaluate detection models on your training and validation sets:

```bash
# Evaluate PP-OCRv4 on train.txt
python tools/eval.py -c configs/det/PP-OCRv4/PP-OCRv4_mobile_det.yml \
  -o Global.pretrained_model=./PP-OCRv4_mobile_det_pretrained.pdparams \
     Global.use_gpu=false \
     Eval.dataset.data_dir=../label_studio/det \
     Eval.dataset.label_file_list='[../label_studio/det/train.txt]'

# Evaluate PP-OCRv4 on val.txt
python tools/eval.py -c configs/det/PP-OCRv4/PP-OCRv4_mobile_det.yml \
  -o Global.pretrained_model=./PP-OCRv4_mobile_det_pretrained.pdparams \
     Global.use_gpu=false \
     Eval.dataset.data_dir=../label_studio/det \
     Eval.dataset.label_file_list='[../label_studio/det/val.txt]'

# Evaluate finetuned PP-OCRv5 on train.txt
python tools/eval.py -c configs/det/PP-OCRv5/PP-OCRv5_mobile_det.yml \
  -o Global.pretrained_model=output/PP-OCRv5_mobile_det/best_model/model.pdparams \
     Eval.dataset.data_dir=../label_studio/det \
     Eval.dataset.label_file_list='[../label_studio/det/train.txt]'

# Evaluate finetuned PP-OCRv5 on val.txt
python tools/eval.py -c configs/det/PP-OCRv5/PP-OCRv5_mobile_det.yml \
  -o Global.pretrained_model=output/PP-OCRv5_mobile_det/best_model/model.pdparams \
     Eval.dataset.data_dir=../label_studio/det \
     Eval.dataset.label_file_list='[../label_studio/det/val.txt]'
```

#### Text Recognition Evaluation

Evaluate recognition models on your training and validation sets:

```bash
# Evaluate PP-OCRv4 on train.txt
python tools/eval.py -c configs/rec/PP-OCRv4/PP-OCRv4_mobile_rec.yml \
  -o Global.pretrained_model=./PP-OCRv4_mobile_rec_pretrained.pdparams \
     Global.use_gpu=false \
     Eval.dataset.data_dir=../label_studio/rec \
     Eval.dataset.label_file_list='[../label_studio/rec/train.txt]'

# Evaluate PP-OCRv4 on val.txt
python tools/eval.py -c configs/rec/PP-OCRv4/PP-OCRv4_mobile_rec.yml \
  -o Global.pretrained_model=./PP-OCRv4_mobile_rec_pretrained.pdparams \
     Global.use_gpu=false \
     Eval.dataset.data_dir=../label_studio/rec \
     Eval.dataset.label_file_list='[../label_studio/rec/val.txt]'

# Evaluate PP-OCRv5 on train.txt (pretrained)
python tools/eval.py -c configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml \
  -o Global.pretrained_model=./PP-OCRv5_mobile_rec_pretrained.pdparams \
     Global.use_gpu=false \
     Eval.dataset.data_dir=../label_studio/rec \
     Eval.dataset.label_file_list='[../label_studio/rec/train.txt]'

# Evaluate finetuned PP-OCRv5 on train.txt
python3 tools/eval.py -c configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml \
  -o Global.pretrained_model=output/PP-OCRv5_mobile_rec/best_accuracy.pdparams \
     Eval.dataset.data_dir=../label_studio/rec \
     Eval.dataset.label_file_list='[../label_studio/rec/train.txt]'

# Evaluate finetuned PP-OCRv5 on val.txt
python3 tools/eval.py -c configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml \
  -o Global.pretrained_model=output/PP-OCRv5_mobile_rec/best_accuracy.pdparams \
     Eval.dataset.data_dir=../label_studio/rec \
     Eval.dataset.label_file_list='[../label_studio/rec/val.txt]'
```

---

### 6\. Export Inference Model

After successful training and evaluation, export your models for inference. This creates a deployable model directory.

#### Text Detection Model Export

```bash
python3 tools/export_model.py -c configs/det/PP-OCRv5/PP-OCRv5_server_det.yml \
  -o Global.pretrained_model=output/PP-OCRv5_server_det/best_accuracy.pdparams \
     Global.save_inference_dir="./PP-OCRv5_server_det_infer/"
```

**Resulting Directory Structure:**

```
PP-OCRv5_server_det_infer/
├── inference.json
├── inference.pdiparams
└── inference.yml
```

#### Text Recognition Model Export

```bash
python3 tools/export_model.py -c configs/rec/PP-OCRv5/PP-OCRv5_server_rec.yml \
  -o Global.pretrained_model=output/xxx/xxx.pdparams \
     Global.save_inference_dir="./PP-OCRv5_server_rec_infer/"
```

**Resulting Directory Structure:**

```
PP-OCRv5_server_rec_infer/
├── inference.json
├── inference.pdiparams
└── inference.yml
```

---
