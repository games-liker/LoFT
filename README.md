## LoFT: Long-Tailed Semi-Supervised Learning via Parameter-Efficient Fine-Tuning in Open-World Scenarios

### Quick Start

#### **1. Dataset Preparation**

The provided scripts (e.g., `train_ssl.sh`) are designed for long-tailed / imbalanced datasets such as:

- **CIFAR100-IR100**
- **SmallImageNet** (default in `train_ssl.sh`)

#### **2. Training with the Example Script (Semi-supervised Learning)**

This repository provides a simple training script `train_ssl.sh` for semi-supervised learning on a chosen dataset. You can modify the variables at the top of the script to switch datasets and models:

```bash
# Edit dataset/model/GPU configs in train_ssl.sh, then run:
bash train_ssl.sh
```

Key arguments in `train_ssl.sh` (partial list):

- **data**: dataset name (e.g., `smallimagenet`, ...)
- **model**: visual backbone (e.g., `clip_vit_b16`)
- **lambda_u / lambda_uc**: losses weights for unlabeled and consistency terms
- **mu**: ratio of labeled vs. unlabeled samples
- **batch_size / micro_batch_size**: total batch size and gradient accumulation micro-batch
- **imb_ratio_label / imb_ratio_unlabel**: imbalance ratios for labeled and unlabeled sets
- **is_open**: enable open-set / OOD setting (concatenates unlabeled dataset with COCO dataset)
- **ood_root**: path to COCO/OOD dataset
- **output_dir**: directory for logs and checkpoints

The script uses `nohup` to run in the background and logs to `output/test.log`:

```bash
tail -f output/test.log
```

You can also call the main training script `main_ssl.py` directly, for example:

```bash
CUDA_VISIBLE_DEVICES=0 python main_ssl.py \
  -d smallimagenet \
  -m clip_vit_b16 \
  total_steps 10000 \
  eval_step 100 \
  num_epochs 100 \
  num_max 50 \
  num_max_u 400 \
  batch_size 16 \
  imb_ratio_label 10 \
  imb_ratio_unlabel 10 \
  flag_reverse_LT 0 \
  img_size 64 \
  micro_batch_size 16 \
  lambda_u 3.0 \
  lambda_uc 1.0 \
  lr 0.01 \
  threshold 0.6 \
  ood_threshold 0.3 \
  mu 7 \
  is_open True \
  ood_root "" \
  output_dir "test" \
  adaptformer True
```

> **Note**: Please adjust `batch_size`, `micro_batch_size`, and `total_steps` according to your available GPUs and memory.

#### **3. Outputs and Logs**

- Training logs and checkpoints are saved under `output/`, e.g.:
  - Log file: `output/test.log`
  - Models and results: `output/test/` (depending on `output_dir`)
- You can monitor the training log via:

```bash
tail -f output/test.log
```

---

### Project Structure (example)

- **`main_ssl.py`**: main entry point for long-tailed semi-supervised training
- **`train_ssl.sh`**: example training script wrapping common configurations
- **`configs/`**: configuration files for datasets, models, and training (if any)
- **`datasets/`**: dataset loading and preprocessing
- **`models/`**: model implementations and LoFT modules (e.g., CLIP backbone, adapters)
- **`trainer.py`**: training and evaluation logic
- **`utils/`**: utilities for logging, distributed training, metrics, etc.
- **`output/`**: training logs and model checkpoints

Please adjust this section to match your actual project structure.

---

### Acknowledgment

This project is inspired by and partially based on the following excellent open-source work:

- **Long-Tail Learning with Foundation Model: Heavy Fine-Tuning Hurts**  
  Code: [LIFT](https://github.com/shijxcs/LIFT)



