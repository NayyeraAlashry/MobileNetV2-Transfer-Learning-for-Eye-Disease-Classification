# Retinal Eye Disease Classification Using MobileNetV2 & Transfer Learning

An end-to-end deep learning project re-implementing and adapting the transfer learning architecture from **Ragab et al. (2022)** from binary chest radiography to multi-class retinal fundus disease identification.

---

## Project Overview
- **Objective:** Automated multi-class classification of ophthalmic fundus photographs across four clinical categories.
- **Classes:** Cataract, Diabetic Retinopathy, Glaucoma, and Normal.
- **Dataset:** 4,217 retinal images sourced from [Kaggle Eye Diseases Classification](https://www.kaggle.com/datasets/gunavenkatuday/eye-diseases-classification).
- **Core Architecture:** Pretrained MobileNetV2 with domain adaptation via bottleneck fine-tuning and a dual-dropout classification head.
- **Deployment:** Interactive diagnostic web interface built using Gradio.

---

## Key Experimental Results

| Model | Architecture / Strategy | Test Accuracy | Test Loss | Macro F1 |
| :--- | :--- | :---: | :---: | :---: |
| **Baseline** | 2-Layer Custom CNN (from scratch) | 41.64% | 1.2552 | 40.85% |
| **Paper Model** | Frozen MobileNetV2 Feature Extractor | 83.44% | 0.4703 | 83.12% |
| **Improvement 1** | Data Augmentation (Evaluated across 3 Seeds) | 82.49% ± 1.10% | ~0.4900 | 82.35% |
| **Improvement 2 (Best)**| **Fine-Tuned MobileNetV2 (`features[17:]`)** | **89.43%** | **0.2988** | **89.39%** |

### Per-Class Test Performance ($N = 634$ Unseen Images)
| Diagnostic Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Cataract** | 92.64% | 88.82% | 90.69% | 170 |
| **Diabetic Retinopathy** | **96.10%** | **95.48%** | **95.79%** | 155 |
| **Glaucoma** | 80.52% | 87.94% | 84.07% | 141 |
| **Normal** | 88.34% | 85.71% | 87.01% | 168 |

---

## Repository Structure
```text
├── data/                                # Dataset root (cataract, diabetic_retinopathy, glaucoma, normal)
├── notebooks/
│   ├── 01_data_exploration.ipynb        # Dataset statistics & class distribution analysis
│   ├── 02_preprocessing_pipeline.ipynb   # Transforms, split 70/15/15 (Seed 42), DataLoaders
│   ├── 03_baseline_model.ipynb          # Simple CNN baseline implementation & training
│   ├── 04_paper_model.ipynb             # Re-implementation of Ragab et al. (frozen MobileNetV2)
│   └── 05_bonus_improvement.ipynb      # Augmentation, 3-seed validation & fine-tuning
├── 06_demo_app.py                       # Standalone interactive Gradio web application
├── mobilenet_v2_finetuned.pth           # Saved PyTorch state dictionary of best model
├── confusion_matrix.png                 # Evaluation confusion matrix heatmap
├── training_curves.png                  # Loss & accuracy progression curves
├── requirements.txt                     # Python dependencies
└── README.md
