# FRAMU: A Lightweight Fractal Framework with Abductive Reasoning for Skin Lesion Segmentation

## Overview

This repository contains the implementation of FRAMU: A Lightweight Fractal Framework with Abductive Reasoning for Skin Lesion Segmentation.

## Key Features

- **Ultra-lightweight architecture**: Only about 10K parameters while maintaining competitive performance
- **Fractal-based multi-scale processing**: Captures features at multiple scales through recursive fractal structures
- **Abductive reasoning mechanism**: Self-reflection module that identifies and corrects prediction uncertainties
- **Adaptive knowledge base**: Domain-knowledge constraints for improved segmentation consistency

## Requirements

```
torch>=1.10.0
torchvision>=0.11.0
numpy>=1.19.0
scikit-learn>=0.24.0
matplotlib>=3.3.0
Pillow>=8.0.0
tensorboardX>=2.4.0
```

## File Structure

```
├── train.py              # Main training script
├── config.py             # Configuration settings
├── model.py              # FRAMU model architecture
├── knowledge_base.py     # Adaptive knowledge base implementation
├── engine.py             # Training/validation/testing loops
├── dataset.py            # Dataset loading utilities
├── utils.py              # Helper functions and loss definitions
└── README.md             # This file
```

## Dataset Structure

The code supports multiple medical imaging datasets:

- ISIC 2017
- ISIC 2018
- HAM10000
- BUSI

The ISIC 2017&2018 dataset utilizes the preprocessed datasets provided by [EGE-UNet](https://github.com/JCruan519/EGE-UNet) from [Google Drive](https://drive.google.com/file/d/1J6c2dDqX8qka1q4EtmTBA0w3Kez7-M6T/view?usp=sharing).

Expected directory structure:

```
data/
├── dataset_name/
│   ├── train/
│   │   ├── images/
│   │   ├── masks/
│   ├── val/
│   │   ├── images/
│   │   └── masks/
│   └── test/
│       ├── images/
│       └── masks/
```

## Train

1. Download the dataset and organize it into the data/ directory.

2. Update the `data_path` and datasets configuration in `config.py`.

3. Run `train.py` to train the model.


## Results

The model achieves competitive performance with significantly fewer parameters compared to state-of-the-art methods on multiple skin lesion imaging datasets.

