# CrowdCounter

CrowdCounter is a deep learning project based on CSRNet (Dilated Convolutional Neural Networks) for counting crowds in highly congested scenes. This repository provides the necessary scripts and instructions to train the model using the ShanghaiTech dataset. The implementation is inspired by the research paper [CSRNet: Dilated Convolutional Neural Networks for Understanding the Highly Congested Scenes](https://arxiv.org/pdf/1802.10062).

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
    - [Data Preparation](#data-preparation)
    - [Training](#training)
    - [Evaluation](#evaluation)
- [Results](#results)
- [References](#references)

## Introduction

CrowdCounter utilizes a Convolutional Neural Network (CNN) with dilated convolutions to address the challenge of counting crowds in highly congested scenes. By leveraging the CSRNet architecture, it effectively learns density maps for accurate crowd estimation.

## Dataset

The model is trained and evaluated using the [ShanghaiTech dataset](https://www.kaggle.com/datasets/tthien/shanghaitech). The dataset contains two parts:

- **Part A**: Includes images of highly congested scenes.
- **Part B**: Contains images of relatively sparse crowds.

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/StackUnderflowProject/CrowdCounter.git
   cd CrowdCounter
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Data Preparation

1. Download the ShanghaiTech dataset from [Kaggle](https://www.kaggle.com/datasets/tthien/shanghaitech).
2. Organize the dataset into the following structure:

   ```
   ShanghaiTech/
   |-- part_A/
   |   |-- train_data/
   |   |   |-- images/
   |   |   |-- ground_truth/
   |   |-- test_data/
   |       |-- images/
   |       |-- ground_truth/
   |-- part_B/
       |-- train_data/
       |   |-- images/
       |   |-- ground_truth/
       |-- test_data/
           |-- images/
           |-- ground_truth/
   ```

3. Preprocess the dataset to generate density maps:

   ```bash
   python preprocess.py --data_dir ./data
   ```

### Training

Start training the model:

```bash
python train.py --data_dir ./data --part A --epochs 100 --batch_size 16
```

- `--data_dir`: Path to the dataset directory.
- `--part`: Dataset part to train on (`A` or `B`).
- `--epochs`: Number of training epochs.
- `--batch_size`: Batch size for training.

### Evaluation

Evaluate the trained model:

```bash
python evaluate.py --data_dir ./data --part A --model_path ./model_best.pth
```

- `--model_path`: Path to the trained model.

## Results

Include your model’s performance metrics, such as Mean Absolute Error (MAE) and Mean Squared Error (MSE), for both Part A and Part B of the ShanghaiTech dataset.

| Dataset Part | MAE  | MSE  |
|--------------|------|------|
| Part A       | 11.9 | XX.X |
| Part B       | XX.X | XX.X |

## References

1. [CSRNet: Dilated Convolutional Neural Networks for Understanding the Highly Congested Scenes](https://arxiv.org/pdf/1802.10062)
2. [ShanghaiTech Crowd Counting Dataset](https://www.kaggle.com/datasets/tthien/shanghaitech)

