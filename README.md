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

    Follow the instructions in the `make_dataset.ipynb` notebook to generate the ground truth. It may take some time to generate the dynamic ground truth. Use the `create_image_paths_json()` function in `utils.py` to generate your own JSON file.

### Training

Start training the model:

```bash
python train.py <train_json> <val_json> <task_id>
```

- `<train_josn>`: Path to the JSON file generated with `create_image_paths_json` for training the model.
- `val_json`: Path to the JSON file generated with `create_image_paths_json` for validating the model.
- `<task_id>`: Task ID for the training process.

### Prediction

Have the model predict the crowd count for a given image:

```bash
python predict.py <image_path> <crowd_type>
```
- `<image_path>`: Path to the image for which the crowd count is to be predicted.
- `<crowd_type>`: Type of crowd ('dense' or 'sparse') for the image.

### Accuracy Evaluation

Evaluate the model's performance using Mean Absolute Error (MAE), 
the script will randomly select 100 images from the test set and calculate the MAE, 
note set up the correct project structure and paths before running the script.:

```bash
python test_model_accuracy.py
```


## Results

Include your model’s performance metrics, such as Mean Absolute Error (MAE) for both Part A and Part B of the ShanghaiTech dataset.

| Dataset Part    | MAE    |
|-----------------|--------|
| Part A (dense)  | 84.194 |
| Part B (sparse) | 11.282 |

## References

1. [CSRNet: Dilated Convolutional Neural Networks for Understanding the Highly Congested Scenes](https://arxiv.org/pdf/1802.10062)
2. [ShanghaiTech Crowd Counting Dataset](https://www.kaggle.com/datasets/tthien/shanghaitech)
