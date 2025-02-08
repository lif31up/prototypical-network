# Prototypical Network for Few-Shot Image Classification
This repository implements a Prototypical Network for few-shot image classification tasks using PyTorch. Prototypical Networks are designed to tackle the challenge of classifying new classes with limited examples by learning a metric space where classification is performed based on distances to prototype representations of each class.

### Overview
Few-shot learning aims to enable models to generalize to new classes with only a few labeled examples. Prototypical Networks achieve this by computing a prototype (mean embedding) for each class and classifying query samples based on their distances to these prototypes in the embedding space.

### Features
* PyTorch Implementation: Utilizes PyTorch for building and training the Prototypical Network.
* Configurable Parameters: Easily adjust settings such as learning rate, batch size, and number of training episodes.
* Custom Dataset Support: Adaptable to various image classification datasets formatted for few-shot learning.

## Usage
Organize your dataset into a structure compatible with PyTorch's ImageFolder:
```
dataset/
  ├── class1/
  │   ├── img1.jpg
  │   ├── img2.jpg
  │   └── ...
  ├── class2/
  │   ├── img1.jpg
  │   ├── img2.jpg
  │   └── ...
  └── ...
 ```
Update the dataset path in your configuration or script accordingly.

### Training
Run the training script with desired parameters:
```
python run.py train --dataset-path path/to/your/dataset --save-to /path/to/save/model --k-shot 2 --n-query 4 --epochs 1 --iterations 4
```
* `dataset-path`: Path to your dataset.
* `k-shot`: Number of support samples per class.
* `n-query`: Number of query samples per class.
* `epochs`: Number of episodes.
* `iters`: Number of training epochs.

### Evaluation
```
python run.py --dataset_path path/to/your/dataset --model_path path/to/saved/model.pth
```
* `testset_path`: Path to your dataset.
* `model_path`: Path to your model.

---
# Prototypical Networks: A Few-Shot Learning Approach

## Introduction
Prototypical Networks are a powerful approach for **few-shot learning**, where the goal is to classify unseen classes with very few labeled examples. The key idea behind Prototypical Networks is to learn an embedding space where each class is represented by a **prototype** (i.e., the mean of support examples in that class), and new queries are classified based on their distance to these prototypes.

## How Prototypical Networks Work
1. **Embedding Representation**: Each input image is passed through a convolutional encoder to obtain a feature embedding.
2. **Prototype Computation**: The prototype for each class is computed as the mean of the embeddings of support samples belonging to that class.
3. **Distance-Based Classification**: Query samples are classified based on the distance (usually Euclidean) to the nearest prototype.
4. **Optimization**: The network is trained to minimize the distance between query samples and their correct prototypes while maximizing the distance to incorrect ones.

## Implementation in This Repository
This repository implements a **Prototypical Network** using **PyTorch**. The main components include:
- **Encoder**: A CNN-based feature extractor that maps input images to an embedding space.
- **Prototype Computation**: Averaging support set embeddings to form class prototypes.
- **Distance Function**: Computing the similarity (using Euclidean distance) between query embeddings and class prototypes.
- **Training & Evaluation**: A pipeline for episodic training with support and query sets.

## Algorithm Breakdown
1. **Support Set & Query Set Creation**
   - Each episode consists of **N classes**, with **K support samples** and **Q query samples** per class.
   - Support set helps define class prototypes.
   - Query samples are classified using learned prototypes.

2. **Computing Class Prototypes**
   - For each class `c`, compute its prototype:
     ```math
     p_c = \frac{1}{K} \sum_{i=1}^{K} f(x_i)
     ```
     where `f(x_i)` is the embedding of the support example `x_i`.

3. **Query Classification**
   - For each query sample, compute distances to all class prototypes:
     ```math
     d(f(x_q), p_c) = ||f(x_q) - p_c||^2
     ```
   - Assign the query sample to the nearest prototype.
---
This implementation is inspired by **"Prototypical Networks for Few-Shot Learning" (Snell et al., 2017)**.

