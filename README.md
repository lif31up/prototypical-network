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
