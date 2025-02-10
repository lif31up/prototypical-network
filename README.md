`torch` `torchvision` `tqdm`

This implementation is inspired by **"Prototypical Networks for Few-Shot Learning" (Snell et al., 2017)**.
* **task**: classifying image with few dataset.
* **dataset**: downloaded from `torch` dataset library.

## Prototypical Network for Few-Shot Image Classification
This repository implements a Prototypical Network for few-shot image classification tasks using PyTorch. Prototypical Networks are designed to tackle the challenge of classifying new classes with limited examples by learning a metric space where classification is performed based on distances to prototype representations of each class.

Few-shot learning aims to enable models to generalize to new classes with only a few labeled examples. Prototypical Networks achieve this by computing a prototype (mean embedding) for each class and classifying query samples based on their distances to these prototypes in the embedding space.

<a href="https://colab.research.google.com/drive/1gsVtGvISCpXQZsKvFjLVocn89ovazusE?usp=sharing">Test Result on Colab</a>

## Instruction
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

### Training
Run the training script with desired parameters:
```
python run.py train --dataset_path path/to/your/dataset --save_to /path/to/save/model --n_way 5 --k_shot 2 --n_query 4 --epochs 1 --iters 4
```
* `dataset_path`: Path to your dataset.
* `save_to`: path to save the trained model.
* `n_way`: number of classes in each episode.
* `k_shot`: Number of support samples per class.
* `n-_query`: Number of query samples per class.
* `epochs`: Number of episodes.
* `iters`: Number of training epochs.

### Evaluation
```
python run.py --path path/to/your/dataset --model path/to/saved/model.pth --n_way 5
```
* `path`: Path to your dataset.
* `model`: Path to your model.
* `n_way`: Number of classes in each episode.

### Download Omniglot Dataset
```
pyhton download --path
```
* `path`: Path to your dataset.

---
### More Explanation
Prototypical Networks are a powerful approach for **few-shot learning**, where the goal is to classify unseen classes with very few labeled examples. The key idea behind Prototypical Networks is to learn an embedding space where each class is represented by a **prototype** (i.e., the mean of support examples in that class), and new queries are classified based on their distance to these prototypes.

* **Embedding Representation with CNN**: Each input image is passed through a convolutional encoder to obtain a feature embedding.
* **Prototype Computation**: The prototype for each class is computed as the mean of the embeddings of support samples belonging to that class.
* **Distance-Based Classification**: Query samples are classified based on the distance (using `torch.cdist`) to the nearest prototype.
* **Optimization**: The network is trained to minimize the distance between query samples and their correct prototypes while maximizing the distance to incorrect ones.