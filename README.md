This implementation is inspired by [**"Prototypical Networks for Few-Shot Learning"**](https://arxiv.org/abs/1703.05175) (2017) by Jake Snell, Kevin Swersky, Richard S. Zemel.

* **Note & References:** [GitBook](https://lif31up.gitbook.io/lif31up/few-shot-learning/prototypical-networks-for-few-shot-learning)
* **Quickstart on Colab:** [Colab](https://colab.research.google.com/drive/1gsVtGvISCpXQZsKvFjLVocn89ovazusE?usp=sharing)
* **Hugging Face:** [Hugging Face](https://huggingface.co/lif31up/prototypical-network)

## Prototypical Network for Few-Shot Image Classification
This repository implements a Prototypical Network for few-shot image classification tasks using PyTorch. Prototypical Networks are designed to tackle the challenge of classifying new classes with limited examples by learning a metric space where classification is performed based on distances to prototype representations of each class.

* **Task**: classifying image with few dataset.
* **Dataset**: downloaded from `torch` dataset library.

Few-shot learning aims to enable models to generalize to new classes with only a few labeled examples. Prototypical Networks achieve this by computing a prototype (mean embedding) for each class and classifying query samples based on their distances to these prototypes in the embedding space.

### Configuration

```python
CONFIG = {
  "n_way": 5,
  ...,
  "kernel_size": 3
} # CONFIG
```
### Training
```python
if __name__ == "__main__": train("../data/omniglot-py/images_background/Futurama", "./model/model.pth")
```

### Evaluation
```python
if __name__ == "__main__": evaluate("../src/model/model.pth", "../data/omniglot-py/images_background/Futurama")
```