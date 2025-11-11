# Prototypical Network for Few-Shot Image Classification
This implementation is inspired by:
[Prototypical Networks for Few-Shot Learning (2017)](https://arxiv.org/abs/1703.05175) by Jake Snell, Kevin Swersky, Richard S. Zemel.

Few-shot learning aims to enable models to generalize to new classes with only a few labeled examples. Prototypical Networks achieve this by computing a prototype (mean embedding) for each class and classifying query samples based on their distances to these prototypes in the embedding space.

* **Task**: Image Recognition
* **Dataset**: Omniglot Futurama Alien Alphabet

### Experiment on CoLab
<a href="https://colab.research.google.com/drive/1gsVtGvISCpXQZsKvFjLVocn89ovazusE?usp=sharing">
  <img alt="colab" src="https://upload.wikimedia.org/wikipedia/commons/thumb/d/d0/Google_Colaboratory_SVG_Logo.svg/2560px-Google_Colaboratory_SVG_Logo.svg.png" width="160"></img>
</a>

### Requirements
To run the code on your own machine, `run pip install -r requirements.txt`.
```text
tqdm>=4.67.1
```

### Configuration
`confing.py` contains the configuration settings for the model, including the framework, dimensions, learning rate, and other hyperparameters

```python
class Config:
  def __init__(self):
    self.input_channels, self.hidden_channels, self.output_channels = 1, 32, 1
    self.n_convs = 4
    self.kernel_size, self.padding, self.stride, self.bias = 3, 1, 1, True
    self.iterations, self.alpha = 50, 1e-4
    self.eps = 1e-5
    self.epochs = 10
    self.batch_size = 32
    self.n_way, self.k_shot, self.n_query = 5, 5, 5
    self.save_to = "/content/drive/MyDrive/Colab Notebooks/PRN"
    self.transform = transform
    self.imageset = get_imageset()
    self.dummy = torch.zeros(1, self.input_channels, 28, 28)
    self.clip_grad = True
```

### Training
`train.py` is a script to train the model on the omniglot dataset. It includes the training loop, evaluation, and saving the model checkpoints.

```python
if __name__ == "__main__":
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  protonet_config = Config()
  imageset = protonet_config.imageset
  seen_classes = [_ for _ in random.sample(list(imageset.class_to_idx.values()), protonet_config.n_way)]
  episoder = FewShotEpisoder(imageset, seen_classes, protonet_config.k_shot, protonet_config.n_query, protonet_config.transform)
  model = ProtoNet(protonet_config)
  train(model=model, path=protonet_config.save_to, config=protonet_config, episoder=episoder, device=device, init=True)
```

### Evaluation
`evaluate.py` is used to evaluate the trained model on the omniglot dataset. It loads the model and tokenizer, processes the dataset, and computes the accuracy of the model.

```python
if __name__ == "__main__":
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  protonet_config = Config()
  my_data = torch.load(
    f='/content/drive/MyDrive/Colab Notebooks/PRN.bin',
    weights_only=False,
    map_location=torch.device('cpu'))
  my_model = ProtoNet(my_data["config"]).to(device)
  my_model.load_state_dict(my_data["state"])
  imageset = protonet_config.imageset
  unseen_classes = [_ for _ in random.sample(list(imageset.class_to_idx.values()), protonet_config.n_way)]
  evisoder = FewShotEpisoder(imageset, unseen_classes, protonet_config.k_shot, protonet_config.n_query,
                             protonet_config.transform)
  evaluate(my_model, my_data, device)
```