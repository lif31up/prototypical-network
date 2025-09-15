This implementation is inspired by [**"Prototypical Networks for Few-Shot Learning"**](https://arxiv.org/abs/1703.05175) (2017) by Jake Snell, Kevin Swersky, Richard S. Zemel.

* **Note & References:** [GitBook](https://lif31up.gitbook.io/lif31up/few-shot-learning/prototypical-networks-for-few-shot-learning)
* **Quickstart on Colab:** [Colab](https://colab.research.google.com/drive/1gsVtGvISCpXQZsKvFjLVocn89ovazusE?usp=sharing)
* **Hugging Face:** [Hugging Face](https://huggingface.co/lif31up/prototypical-network)

**Results:**

|                     | 5w5s ACC         |
|---------------------|------------------|
| `omniglot futurmaa` | `83% (833/1000)` |

## Prototypical Network for Few-Shot Image Classification
This repository implements a Prototypical Network for few-shot image classification tasks using PyTorch. Prototypical Networks are designed to tackle the challenge of classifying new classes with limited examples by learning a metric space where classification is performed based on distances to prototype representations of each class.

* **Task**: classifying image with few dataset.
* **Dataset**: downloaded from `torch` dataset library.

Few-shot learning aims to enable models to generalize to new classes with only a few labeled examples. Prototypical Networks achieve this by computing a prototype (mean embedding) for each class and classifying query samples based on their distances to these prototypes in the embedding space.

### Configuration
`confing.py` contains the configuration settings for the model, including the framework, dimensions, learning rate, and other hyperparameters

```python
class Config:
  def __init__(self):
    self.input_channels, self.hidden_channels, self.output_channels = 1, 32, 1
    self.n_convs = 4
    self.kernel_size, self.padding, self.stride, self.bias = 3, 1, 1, True
    self.iterations, self.alpha = 100, 1e-3
    self.eps = 1e-5
    self.epochs, self.beta = 30, 1e-4
    self.batch_size = 8
    self.n_way, self.k_shot, self.n_query = 5, 5, 5
    self.save_to = "./models"
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
`eval.py` is used to evaluate the trained model on the omniglot dataset. It loads the model and tokenizer, processes the dataset, and computes the accuracy of the model.

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

## Technical Highlights

### Prototyping 
It optimizes the embedding space to create distinct class prototypes. These prototypes are calculated using mean values and are resampled during each iteration.
```python
def get_prototypes(support_set):
  prototypes = list()
  embedded_features_list = [[] for _ in range(len(support_set.classes))]
  for embedded_feature, label in support_set:
    idx = support_set.classes.index(label)
    embedded_features_list[idx].append(embedded_feature)
  for embedded_features in embedded_features_list:
    class_prototype = torch.stack(embedded_features).mean(dim=0)
    prototypes.append(class_prototype.flatten())
  prototypes = torch.stack(prototypes)
  return prototypes
```

### Euclidean Distance / Model Definition
The model architecture doesn't use any pooling layers but instead employs residual connections. The use of residual connections in Few Shot Learning approaches like Prototypical Networks has been proven to stabilize the learning process.
```python
class ProtoNet(nn.Module):
  def forward(self, x):
    assert self.prototypes is not None, "self.prototypes is None"
    x = self.convs[0](x)
    x = self.act(x)
    for conv in self.convs[1:-1]:
      res = x
      x = conv(x)
      x = self.act(x)
      x += res
    x = self.convs[-1](x)
    x = self.cdist(x, self.prototypes)
    return torch.negative(x)

  def cdist(self, x, prototypes):
    flatten_x = self.flat(x)
    return torch.cdist(flatten_x, prototypes, p=2)
```

### Training
I must say the training code is very well structured. It consists of meta-learning and basic learning stages. In the meta-learning stage, it calculates the prototypes, while in the basic learning stage, it learns toward these prototypes.
```python
def train(model, path, config:Config, episoder:FewShotEpisoder, device, init=True):
  model.to(device)
  if init: model.apply(init_weights)
  optim = torch.optim.Adam(model.parameters(), lr=config.alpha, eps=config.eps)
  criterion = nn.CrossEntropyLoss(reduction="sum")

  progression = tqdm(range(config.epochs))
  for _ in progression:
    epoch_loss = float(0)
    support_set, query_set = episoder.get_episode()
    model.get_prototypes(support_set)
    for _ in range(config.iterations):
      iter_loss = float(0)
      for feature, label in DataLoader(query_set, batch_size=config.batch_size, shuffle=True, pin_memory=True, num_workers=4):
        feature, label = feature.to(device, non_blocking=True), label.to(device, non_blocking=True)
        pred = model.forward(feature)
        loss = criterion(pred, label)
        optim.zero_grad()
        loss.backward()
        if config.clip_grad: nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()
        iter_loss += loss.item()
      iter_loss /= len(query_set)
      progression.set_postfix(iter_loss=iter_loss)

  features = {
    "state": model.state_dict(),
    "config": config,
  } # features
  torch.save(features, f'{path}.bin')
```
