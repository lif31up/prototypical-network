This implementation is inspired by [**"Prototypical Networks for Few-Shot Learning"**](https://arxiv.org/abs/1703.05175) (2017) by Jake Snell, Kevin Swersky, Richard S. Zemel.

* **Note & References:** [GitBook](https://lif31up.gitbook.io/lif31up/few-shot-learning/prototypical-networks-for-few-shot-learning)
* **Quickstart on Colab:** [Colab](https://colab.research.google.com/drive/1gsVtGvISCpXQZsKvFjLVocn89ovazusE?usp=sharing)
* **Hugging Face:** [Hugging Face](https://huggingface.co/lif31up/prototypical-network)

|            | 5 Way ACC (5 shot) | 5 Way ACC(1 shot) |
|------------|--------------------|------------------|
|**Omniglot**|`100%` **(100/100)**|`96%` **(96/100)**|

## Prototypical Network for Few-Shot Image Classification
This repository implements a Prototypical Network for few-shot image classification tasks using PyTorch. Prototypical Networks are designed to tackle the challenge of classifying new classes with limited examples by learning a metric space where classification is performed based on distances to prototype representations of each class.

* **Task**: classifying image with few dataset.
* **Dataset**: downloaded from `torch` dataset library.

Few-shot learning aims to enable models to generalize to new classes with only a few labeled examples. Prototypical Networks achieve this by computing a prototype (mean embedding) for each class and classifying query samples based on their distances to these prototypes in the embedding space.

### Configuration
`confing.py` contains the configuration settings for the model, including the framework, dimensions, learning rate, and other hyperparameters

```python
CONFIG = {
  "version": "1.0.1",
  # framework
  "n_way": 5,
  "k_shot": 1,
  "n_query": 2,
  # model
  "inpt_dim": 3,
  "hidn_dim": 6,
  "oupt_dim": 5,
  # hp
  "iters": 5,
  "epochs": 10,
  "batch_size": 8,
  "inner_batch_size": 5,
  "alpha": 1e-2,
  "beta": 1e-4,
} # CONFIG
```

### Training
`train.py` is a script to train the model on the omniglot dataset. It includes the training loop, evaluation, and saving the model checkpoints.

```python
if __name__ == "__main__": train("../data/omniglot-py/images_background/Futurama", "./model/model.pth")
```

### Evaluation
`eval.py` is used to evaluate the trained model on the omniglot dataset. It loads the model and tokenizer, processes the dataset, and computes the accuracy of the model.

```python
if __name__ == "__main__": evaluate("../src/model/model.pth", "../data/omniglot-py/images_background/Futurama")
```

## Technical Highlights

### Prototyping

```python
def get_prototypes(support_set, seen_classes):
  prototypes = []
  embedded_features_list = [[] for _ in range(len(support_set.classes))]
  for embedded_feature, label in support_set: embedded_features_list[seen_classes.index(label)].append(embedded_feature)
  for embedded_features in embedded_features_list:
    class_prototype = torch.stack(embedded_features).mean(dim=0)
    prototypes.append(class_prototype)
  return torch.stack(prototypes)
# get_prototypes
```

### Euclidean Distance / Model Definition

```python
class ProtoNet(nn.Module):
  def __init__(self, config):
    super(ProtoNet, self).__init__()
    self.config, self.prototypes = config, None
    self.flatten, self.act, self.softmax = nn.Flatten(1), nn.SiLU(), nn.Softmax(dim=1)

    self.conv1 = nn.Conv2d(in_channels=config["in_channels"], out_channels=config["hidden_channels"], kernel_size=config["kernel_size"], stride=1, padding=1)
    self.conv2 = nn.Conv2d(in_channels=config["hidden_channels"], out_channels=config["hidden_channels"], kernel_size=config["kernel_size"], stride=1, padding=1)
    self.conv3 = nn.Conv2d(in_channels=config["hidden_channels"], out_channels=config["out_channels"], kernel_size=config["kernel_size"], stride=1, padding=1)
  # __init__():

  def forward(self, x):
    assert self.prototypes is not None, "self.prototypes is None"
    x = self.conv1(x)
    x = self.conv2(self.act(x))
    x = self.conv3(self.act(x))
    x = self.cdist(x, self.prototypes)
    return self.softmax(x)
  # forward():

  def cdist(self, x, prototypes):
    flatten_x = self.flatten(x)
    return torch.cdist(flatten_x, prototypes, p=2)
  # cdist():
# ProtoNet
```

### Training
```python
def train(DATASET, SAVE_TO, config=CONFIG):
  transform = tv.transforms.Compose([
    tv.transforms.Resize((224, 224)),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ]) # transform

  imageset = tv.datasets.ImageFolder(root=DATASET)
  seen_classes = [_ for _ in random.sample(list(imageset.class_to_idx.values()), config['n_way'])]
  episoder = FewShotEpisoder(imageset, seen_classes, config['k_shot'], config['n_query'], transform)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = ProtoNet(config).to(device)
  optim = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
  criterion = nn.CrossEntropyLoss()
  scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lambda step: min((step + 1) ** -0.5, (step + 1) * 1e-3))

  progress_bar, whole_loss = tqdm(range(config['epochs'])), float()
  for _ in progress_bar:
    support_set, query_set = episoder.get_episode()
    # STAGE1: compute prototype from support examples
    prototypes = get_prototypes(support_set, seen_classes)
    # STAGE2: update parameters form loss associated with prototypes
    epoch_loss = list()
    for _ in range(config['iters']):
      iter_loss, vuffer = list(), 0
      for feature, label in DataLoader(query_set, batch_size=config['batch_size'], shuffle=True):
        pred = model.forward(feature, prototypes)
        loss = criterion(pred, label)
        optim.zero_grad()
        loss.backward()
        optim.step()
        scheduler.step()
        iter_loss.append(loss.item())
      vuffer = sum(iter_loss) / len(iter_loss)
      progress_bar.set_postfix(iter_loss=vuffer)
      epoch_loss.append(vuffer)
    progress_bar.set_postfix(loss=sum(epoch_loss) / len(epoch_loss))
  # for for for

  # saving model
  features = {
    "state": model.state_dict(),
    "config": config,
    "seen_classes": seen_classes
  }  # feature
  torch.save(features, SAVE_TO)
  save_file(model.state_dict(), SAVE_TO.replace(".pth", ".safetensors"))
# main()
```
