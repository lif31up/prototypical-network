import random
import torch.cuda
import torchvision as tv
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from config import CONFIG
from src.FewShotEpisoder import FewShotEpisoder
from src.model.ProtoNet import ProtoNet, get_prototypes
from safetensors.torch import save_file

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

if __name__ == "__main__": train("../data/omniglot-py/images_background/Futurama", "./model/model.pth")