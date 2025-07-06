import random
import torch.cuda
import torchvision as tv
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from config import CONFIG
from FewShotEpisoder import FewShotEpisoder
from model.ProtoNet import ProtoNet, get_prototypes
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
  criterion, optim = nn.CrossEntropyLoss(), torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
  scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lambda step: min((step + 1) ** -0.5, (step + 1) * 1e-3))

  progress_bar, whole_loss = tqdm(range(CONFIG["epochs"])), float(0)
  for _ in progress_bar:
    support_set, query_set = episoder.get_episode()
    model.prototypes, epoch_losses = get_prototypes(support_set).to(device), float(0)
    for _ in range(CONFIG["iters"]):
      iter_loss = float(0)
      for feature, label in DataLoader(query_set, batch_size=CONFIG["batch_size"], shuffle=True, pin_memory=True, num_workers=4):
        feature, label = feature.to(device, non_blocking=True), label.to(device, non_blocking=True)
        pred = model.forward(feature)
        loss = criterion(pred, label)
        iter_loss += loss.item()
        optim.zero_grad()
        loss.backward()
        if config["clip_grad"]: nn.utils.clip_grad_norm(model.parameters(), max_norm=1.0)
        optim.step()
        scheduler.step()
      # for DataLoader(query_set)
      epoch_losses += iter_loss / len(query_set)
    # for range(iters)
    epoch_losses = epoch_losses / CONFIG["iters"]
    progress_bar.set_postfix(loss=epoch_losses)
  # for progress_bar

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