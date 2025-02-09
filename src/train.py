import torch.cuda
import torchvision as tv
from torch import nn
from torch.utils.data import DataLoader
from src.FewShotEpisoder import FewShotEpisoder
from src.model.ProtoNet import ProtoNet
from tqdm import tqdm

def main(path, save_to, k_shot=3, n_query=3, iters=10, epochs=1):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # init device

  # create FSL episode generator
  transform = tv.transforms.Compose([
    tv.transforms.Resize((224, 224)),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ]) # transform
  imageset = tv.datasets.ImageFolder(root=path)
  episoder = FewShotEpisoder(imageset, imageset.class_to_idx.values(), k_shot, n_query, transform)

  # init learning
  model = ProtoNet(3).to(device)
  optim = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
  criterion = nn.CrossEntropyLoss()

  for _ in tqdm(range(epochs), desc="epochs/episodes"):
    support_set, query_set = episoder.get_episode()
    # compute prototype from support examples
    prototypes = list()
    embedded_features_list = [[] for _ in range(len(support_set.classes))]
    for embedded_feature, label in support_set: embedded_features_list[label].append(embedded_feature)
    for embedded_features in embedded_features_list:
      class_prototype = torch.stack(embedded_features).mean(dim=0)
      prototypes.append(class_prototype.flatten())
    prototypes = torch.stack(prototypes)
    model.prototyping(prototypes)
    for _ in tqdm(range(iters), desc="\titerations/queries"):
      total_loss = 0.0
      for feature, label in DataLoader(query_set, shuffle=True):
        loss = criterion(model.forward(feature), label)
        total_loss += loss.item()
        optim.zero_grad()
        loss.backward()
        optim.step()
      print(f"loss: {total_loss / len(query_set):.4f}")
    # for for

  # saving the model's parameters and the other data
  features = {
    "state": model.state_dict(),
    "transform": transform,
  }  # features
  torch.save(features, save_to)
# main()

if __name__ == "__main__": main(path="../data/raw/omniglot-py/images_background/Futurama", save_to="./model/model.pth")