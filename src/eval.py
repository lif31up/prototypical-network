import torch
from torch import nn
from torch.utils.data import DataLoader
from src.FewShotEpisoder import FewShotEpisoder
from src.model.ProtoNet import ProtoNet
import torchvision as tv


def main(model_path: str, dataset_path: str):
  data = torch.load(model_path)
  state = data["state"]
  transform = data["transform"]

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = ProtoNet().to(device)
  model.load_state_dict(state)
  model.eval()

  imageset = tv.datasets.ImageFolder(root=dataset_path)
  episoder = FewShotEpisoder(imageset, 3, 3, transform)

  # compute prototype from support examples
  support_set, query_set = episoder.get_episode()
  # compute prototype from support examples
  prototypes = list()
  embedded_features_list = [[] for _ in range(len(support_set.classes))]
  for embedded_feature, label in support_set: embedded_features_list[label].append(embedded_feature)
  for embedded_features in embedded_features_list:
    sum = torch.zeros_like(embedded_features[0])
    for embedded_feature in embedded_features: sum += embedded_feature
    sum /= len(embedded_features)
    prototypes.append(sum.flatten())
  prototypes = torch.stack(prototypes)
  model.prototyping(prototypes)
  # eval
  loss = float()
  criterion = nn.CrossEntropyLoss()
  for feature, label in DataLoader(query_set, shuffle=True):
    loss = criterion(model.forward(feature), label.squeeze(dim=0))
  print(f"loss: {loss:.4f}")
# main()

if __name__ == "__main__": main("./model/model.pth", "../data/raw/omniglot-py/images_background/Futurama")