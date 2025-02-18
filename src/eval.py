import torch
from torch import nn
from torch.utils.data import DataLoader
from src.FewShotEpisoder import FewShotEpisoder
from src.model.ProtoNet import ProtoNet
import torchvision as tv

def main(model: str, path: str, n_way=5):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # load model
  data = torch.load(model)
  state = data["state"]
  transform = data["transform"]
  model = ProtoNet().to(device)
  model.load_state_dict(state)
  model.eval()

  # create FSL episode generator
  imageset = tv.datasets.ImageFolder(root=path)
  chosen_classes = list(imageset.class_to_idx.values())[:n_way]
  episoder = FewShotEpisoder(imageset, chosen_classes, 2, 1, transform)

  # compute prototype from support examples
  support_set, query_set = episoder.get_episode()
  prototypes = list()
  embedded_features_list = [[] for _ in range(len(support_set.classes))]
  for embedded_feature, label in support_set: embedded_features_list[label].append(embedded_feature)
  for embedded_features in embedded_features_list:
    class_prototype = torch.stack(embedded_features).mean(dim=0)
    prototypes.append(class_prototype.flatten())
  prototypes = torch.stack(prototypes)
  model.prototyping(prototypes)

  # eval model
  total_loss, count, n_problem = 0., 0, len(query_set)
  criterion = nn.CrossEntropyLoss()
  for feature, label in DataLoader(query_set, shuffle=True):
    pred = model.forward(feature)
    loss = criterion(pred, label)
    total_loss += loss.item()
    if torch.argmax(pred) == torch.argmax(label): count += 1
  print(f"loss: {total_loss / len(query_set):.4f} accuracy: {count / n_problem:.4f}({count}/{n_problem})")
# main()

if __name__ == "__main__": main("./model/model.pth", "../data/raw/omniglot-py/images_background/Futurama")