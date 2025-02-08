import torch

from src.FewShotEpisoder import FewShotEpisoder
from src.model.ProtoNet import ProtoNet
import torchvision as tv

from src.utils import dist


def main(model_path: str, dataset_path: str):
  data = torch.load(model_path)
  state = data["state"]
  transform = data["transform"]

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = ProtoNet().to(device)
  model.load_state_dict(state)
  model.eval()

  imageset = tv.datasets.ImageFolder(root=dataset_path)
  episoder = FewShotEpisoder(imageset, 4, 2, transform)
  support_set, query_set = episoder.get_episode()

  prototypes = list()
  # compute prototype from support examples
  embedded_features_list = [[] for _ in range(len(support_set.classes))]
  for embedded_feature, label in support_set: embedded_features_list[label].append(embedded_feature)
  for embedded_features in embedded_features_list:
    sum = torch.zeros_like(embedded_features[0])
    for embedded_feature in embedded_features: sum += embedded_feature
    sum /= len(embedded_features)
    prototypes.append(sum.requires_grad_(True))

  # eval
  feature, c_k = query_set.prototyping(prototypes)[0]
  print(model.forward(feature, c_k))
  print(dist(feature, c_k))
# main()

if __name__ == "__main__": main("./model/model.pth", "../data/raw/omniglot-py/images_background/Korean")