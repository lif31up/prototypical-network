import torch.cuda
import torchvision as tv
from torch import nn
from torch.utils.data import DataLoader
from src.FewShotEpisoder import FewShotEpisoder
from src.model.ProtoNet import ProtoNet
from tqdm import tqdm

def main(path, save_to, k_shot, n_query, iters=10, epochs=1):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # init device

  # create FSL episode generator
  transform = tv.transforms.Compose([
    tv.transforms.Resize((224, 224)),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ]) # transform
  imageset = tv.datasets.ImageFolder(root=path)
  episoder = FewShotEpisoder(imageset, 3, 3, transform)

  # init learning
  model = ProtoNet().to(device)
  optim = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
  criterion = nn.CrossEntropyLoss()

  for _ in tqdm(range(epochs), desc="epochs/episodes"):
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
    for _ in tqdm(range(iters), desc="\titerations/queries"):
      # update loss
      loss = float()
      for feature, label in DataLoader(query_set, shuffle=True):
        loss = criterion(model.forward(feature), label.squeeze(dim=0))
        optim.zero_grad()
        loss.backward()
        optim.step()
      print(f"{loss:.4f}")
  # for for

  # saving the model's parameters and the other data
  features = {
    "state": model.state_dict(),
    "transform": transform,
  }  # features
  torch.save(features, save_to)
# main()

if __name__ == "__main__": main("../data/raw/omniglot-py/images_background/Futurama", "./model/model.pth")