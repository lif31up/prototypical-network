import torch.cuda
import torchvision as tv
from torch import nn
from torch.utils.data import DataLoader
from src.FewShotEpisoder import FSLEpisoder
from src.model.ProtoNet import ProtoNet

def main(path, save_to, epochs=10, iters=5):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # init device

  # create FSL episode generator
  transform = tv.transforms.Compose([
    tv.transforms.Resize((224, 224)),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ]) # transform
  imageset = tv.datasets.ImageFolder(root=path)
  episoder = FSLEpisoder(imageset, 4, 4, 4, transform)

  # init learning
  n_classes = episoder.n_classes
  model = ProtoNet(episoder.n_way).to(device)
  optim = torch.optim.Adam(model.parameters(), lr=0.001)
  criterion = nn.CrossEntropyLoss()

  for _ in range(epochs):
    support_set, query_set = episoder.get_episode()
    for _ in range(iters):
      for feature, label in DataLoader(support_set, shuffle=True):
        loss = criterion(model.forward(feature), support_set.prototypes[label])
        optim.zero_grad()
        loss.backward()
        optim.step()
      for feature, label in DataLoader(query_set, shuffle=True):
        print("", end="")
  # for for

  # saving the model's parameters and the other data
  features = {
    "state": model.state_dict(),
    "n_oupt": n_classes
  }  # features
  torch.save(features, save_to)
# main()

if __name__ == "__main__": main("../data/raw/omniglot-py/images_background/Korean", "./model/model.pth")