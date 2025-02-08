import torch.cuda
import torchvision as tv
from torch import nn
from torch.utils.data import DataLoader
from src.FewShotEpisoder import FewShotEpisoder
from src.model.ProtoNet import ProtoNet
from tqdm import tqdm

def main(path, save_to, epochs=10, iters=5):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # init device

  # create FSL episode generator
  transform = tv.transforms.Compose([
    tv.transforms.Resize((224, 224)),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ]) # transform
  imageset = tv.datasets.ImageFolder(root=path)
  episoder = FewShotEpisoder(imageset, 2, 1, transform)

  # init learning
  model = ProtoNet().to(device)
  optim = torch.optim.Adam(model.parameters(), lr=0.001)
  criterion = nn.CrossEntropyLoss()

  for _ in tqdm(range(epochs)):
    support_set, query_set = episoder.get_episode()
    model.q_points = support_set.prototypes
    loss = 0.
    for _ in range(iters):
      for feature, label in DataLoader(query_set, shuffle=True):
        loss = criterion(model.forward(feature), support_set.prototypes[label])
        optim.zero_grad()
        loss.backward()
        optim.step()
      # for
      print(f"loss: {loss:.4f}")
  # for for

  # saving the model's parameters and the other data
  features = {
    "state": model.state_dict(),
  }  # features
  torch.save(features, save_to)
# main()

if __name__ == "__main__": main("../data/raw/omniglot-py/images_background/Korean", "./model/model.pth")