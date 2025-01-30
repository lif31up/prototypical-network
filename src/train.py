import torch.cuda
import torchvision as tv
from torch import nn
from torch.utils.data import DataLoader
from data.FewShotDataset import FewShotEpisoder
from model.ProtoNet import ProtoNet

def main(path, save_to, epochs=10):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # init device
  # create FSL episode generator
  transform = tv.transforms.Compose([
    tv.transforms.Resize((224, 224)),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ]) # transform
  imageset = tv.datasets.ImageFolder(root=path)
  episoder = FewShotEpisoder(imageset, 4, 4, 4, transform)
  # init learning
  model = ProtoNet(1, 64, 64).to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
  criterion = nn.CrossEntropyLoss()
  # define learning step:
  global loss
  for epoch in range(epochs):
    for support_set, query_set in episoder:
      support_loader, query_loader = DataLoader(support_set), DataLoader(query_set)
      # train algorithm
      for x, y in support_loader:
        loss = criterion(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
      # for
      # evaluation algorithm
      for x, y in query_loader: loss = criterion(model(x), y)
      print(loss)
      # for
  # for
  # saving the model's parameters and the other data
  features = {
    "state": model.state_dict(),
  }  # features
  torch.save(features, save_to)
# main()

if __name__ == "__main__": main("./data/raw/omniglot-py/images_background/Korean", "")