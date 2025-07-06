import random
import torch
from torch.utils.data import DataLoader
import torchvision as tv

from model.ProtoNet import ProtoNet, get_prototypes
from FewShotEpisoder import FewShotEpisoder


def evaluate(MODEL: str, DATASET: str):
  transform = tv.transforms.Compose([
    tv.transforms.Resize((224, 224)),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])  # transform

  # load model
  data = torch.load(MODEL)
  state, config = data["state"], data["config"]

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # select device
  model = ProtoNet(config).to(device)
  model.load_state_dict(state)
  model.eval()

  # create FSL episode generator
  imageset = tv.datasets.ImageFolder(root=DATASET)
  unseen_classes = [_ for _ in random.sample(list(imageset.class_to_idx.values()), config["n_way"])]
  episoder = FewShotEpisoder(imageset, unseen_classes, config["k_shot"], config["n_query"], transform)

  # compute prototype from support examples
  support_set, query_set = episoder.get_episode()
  model.prototypes = get_prototypes(support_set).to(device)
  # eval model
  total_loss, count, n_problem = 0., 0, len(query_set)
  for feature, label in DataLoader(query_set, shuffle=True, pin_memory=True, num_workers=4):
    feature, label = feature.to(device, non_blocking=True), label.to(device, non_blocking=True)
    pred = model.forward(feature)
    if torch.argmax(pred) == torch.argmax(label): count += 1
  print(f"seen classes: {data['seen_classes']}\nunseen classes: {unseen_classes}\naccuracy: {count / n_problem:.4f}({count}/{n_problem})")
# main()

if __name__ == "__main__": evaluate("../src/model/model.pth", "../data/omniglot-py/images_background/Futurama")