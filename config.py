import torch
import torchvision as tv

class Config:
  def __init__(self):
    self.input_channels, self.hidden_channels, self.output_channels = 1, 32, 1
    self.n_convs = 4
    self.kernel_size, self.padding, self.stride, self.bias = 3, 1, 1, True
    self.iterations, self.alpha = 50, 1e-4
    self.eps = 1e-5
    self.epochs = 10
    self.batch_size = 32
    self.n_way, self.k_shot, self.n_query = 5, 5, 5
    self.save_to = "/content/drive/MyDrive/Colab Notebooks/PRN"
    self.transform = transform
    self.imageset = get_imageset()
    self.dummy = torch.zeros(1, self.input_channels, 28, 28)
    self.clip_grad = True
  # __init__():
# MAMLConfig

transform = tv.transforms.Compose([
  tv.transforms.Resize((28, 28)),
  tv.transforms.Grayscale(num_output_channels=1),
  tv.transforms.ToTensor(),
  tv.transforms.Normalize(mean=[0.5], std=[0.5]),
]) # transform

def get_imageset():
  tv.datasets.Omniglot(root="./data/", background=True, download=True)
  return tv.datasets.ImageFolder(root="./data/omniglot-py/images_background/Futurama")
# _get_imageset()