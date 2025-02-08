import argparse

import torchvision.datasets


def train(path: str, save_to: str, iters: int, k_shot: int, n_query: int, epochs: int):
  import src.train as trainer
  trainer.main(path=path.strip(), save_to=save_to, k_shot=k_shot, n_query=n_query, iters=iters, epochs=epochs)
# train()

def eval(model: str, path: str):
  import src.eval as evaler
  evaler.main(model=model, path=path)
# eval()

def download(path: str):
  import torchvision as tv
  tv.datasets.Omniglot(root=path, background=True, download=True)
# download()

def main():
  # eval(default)
  parser = argparse.ArgumentParser(description="maincmd")
  parser.add_argument("--path", type=str, help="path of your model")
  parser.add_argument("--model", type=str, help="path of your model")

  # train
  subparser = parser.add_subparsers(title="subcmd")
  parser_train = subparser.add_parser("train", help="train your model")
  parser_train.add_argument("--path", type=str, help="path to your dataset")
  parser_train.add_argument("--save_to", type=str, help="path to save your model")
  parser_train.add_argument("--k_shot", type=int, help="Number of support samples per class")
  parser_train.add_argument("--n_query", type=int, help="number of query samples per class")
  parser_train.add_argument("--iters", type=int, help="how much iteration your model does for an episode")
  parser_train.add_argument("--epochs", type=int, help="how much epochs your model does for training")
  parser_train.set_defaults(func=lambda kwargs: train(
    path=kwargs.dataset_path,
    save_to=kwargs.save_to,
    k_shot=kwargs.k_shot,
    n_query=kwargs.n_query,
    iters=kwargs.iters,
    epochs=kwargs.epochs)
  ) # parser_train.set_defaults()

  # download dataset
  parser_download = subparser.add_parser("download", help="download dataset")
  parser_download.add_argument("--path", type=str, help="path to download dataset")
  parser_download.set_defaults(func=lambda kwargs: download(kwargs.path))

  args = parser.parse_args()
  if hasattr(args, 'func'): args.func(args)
  elif args.path and args.model: eval(model=args.path, path=args.model)
  else: print("invalid argument. exiting program.")
# main():

if __name__ == "__main__": main()