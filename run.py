import argparse
import torchvision as tv
from src.train import train
from src.eval import evaluate

def main():
  # eval(default)
  parser = argparse.ArgumentParser(description="Few-shot learning using Prototypical Network")
  parser.add_argument("--model", type=str, help="path of your model")
  parser.add_argument("--dataset", type=str, help="path of your dataset")

  # train
  subparser = parser.add_subparsers(title="subcommands", dest="subcommand")
  parser_train = subparser.add_parser("train", help="train your model")
  parser_train.add_argument("--dataset", type=str, help="path to your dataset")
  parser_train.add_argument("--save_to", type=str, help="path to save your model")
  parser_train.add_argument("--n_way", type=int, help="number of classes per episode")
  parser_train.add_argument("--k_shot", type=int, help="number of support samples per class")
  parser_train.add_argument("--n_query", type=int, help="number of query samples per class")
  parser_train.add_argument("--iters", type=int, help="how much iteration your model does for an episode")
  parser_train.add_argument("--epochs", type=int, help="how much epochs your model does for training")
  parser_train.set_defaults(func=lambda kwargs: train(
    DATASET=kwargs.dataset,
    SAVE_TO=kwargs.save_to,
    N_WAY=kwargs.n_way,
    K_SHOT=kwargs.k_shot,
    N_QUERY=kwargs.n_query)
  ) # parser_train.set_defaults()

  # download dataset
  parser_download = subparser.add_parser("download", help="download dataset")
  parser_download.add_argument("--path", type=str, help="path to download dataset")
  parser_download.set_defaults(func=lambda kwargs: tv.datasets.Omniglot(root=kwargs.path, background=True, download=True))

  # parse logic
  args = parser.parse_args()
  if hasattr(args, 'func'): args.func(args)
  else: evaluate(MODEL=args.model, DATASET=args.dataset)
# main():

if __name__ == "__main__": main()