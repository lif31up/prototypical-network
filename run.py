import argparse

def train(path: str, save_to: str, iters: int, k_shot: int, n_query: int, epochs: int):
  import src.train as trainer
  trainer.main(path.strip(), save_to, k_shot, n_query, iters, epochs)
# train()

def eval(model_path: str, testset_path: str):
  import src.eval as evaler
  evaler.main(model_path, testset_path)
# eval()

def main():
  parser = argparse.ArgumentParser(description="maincmd")
  parser.add_argument("--model_path", type=str, help="path of your model")
  parser.add_argument("--testset_path", type=str, help="path of your model")

  subparser = parser.add_subparsers(title="subcmd")
  # train
  parser_train = subparser.add_parser("train", help="train your model")
  parser_train.add_argument("--dataset_path", type=str, help="path to your dataset")
  parser_train.add_argument("--save_to", type=str, help="path to save your model")
  parser_train.add_argument("--k_shot", type=int, help="Number of support samples per class")
  parser_train.add_argument("--n_query", type=int, help="number of query samples per class")
  parser_train.add_argument("--iters", type=int, help="how much iteration your model does for an episode")
  parser_train.add_argument("--epochs", type=int, help="how much epochs your model does for training")
  parser_train.set_defaults(func=lambda kwargs: train(kwargs.dataset_path, kwargs.save_to, kwargs.k_shot, kwargs.n_query, kwargs.iters, kwargs.epochs))

  args = parser.parse_args()
  if hasattr(args, 'func'): args.func(args)
  elif args.model_path and args.testset_path: eval(args.model_path, args.testset_path)
  else: print("invalid argument. exiting program.")
# main():

if __name__ == "__main__": main()