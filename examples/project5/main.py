"""Test MLP layers and window size
"""
import sys
sys.path.insert(0, "../..")
import qlib, torch, os, argparse, copy, json, glob
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('seaborn-poster')
matplotlib.style.use('ggplot')

from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_logger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from qlib.config import REG_CN
from qlib.utils import init_instance_by_config
from lib import *


def main(args, model_dir):
  model_name = f"r{args.repeat_ind}_y{args.train_start}-y{args.test_end}"
  if os.path.exists(f"{model_dir}/{model_name}/model.pth"):
    print(f"=> {model_dir}/{model_name}/model.pth exists, skipped.")
    return
  args.label_expr = fetch_label(args.label_type)
  task = get_train_config(args)
  model = init_instance_by_config(task["model"]).cuda()
  learner_config = copy.deepcopy(task["learner"])
  learner_config["kwargs"]["model"] = model
  learner = init_instance_by_config(learner_config)
  dataset = init_instance_by_config(task["dataset"])

  df_train, df_valid = dataset.prepare(["train", "valid"],
    col_set=["feature", "label"],
    data_key="learn")
  x_train, y_train = df_train["feature"], df_train["label"]
  x_valid, y_valid = df_valid["feature"], df_valid["label"]
  x_train = torch.from_numpy(x_train.values).float()
  y_train = torch.from_numpy(y_train.values).float()
  x_valid = torch.from_numpy(x_valid.values).float()
  y_valid = torch.from_numpy(y_valid.values).float()
  if args.loss_type == "cls":
    y_train_orig = y_train.detach().clone()
    y_train = assign_5label(y_train)
  del df_train, df_valid

  train_dl = DataLoader(TensorDataset(x_train, y_train),
    batch_size=4096, shuffle=True, num_workers=1)
  val_dl = DataLoader(TensorDataset(x_valid, y_valid),
    batch_size=4096, shuffle=False, num_workers=1)

  mc = ModelCheckpoint(mode="max",
    save_weights_only=True,
    dirpath=f"{model_dir}/{model_name}",
    filename="{epoch}-{val_R:.6f}", monitor="val_R")
  es = EarlyStopping("lr", stopping_threshold=5e-6)
  logger = pl_logger.TensorBoardLogger(f"{model_dir}/{model_name}")
  trainer = pl.Trainer(
    logger=logger,
    max_epochs=args.n_epoch,
    progress_bar_refresh_rate=1,
    callbacks=[mc, es],
    gpus=1,
    distributed_backend='dp')
  trainer.fit(learner, train_dl, val_dl)
  best_model = glob.glob(f"{model_dir}/{model_name}/*.ckpt")[0]
  learner.load_state_dict(torch.load(best_model)["state_dict"])
  res, _, _ = simple_backtest(learner, dataset, args)

  df_test = dataset.prepare(["test"], col_set=["feature", "label"],
    data_key="infer")[0]
  x_test, y_test = df_test["feature"], df_test["label"]
  x_test = torch.from_numpy(x_test.values).float()
  y_test = torch.from_numpy(y_test.values).float()
  del df_test
  test_dl = DataLoader(TensorDataset(x_test, y_test),
    batch_size=4096, shuffle=False, num_workers=1)
  train_dl = DataLoader(TensorDataset(x_train, y_train),
    batch_size=4096, shuffle=False, num_workers=1)

  points = []
  x_dls = [train_dl, val_dl, test_dl]
  if args.loss_type == "br":
    for i in range(len(x_dls)):
      out = torch.cat(trainer.predict(learner, x_dls[i]))
      mu = torch2numpy(out[:, 0])
      sigma = torch2numpy(torch.sigmoid(out[:, 1]))
      score = torch2numpy(learner.pred2score(out))
      points.append((mu, sigma, score))

    xnames = ["mu", "sigma", "score"] # i
    names = ["Train", "Valid", "Test"] # j
    ys = [y_train_orig if args.loss_type == "cls" else y_train,
      y_valid, y_test] # j
    for i in range(3):
      plt.figure(figsize=(30, 7))
      for j in range(3):
        ax = plt.subplot(1, 3, j + 1)
        plot_scatter_ci(ax, points[j][i], ys[j].squeeze())
        ax.set_title(f"{names[j]} Pred ({xnames[i]}) v.s. Return")
      plt.tight_layout()
      plt.savefig(f"{model_dir}/{model_name}/pvr_{xnames[i]}.png")
      plt.close()
  else:
    for i in range(3):
      out = torch.cat(trainer.predict(learner, test_dl))
      score = learner.pred2score(out)
      points.append(torch2numpy(score))

    names = ["Train", "Valid", "Test"] # j
    ys = [y_train_orig if args.loss_type == "cls" else y_train,
      y_valid, y_test] # j
    plt.figure(figsize=(30, 7))
    for i in range(3):
      ax = plt.subplot(1, 3, i + 1)
      plot_scatter_ci(ax, points[i], ys[i].squeeze())
      ax.set_title(f"{names[i]} Pred v.s. Return")
    plt.tight_layout()
    plt.savefig(f"{model_dir}/{model_name}/pvr.png")
    plt.close()

  eval_result = {
    "ER" : float(res['excess_return_without_cost'].risk['annualized_return']),
    "ERC" : float(res['excess_return_with_cost'].risk['annualized_return'])}
  config = {
    "learner_config" : task["learner"],
    "model_config" : task["model"],
    "dataset_config" : task["dataset"]}
  with open(f"{model_dir}/{model_name}/config.json", "w") as f:
    json.dump(config, f)
  with open(f"{model_dir}/{model_name}/result.json", "w") as f:
    json.dump(eval_result, f)
  torch.save(learner.model.state_dict(),
    f"{model_dir}/{model_name}/model.pth") # this is the latest model

  return res


def fetch_label(label_type):
  if label_type == "pc-1":
    return "Ref($close,-2)/Ref($close,-1)-1"


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  # training options
  parser.add_argument("--n-epoch", default=100, type=int,
    help="The total training epochs.")
  parser.add_argument("--gpu-id", default="0", type=str)
  parser.add_argument("--market", default="main", type=str,
    choices=["csi300", "main"])
  parser.add_argument("--train-start", default="2011", type=str)
  parser.add_argument("--train-end", default="2012", type=str)
  parser.add_argument("--valid-start", default="2013", type=str)
  parser.add_argument("--valid-end", default="2013", type=str)
  parser.add_argument("--repeat-ind", default=0, type=int,
    help="The index of repeats (to distinguish different runs).")
  # architecture options
  parser.add_argument("--hidden-size", default=256, type=int)
  parser.add_argument("--n-layer", default=1, type=int,
    help="The number of MLP hidden layers.")
  parser.add_argument("--win-size", default=1, type=int,
    help="The input window")
  parser.add_argument("--loss-type", default="rgr", type=str,
    help="The type of loss function and output format. rgr - regression; cls - classification; br - bayesian regression",
    choices=["rgr", "cls", "br"])
  parser.add_argument("--data-type", default="raw", type=str,
    help="The data type and preprocessing method.",
    choices=["raw", "zscorenorm", "alpha158"])
  parser.add_argument("--label-type", default="pc-1",
    help="The label for prediction",
    choices=["pc-1"])
  # evaluation
  parser.add_argument("--benchmark", default="SH000300", type=str)
  parser.add_argument("--test-start", default="2014", type=str)
  parser.add_argument("--test-end", default="2014", type=str)
  args = parser.parse_args()
  set_cuda_devices(args.gpu_id)

  provider_uri = f"../../data/china_stock_qlib_adj"
  qlib.init(provider_uri=provider_uri, region=REG_CN)

  model_dir = f"expr/{args.market}_{args.data_type}_{args.label_type}/{args.loss_type}_l{args.n_layer}_w{args.win_size}"
  if not os.path.exists(model_dir):
    os.makedirs(model_dir)
  
  res = main(args, model_dir)