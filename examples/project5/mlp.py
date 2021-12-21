"""Test MLP layers and window size
"""
import sys
sys.path.insert(0, "../..")
import qlib, torch, os, argparse
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('seaborn-poster')
matplotlib.style.use('ggplot')

from qlib.config import REG_CN
from qlib.utils import init_instance_by_config
from lib import *


def predict_mu_sigma(model, df, batch_size=8192):
  x_values = df.values
  sample_num = x_values.shape[0]
  preds = []
  with torch.no_grad():
    for begin in range(sample_num)[:: batch_size]:
      end = min(sample_num, begin + batch_size)
      x_batch = torch.from_numpy(x_values[begin:end]).float().to(model.device)
      preds.append(model.model(x_batch))
    preds = torch.cat(preds)
    preds[:, 1] = torch.sigmoid(preds[:, 1])
    preds = preds.cpu().detach().numpy()
  return preds[:, 0], preds[:, 1]


def main(args, model_dir):
  # model initiaiton
  args.label_expr = fetch_label(args.label_type)
  task = get_train_config(args)
  model = init_instance_by_config(task["model"])
  dataset = init_instance_by_config(task["dataset"])

  df_train, df_valid, df_test = dataset.prepare(["train", "valid", "test"],
    col_set=["feature", "label"],
    data_key="learn")
  x_train, y_train = df_train["feature"], df_train["label"]
  x_valid, y_valid = df_valid["feature"], df_valid["label"]
  x_test, y_test = df_test["feature"], df_test["label"]
  model.fit_epoch(x_train, y_train, x_valid, y_valid)
  res, _, _ = simple_backtest(model, dataset, args)

  preds = []
  xs = [x_train, x_valid, x_test]
  for i in range(3):
    mu, sigma = predict_mu_sigma(model, xs[i])
    pred = mu - sigma
    preds.append((mu, sigma, pred))

  xnames = ["mu", "sigma", "score"] # i
  names = ["Train", "Valid", "Test"] # j
  ys = [y_train, y_valid, y_test] # j
  for i in range(3):
    fig = plt.figure(figsize=(30, 7))
    for j in range(3):
      ax = plt.subplot(1, 3, j + 1)
      plot_scatter_ci(ax, preds[j][i], ys[j].values.squeeze())
      ax.set_title(f"{names[j]} Pred ({xnames[i]}) v.s. Return")
    plt.tight_layout()
    plt.savefig(f"{model_dir}/r{args.repeat_ind}_{xnames[i]}.png")
    plt.close()

  eval_result = {
    "return" : res['excess_return_without_cost'].risk['annualized_return'],
    "best_val" : model.best_score,
    "best_epoch" : model.best_epoch}
  res = {
    "model_config" : task["model"],
    "dataset_config" : task["dataset"],
    "model_param": model.model.state_dict(),
    "eval_result" : eval_result}
  torch.save(res, f"{model_dir}/r{args.repeat_ind}_expr.pth")

  return res

def fetch_label(label_type):
  if label_type == "pc-1":
    return "Ref($close,-2)/Ref($close,-1)-1"


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  # training options
  parser.add_argument("--gpu-id", default=0, type=int)
  parser.add_argument("--market", default="csi300", type=str)
  parser.add_argument("--train-start", default="2011", type=str)
  parser.add_argument("--train-end", default="2012", type=str)
  parser.add_argument("--valid-start", default="2013", type=str)
  parser.add_argument("--valid-end", default="2013", type=str)
  parser.add_argument("--test-start", default="2014", type=str)
  parser.add_argument("--test-end", default="2014", type=str)
  # architecture options
  parser.add_argument("--n-layer", default=1, type=int)
  parser.add_argument("--win-size", default=1, type=int)
  # repeat
  parser.add_argument("--repeat-ind", default=0, type=int)
  # evaluation
  parser.add_argument("--benchmark", default="SH000300", type=str)
  parser.add_argument("--name", default="raw", type=str,
    help="raw | zscorenorm")
  parser.add_argument("--dataset", default="china_stock_qlib_adj",
    help="qlib_cn_stock | china_stock_qlib_adj")
  parser.add_argument("--label-type", default="pc-1")
  args = parser.parse_args()

  provider_uri = f"../../data/{args.dataset}"
  qlib.init(provider_uri=provider_uri, region=REG_CN)

  model_dir = f"expr/{args.name}/{args.dataset}/{args.market}/l{args.n_layer}_w{args.win_size}"
  if not os.path.exists(model_dir):
    os.makedirs(model_dir)
  
  res = main(args, model_dir)