"""Test MLP layers and window size
"""
import sys
sys.path.insert(0, "../..")

import qlib, torch, os, argparse, glob
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('seaborn-poster')
matplotlib.style.use('ggplot')

from qlib.config import REG_CN
from qlib.utils import init_instance_by_config


def get_data(df, N=5):
  index_df = df.index.to_frame()
  all_codes = index_df.instrument.unique()
  all_dates = index_df.datetime.unique()
  rng = np.random.RandomState(1)
  selected_codes = rng.choise(all_codes, (N,))
  start_date = rng.randint(0, all_dates.shape[0] - 300)
  selected_dates = all_dates[start_date:]
  return selected_codes, selected_dates


def predict_dataframe(model, df, device="cuda", batch_size=1024):
  N = df.shape[0]
  arr = df.values
  preds = []
  with torch.no_grad():
    for begin in range(N)[:: batch_size]:
      end = min(N, begin + batch_size)
      x_batch = torch.from_numpy(arr[begin:end, :-1]).float().to(device)
      pred = model.model(x_batch).squeeze()
      preds.append(pred.detach().cpu().numpy())
  return pd.Series(np.concatenate(preds), index=df.index)


def experiment(expr_dir, dataset_config, models, args):
  # model initiaiton
  dataset = init_instance_by_config(dataset_config)
  train_df, val_df, test_df = dataset.prepare(["train", "valid", "test"])
  device = f"cuda:{args.gpu_id}"

  for i, model in enumerate(models):
    y = train_df.values[:, -1]
    pred = predict_dataframe(model, train_df, device)
    mask = (y >= -0.1) & (y <= 0.1)
    plt.scatter(y[mask], pred[mask], s=1, alpha=0.1)
    plt.savefig(f"{expr_dir}/r{i}_scatter.png")
    plt.close()

  return y, pred


def str_table_single_std(dic, output_std=True):
  row_names = list(dic.keys())
  col_names = list(dic[row_names[0]].keys())
  strs = [[str(c) for c in col_names]]
  for row_name in row_names:
    if len(dic[row_name]) == 0:
      continue
    s = [str(row_name)]
    for col_name in col_names:
      if len(dic[row_name][col_name]) == 0:
        continue
      mean = dic[row_name][col_name]["mean"]
      std = dic[row_name][col_name]["std"]
      if output_std:
        item_str = f"{mean * 100:.1f} $\\pm$ {std * 100:.1f}"
      else:
        item_str = f"{mean * 100:.1f}"
      s.append(item_str)
    strs.append(s)
  return strs


def str_latex_table(strs):
  """Format a string table to a latex table.
  
  Args:
    strs : A 2D string table. Each item is a cell.
  Returns:
    A single string for the latex table.
  """
  for i in range(len(strs)):
    for j in range(len(strs[i])):
      if "_" in strs[i][j]:
        strs[i][j] = strs[i][j].replace("_", "-")

    ncols = len(strs[0])
    seps = "".join(["c" for i in range(ncols)])
    s = []
    s.append("\\begin{table}")
    s.append("\\centering")
    s.append("\\begin{tabular}{%s}" % seps)
    s.append(" & ".join(strs[0]) + " \\\\\\hline")
    for line in strs[1:]:
      s.append(" & ".join(line) + " \\\\")
    s.append("\\end{tabular}")
    s.append("\\end{table}")

    for i in range(len(strs)):
      for j in range(len(strs[i])):
        if "_" in strs[i][j]:
          strs[i][j] = strs[i][j].replace("\\_", "_")

  return "\n".join(s)


def dic_mean_std(dic):
  new_dic = {}
  for row_key in dic:
    new_dic[row_key] = {}
    for col_key in dic[row_key]:
      obs = np.array(dic[row_key][col_key])
      mean, std = obs.mean(), obs.std(ddof=1)
      new_dic[row_key][col_key] = {"mean": mean, "std": std}
  return new_dic


def set_dic(dic, key1, key2, val):
  if key1 not in dic:
    dic[key1] = {}
  if key2 not in dic[key1]:
    dic[key1][key2] = []
  dic[key1][key2].append(val)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--gpu-id", default=0, type=int)
  parser.add_argument("--name", default="raw", type=str,
    help="raw | zscorenorm")
  parser.add_argument("--dataset", default="china_stock_qlib_adj",
    help="qlib_cn_stock | china_stock_qlib_adj")
  args = parser.parse_args()

  provider_uri = f"../../data/{args.dataset}"
  qlib.init(provider_uri=provider_uri, region=REG_CN)

  expr_dir = f"expr/{args.name}"

  model_dirs = glob.glob(f"{expr_dir}/{args.dataset}/*")
  dic = {}
  for model_dir in model_dirs:
    expr_names = glob.glob(f"{model_dir}/*.pth")
    models = []
    for expr_name in expr_names:
      print(expr_name)
      expr_res = torch.load(expr_name)
      # load model
      #model = init_instance_by_config(expr_res["model_config"])
      #model.model.load_state_dict(expr_res["model_param"])
      #model.model = model.model.to(f"cuda:{args.gpu_id}")
      #models.append(model)
      # load evaluation results
      res = expr_res["eval_result"]
      args = expr_name[expr_name.rfind("/")+1:].strip().split("_")
      args = [a[1:] for a in args]
      set_dic(dic, args[0], args[1], res["return"])
    #dataset_config = expr_res["dataset_config"]

  std_dic = dic_mean_std(dic)
  with open(f"{expr_dir}/{args.dataset}/res.tex", "w") as f:
    f.write(str_latex_table(str_table_single_std(std_dic)))
