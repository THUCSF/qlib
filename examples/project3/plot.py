"""Test MLP layers and window size
"""
import sys
sys.path.insert(0, "../..")

import qlib, torch, os
import numpy as np
import pandas as pd
from qlib.config import REG_CN
from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.data.handler import Alpha158
from qlib.contrib.evaluate import (
  backtest as normal_backtest,
  risk_analysis,
)
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
from qlib.utils import flatten_dict

from mlp import get_train_config


provider_uri = "../../data/china_stock_qlib_adj"
qlib.init(provider_uri=provider_uri, region=REG_CN)
market = "csi300"
benchmark = "SH000300"


def get_data(df, N=5):
  index_df = df.index.to_frame()
  all_codes = index_df.instrument.unique()
  all_dates = index_df.datetime.unique()
  rng = np.random.RandomState(1)
  selected_codes = rng.choise(all_codes, (N,))
  start_date = rng.randint(0, all_dates.shape[0] - 300)
  selected_dates = all_dates[start_date:]
  return selected_codes, selected_dates


def experiment(args):
  model_dir = f"./{expr_dir}/l{n_layer}_w{winsize}"
  if not os.path.exists(model_dir):
    os.makedirs(model_dir)
  # model initiaiton
  task = get_train_config(n_layer, winsize)
  model = init_instance_by_config(task["model"])
  dataset = init_instance_by_config(task["dataset"])

  expr_dict = torch.load(f"{model_dir}/expr.pth")
  model.model.load_state_dict(expr_dict["model_param"])
  train_df, val_df, test_df = dataset.prepare(["train", "valid", "test"])

  codes, dates = get_data(train_df)
  for code in codes:
    x = torch.from_numpy(train_df.loc[code, dates]).cuda()
    y = model.model(x)


  return model, train_df, val_df, test_df



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



if __name__ == "__main__":
  ret_dic, mse_dic = {}, {}
  n_repeats = 5
  n_layers = [1, 2, 4, 8]
  win_sizes = [1, 2, 4, 8, 16, 32]

  for repeat in range(n_repeats):
    for n_layer in n_layers:
      for win_size in win_sizes:
        model, train_df, val_df, test_df = experiment(n_layer, win_size)
        break
      break
    break