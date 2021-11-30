"""Test MLP layers and window size
"""
import sys
sys.path.insert(0, "../..")

import qlib
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


provider_uri = "../../data/china_stock_qlib_adj"
qlib.init(provider_uri=provider_uri, region=REG_CN)
market = "csi300"
benchmark = "SH000300"
device_id = 9


def get_train_config(n_layer, winsize):
  hidden_sizes = [256] * n_layer
  data_handler_config = {
    "start_time": "2008-01-01",
    "end_time": "2020-08-01",
    "fit_start_time": "2008-01-01",
    "fit_end_time": "2014-12-31",
    "instruments": market,
    "infer_processors": [
        {"class" : "DropnaProcessor", "kwargs": {"fields_group": "feature"}},
        {"class" : "DropnaProcessor", "kwargs": {"fields_group": "label"}},
    ],
    "learn_processors": [
        {"class" : "DropnaProcessor", "kwargs": {"fields_group": "feature"}},
        {"class" : "DropnaProcessor", "kwargs": {"fields_group": "label"}},
    ],
    "label": ["Ref($close, -2) / Ref($close, -1) - 1"],
    "window" : winsize,
    "process_type": "independent"
  }

  task = {
    "model": {
      "class": "DNNModelPytorch",
      "module_path": "qlib.contrib.model.pytorch_nn",
      "kwargs": {
        "input_dim" : 5 * winsize,
        "output_dim" : 1,
        "layers" : hidden_sizes,
        "lr" : 0.001,
        "max_steps" : 300,
        "batch_size" : 4096,
        "early_stop_rounds" : 50,
        "eval_steps" : 20,
        "lr_decay" : 0.96,
        "lr_decay_steps" : 100,
        "optimizer" : "adam",
        "loss" : "mse",
        "GPU" : device_id,
        "seed" : None,
        "weight_decay" : 1e-4
      },
    },
    "dataset": {
      "class": "DatasetH",
      "module_path": "qlib.data.dataset",
      "kwargs": {
        "handler": {
          "class": "CustomAlpha",
          "module_path": "qlib.contrib.data.handler",
          "kwargs": data_handler_config,
        },
        "segments": {
          "train": ("2008-01-01", "2014-12-31"),
          "valid": ("2015-01-01", "2016-12-31"),
          "test": ("2017-01-01", "2020-08-01"),
        },
      },
    },
  }
  return task


def get_eval_config(model, dataset):
  port_analysis_config = {
    "executor": {
      "class": "SimulatorExecutor",
      "module_path": "qlib.backtest.executor",
      "kwargs": {
        "time_per_step": "day",
        "generate_portfolio_metrics": True,
      },
    },
    "strategy": {
      "class": "TopkDropoutStrategy",
      "module_path": "qlib.contrib.strategy.signal_strategy",
      "kwargs": {
        "model": model,
        "dataset": dataset,
        "topk": 50,
        "n_drop": 5,
      },
    },
    "backtest": {
      "start_time": "2017-01-01",
      "end_time": "2020-08-01",
      "account": 100000000,
      "benchmark": benchmark,
      "exchange_kwargs": {
        "freq": "day",
        "limit_threshold": 0.095,
        "deal_price": "close",
        "open_cost": 0.0005,
        "close_cost": 0.0015,
        "min_cost": 5,
      },
    },
  }
  return port_analysis_config


def experiment(n_layer, winsize):
  task = get_train_config(n_layer, winsize)

  # model initiaiton
  model = init_instance_by_config(task["model"])
  dataset = init_instance_by_config(task["dataset"])

  # start exp to train model
  with R.start(experiment_name="train_model"):
    R.log_params(**flatten_dict(task))
    model.fit(dataset)
    best_score = model.best_score
    R.save_objects(trained_model=model)
    rid = R.get_recorder().id

  port_analysis_config = get_eval_config(model, dataset)

  # backtest and analysis
  with R.start(experiment_name="backtest_analysis"):
    recorder = R.get_recorder(recorder_id=rid, experiment_name="train_model")
    model = recorder.load_object("trained_model")

    # prediction
    recorder = R.get_recorder()
    ba_rid = recorder.id
    sr = SignalRecord(model, dataset, recorder)
    sr.generate()

    # backtest & analysis
    par = PortAnaRecord(recorder, port_analysis_config, "day")
    par.generate()
  
  res = par.analysis
  print(res)
  keys = ['mean', 'std', 'annualized_return', 'max_drawdown']
  items = ['excess_return_with_cost', 'excess_return_without_cost']
  return {
    "return" : res['excess_return_without_cost'].risk['annualized_return'],
    "mse" : -best_score}


def str_table_single_std(dic, output_std=True):
  row_names = list(dic.keys())
  col_names = list(dic[row_names[0]].keys())
  strs = [[str(c) for c in col_names]]
  for row_name in row_names:
    if len(dic[row_name]) == 0:
      continue
    s = [row_name]
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
  for n_layer in n_layers:
    ret_dic[n_layer] = {}
    mse_dic[n_layer] = {}
    for win_size in win_sizes:
      ret_dic[n_layer][win_size] = []
      mse_dic[n_layer][win_size] = []

  for repeat in range(n_repeats):
    for n_layer in n_layers:
      for win_size in win_sizes:
        dic = experiment(n_layer, win_size)
        ret_dic[n_layer][win_size].append(dic["return"])
        mse_dic[n_layer][win_size].append(dic["mse"])

  ret_dic_agg = dic_mean_std(ret_dic)
  mse_dic_agg = dic_mean_std(mse_dic)
  with open("./ret.tex", "w") as f:
    f.write(str_latex_table(str_table_single_std(ret_dic_agg)))
  with open("./mse.tex", "w") as f:
    f.write(str_latex_table(str_table_single_std(mse_dic_agg)))