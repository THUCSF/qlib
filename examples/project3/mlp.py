"""Test MLP layers and window size
"""
import sys

from numpy.core.fromnumeric import argsort
sys.path.insert(0, "../..")

import qlib, torch, os, argparse
import numpy as np
import pandas as pd
from qlib.config import REG_CN
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord


def get_train_config(args):
  hidden_sizes = [256] * args.n_layer

  infer_processors = [
    {
      "class" : "DropnaProcessor",
      "kwargs" : {"fields_group": "feature"}},
      {"class" : "DropnaProcessor",
      "kwargs": {"fields_group": "label"}}
  ]
  learn_processors = []

  if "zscorenorm" in args.name:
    infer_processors.append({
      "class" : "RobustZScoreNorm",
      "kwargs" : {"fields_group": "feature", "clip_outlier": True}
    })
    learn_processors.append({
      "class" : "CSRankNorm",
      "kwargs": {"fields_group": "label"}
    })

  data_handler_config = {
    "start_time": "2008-01-01",
    "end_time": "2020-08-01",
    "fit_start_time": "2008-01-01",
    "fit_end_time": "2014-12-31",
    "instruments": market,
    "infer_processors": infer_processors,
    "learn_processors": learn_processors,
    "label": ["Ref($close, -2) / Ref($close, -1) - 1"],
    "window" : args.win_size,
  }

  task = {
    "model": {
      "class": "DNNModelPytorch",
      "module_path": "qlib.contrib.model.pytorch_nn",
      "kwargs": {
        "input_dim" : 5 * args.win_size,
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
        "GPU" : args.gpu_id,
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


def main(args):
  # model initiaiton
  task = get_train_config(args)
  model = init_instance_by_config(task["model"])
  dataset = init_instance_by_config(task["dataset"])

  model.fit(dataset)
  best_score = model.best_score

  # backtest and analysis
  port_analysis_config = get_eval_config(model, dataset)
  with R.start(experiment_name="backtest_analysis"):
    # prediction
    recorder = R.get_recorder()
    sr = SignalRecord(model, dataset, recorder)
    sr.generate()

    # backtest & analysis
    par = PortAnaRecord(recorder, port_analysis_config, "day")
    par.generate()
  
  res = par.analysis
  keys = ['mean', 'std', 'annualized_return', 'max_drawdown']
  items = ['excess_return_with_cost', 'excess_return_without_cost']
  eval_result = {
    "return" : res['excess_return_without_cost'].risk['annualized_return'],
    "mse" : -best_score}

  return {
    "model_config" : task["model"],
    "dataset_config" : task["dataset"],
    "model_param": model.model.state_dict(),
    "eval_result" : eval_result}


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--gpu-id", default=0, type=int)
  parser.add_argument("--n-layer", default=1, type=int)
  parser.add_argument("--win-size", default=1, type=int)
  parser.add_argument("--repeat-ind", default=0, type=int)
  parser.add_argument("--name", default="raw", type=str,
    help="raw | zscorenorm")
  args = parser.parse_args()

  provider_uri = "../../data/china_stock_qlib_adj"
  qlib.init(provider_uri=provider_uri, region=REG_CN)
  market = "csi300"
  benchmark = "SH000300"

  model_dir = f"expr/{args.name}/l{args.n_layer}_w{args.win_size}"
  if not os.path.exists(model_dir):
    os.makedirs(model_dir)
  
  res = main(args)
  torch.save(res, f"{model_dir}/r{args.repeat_ind}_expr.pth")
