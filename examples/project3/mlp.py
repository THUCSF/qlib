"""Test MLP layers and window size
"""
import sys
sys.path.insert(0, "../..")

import qlib, torch, os, argparse
from qlib.config import REG_CN
from qlib.utils import init_instance_by_config
from qlib.contrib.evaluate import risk_analysis
from qlib.backtest import backtest


def get_train_config(args):
  hidden_sizes = [256] * args.n_layer

  # non-official configurations
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
  
  # offical configurations - if specified
  if "official" in args.name:
    infer_processors = [
      {
        "class" : "DropCol", 
        "kwargs":{"col_list": ["VWAP0"]}
      },
      {
        "class" : "CSZFillna", 
        "kwargs":{"fields_group": "feature"}
      }
    ]
    learn_processors = [
      {
        "class" : "DropCol", 
        "kwargs":{"col_list": ["VWAP0"]}
      },
      {
        "class" : "DropnaProcessor", 
        "kwargs":{"fields_group": "feature"}
      },
      "DropnaLabel",
      {
        "class": "CSZScoreNorm", 
        "kwargs": {"fields_group": "label"}
      }
    ]
    data_handler_config = {
      "start_time": "2011-01-01",
      "end_time": "2020-08-01",
      "fit_start_time": "2011-01-01",
      "fit_end_time": "2014-12-31",
      "instruments": args.market,
      "infer_processors": infer_processors,
      "learn_processors": learn_processors,
      "label": ["Ref($close, -2) / Ref($close, -1) - 1"],
      "process_type": "independent"
    }
    alpha_config = {
      "class": "Alpha158",
      "module_path": "qlib.contrib.data.handler",
      "kwargs": data_handler_config,
    }
    input_size = 157
    max_steps = 8000
  else:
    data_handler_config = {
      "start_time": "2011-01-01",
      "end_time": "2020-08-01",
      "fit_start_time": "2011-01-01",
      "fit_end_time": "2014-12-31",
      "instruments": args.market,
      "infer_processors": infer_processors,
      "learn_processors": learn_processors,
      "label": ["Ref($close, -2) / Ref($close, -1) - 1"],
      "window" : args.win_size
    }
    alpha_config = {
      "class": "CustomAlpha",
      "module_path": "qlib.contrib.data.handler",
      "kwargs": data_handler_config,
    }
    input_size = 5 * args.win_size
    max_steps = 300

  task = {
    "model": {
      "class": "DNNModelPytorch",
      "module_path": "qlib.contrib.model.pytorch_nn",
      "kwargs": {
        "input_dim" : input_size,
        "output_dim" : 1,
        "layers" : hidden_sizes,
        "lr" : 0.002,
        "max_steps" : max_steps,
        "batch_size" : 8192,
        "early_stop_rounds" : 50,
        "eval_steps" : 20,
        "lr_decay" : 0.96,
        "lr_decay_steps" : 100,
        "optimizer" : "adam",
        "loss" : "mse",
        "GPU" : args.gpu_id,
        "seed" : None,
        "weight_decay" : 0.0002
      },
    },
    "dataset": {
      "class": "DatasetH",
      "module_path": "qlib.data.dataset",
      "kwargs": {
        "handler": alpha_config,
        "segments": {
          "train": ("2011-01-01", "2014-12-31"),
          "valid": ("2015-01-01", "2016-12-31"),
          "test": ("2017-01-01", "2020-08-01"),
        },
      },
    },
  }
  return task


def get_eval_config(model, dataset, args):
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
      "benchmark": args.benchmark,
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


def simple_backtest(model, dataset, args):
  # backtest and analysis
  port_analysis_config = get_eval_config(model, dataset, args)
  portfolio_metric_dict, indicator_dict = backtest(
    executor=port_analysis_config["executor"],
    strategy=port_analysis_config["strategy"],
    **port_analysis_config["backtest"])
  res = {}
  for _freq, (report_normal, positions_normal) in portfolio_metric_dict.items():
    res["excess_return_without_cost"] = risk_analysis(
      report_normal["return"] - report_normal["bench"], freq=_freq)
    res["excess_return_with_cost"] = risk_analysis(
      report_normal["return"] - report_normal["bench"] - report_normal["cost"], freq=_freq)

  return res, portfolio_metric_dict, indicator_dict


def main(args):
  # model initiaiton
  task = get_train_config(args)
  model = init_instance_by_config(task["model"])
  dataset = init_instance_by_config(task["dataset"])
  print(task["dataset"])
  print(model)

  if args.name == "official":
    model.fit(dataset)
  else:
    model.fit_epoch(dataset)
  best_score = model.best_score

  res, _, _ = simple_backtest(model, dataset, args)

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
  parser.add_argument("--market", default="csi300", type=str)
  parser.add_argument("--benchmark", default="SH000300", type=str)
  parser.add_argument("--name", default="raw", type=str,
    help="raw | zscorenorm")
  parser.add_argument("--dataset", default="china_stock_qlib_adj",
    help="qlib_cn_stock | china_stock_qlib_adj")
  args = parser.parse_args()

  provider_uri = f"../../data/{args.dataset}"
  qlib.init(provider_uri=provider_uri, region=REG_CN)

  model_dir = f"expr/{args.name}/{args.dataset}/l{args.n_layer}_w{args.win_size}"
  if not os.path.exists(model_dir):
    os.makedirs(model_dir)
  
  res = main(args)
  torch.save(res, f"{model_dir}/r{args.repeat_ind}_expr.pth")
