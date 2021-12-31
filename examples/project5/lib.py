"""The helper functions.
"""
import torch, os
import numpy as np
import pandas as pd

from qlib.contrib.evaluate import risk_analysis
from qlib.backtest import backtest
from qlib.backtest.signal import ModelSignal


def set_cuda_devices(device_ids, use_cuda=True):
  """Sets visible CUDA devices.

  Example:

  set_cuda_devices('0,1', True)  # Enable device 0 and 1.
  set_cuda_devices('3', True)  # Enable device 3 only.
  set_cuda_devices('all', True)  # Enable all devices.
  set_cuda_devices('-1', True)  # Disable all devices.
  set_cuda_devices('0', False)  # Disable all devices.

  Args:
    devices_ids: A string, indicating all visible devices. Separated with comma.
      To enable all devices, set this field as `all`.
    use_cuda: Whether to use cuda. If set as False, all devices will be
      disabled. (default: True)
  """
  if not use_cuda:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    return 0
  assert isinstance(device_ids, str)
  if device_ids.lower() == 'all':
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
      del os.environ['CUDA_VISIBLE_DEVICES']
    return 8
  os.environ['CUDA_VISIBLE_DEVICES'] = device_ids.replace(' ', '')
  return len(device_ids.split(","))


def torch2numpy(x):
  if type(x) is float:
    return x
  return x.detach().cpu().numpy()


def simple_backtest(model, dataset, args):
  """backtest and analysis
  Returns:
    res: The return dictionary.
    portfolio_metric_dict, indicator_dict: The raw results.
  """
  port_analysis_config = get_eval_config(model, dataset, args)
  portfolio_metric_dict, indicator_dict = backtest(
    executor=port_analysis_config["executor"],
    strategy=port_analysis_config["strategy"],
    **port_analysis_config["backtest"])
  res = {}
  for _freq, (report_normal, positions_normal) in portfolio_metric_dict.items():
    res["ER"] = risk_analysis(
      report_normal["return"] - report_normal["bench"], freq=_freq)
    res["ERC"] = risk_analysis(
      report_normal["return"] - report_normal["bench"] - report_normal["cost"], freq=_freq)
    res["benchmark"] = risk_analysis(report_normal["bench"], freq=_freq)

  keys = ['mean', 'std', 'annualized_return', 'max_drawdown']
  items = ['excess_return_with_cost', 'excess_return_without_cost']

  return res, portfolio_metric_dict, indicator_dict


def assign_5label(x):
  """Convert price change into 5 classes."""
  label = torch.zeros_like(x).long()
  label[x > -0.05] = 1
  label[x > -0.01] = 2
  label[x > 0.01] = 3
  label[x > 0.05] = 4
  return label


def get_train_config(args):
  """
  Returns:
    task: The config in qlib format.
  """
  hidden_sizes = [args.hidden_size] * args.n_layer

  # non-official configurations
  infer_processors = [
    {
      "class" : "DropnaProcessor",
      "kwargs" : {"fields_group": "feature"}},
      {"class" : "DropnaProcessor",
      "kwargs": {"fields_group": "label"}}
  ]
  learn_processors = []
  if "zscorenorm" in args.data_type:
    infer_processors.append({
      "class" : "RobustZScoreNorm",
      "kwargs" : {"fields_group": "feature", "clip_outlier": True}
    })
    infer_processors.append({
      "class" : "RobustZScoreNorm",
      "kwargs" : {"fields_group": "label", "clip_outlier": True}
    })

  # offical configurations - alpha158
  if "alpha158" in args.data_type: 
    infer_processors = [
      {
        "class" : "DropCol", 
        "kwargs":{"col_list": ["VWAP0"]}
      },
      {
        "class" : "CSZFillna", 
        "kwargs":{"fields_group": "feature"}
      }]
    learn_processors = [
      {
        "class" : "DropCol", 
        "kwargs":{"col_list": ["VWAP0"]}
      },
      {
        "class" : "DropnaProcessor", 
        "kwargs":{"fields_group": "feature"}
      }]
    data_handler_config = {
      "start_time": f"{args.train_start}-01-01",
      "end_time": f"{args.test_end}-12-31",
      "fit_start_time": f"{args.train_start}-01-01",
      "fit_end_time": f"{args.train_end}-12-31",
      "instruments": args.market,
      "infer_processors": infer_processors,
      "learn_processors": learn_processors,
      "label": [args.label_expr],
      "process_type": "independent"
    }
    alpha_config = {
      "class": "Alpha158",
      "module_path": "qlib.contrib.data.handler",
      "kwargs": data_handler_config,
    }
    input_size = 157
  else:
    data_handler_config = {
      "start_time": f"{args.train_start}-01-01",
      "end_time": f"{args.test_end}-08-01",
      "fit_start_time": f"{args.train_start}-01-01",
      "fit_end_time": f"{args.train_end}-12-31",
      "instruments": args.market,
      "infer_processors": infer_processors,
      "learn_processors": learn_processors,
      "label": [args.label_expr],
      "window" : args.win_size
    }
    alpha_config = {
      "class": "RawPriceChange",
      "module_path": "qlib.contrib.data.handler",
      "kwargs": data_handler_config,
    }
    input_size = 5 * args.win_size
  
  if args.loss_type == "br":
    output_dim = 2 # regress with bayesian
  elif args.loss_type == "cls":
    output_dim = 5 # 5 class for price change
  elif args.loss_type == "rgr":
    output_dim = 1 # regress the price directly

  task = {
    "model": {
      "class": "MLP",
      "module_path": "qlib.contrib.model.mlp",
      "kwargs": {
        "input_dim" : input_size,
        "output_dim" : output_dim,
        "hidden_dims" : hidden_sizes,
        "dropout" : 0.05
      },
    },
    "learner": {
      "class": "Learner",
      "module_path": "qlib.contrib.model.mlp",
      "kwargs": {
        "loss_type" : args.loss_type,
        "lr" : 0.002,
        "n_epoch" : args.n_epoch,
        "early_stop_patience" : args.n_epoch, # no early stopping
        "eval_n_epoch" : 20,
        "weight_decay" : 0.0002
      },
    },
    "dataset": {
      "class": "DatasetH",
      "module_path": "qlib.data.dataset",
      "kwargs": {
        "handler": alpha_config,
        "segments": {
          "train": (f"{args.train_start}-01-01", f"{args.train_end}-12-31"),
          "valid": (f"{args.valid_start}-01-01", f"{args.valid_end}-12-31"),
          "test": (f"{args.test_start}-01-01", f"{args.test_end}-12-31"),
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
        "topk": 50,
        "n_drop": 5,
        "signal": ModelSignal(model, dataset)
      },
    },
    "backtest": {
      "start_time": f"{args.test_start}-01-01",
      "end_time": f"{args.test_end}-12-31",
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


def plot_scatter_ci(ax, x, y, div_num=20):
  """Plot the conditional mean and confidence interval of scatter plot.
  The exact scatter plot is not shown.
  Args:
    ax: the figure ax.
    x: (N,) the x-axis of points.
    y: (N,) the y-axis of points.
    div_num: number of bins to compute mean
  """
  ind = x.argsort()
  sx, sy = x[ind], y[ind]
  xmin, xmax = sx[0], sx[-1]
  win_size = (xmax - xmin) / div_num
  step_size = win_size / 2

  cr_stds, cr_means, counts = [], [], []
  xs = []
  for i in range(2 * div_num - 1):
    mid = xmin + win_size / 2 + i * step_size
    st, ed = mid - win_size / 2, mid + win_size / 2
    if ed >= xmax: # out of border
      break
    ind_st = sx.searchsorted(st)
    ind_ed = sx.searchsorted(ed)
    xs.append(mid)
    counts.append(ind_ed - ind_st)
    cr_means.append(sy[ind_st:ind_ed].mean())
    cr_stds.append(sy[ind_st:ind_ed].std())
  xs, counts = np.array(xs), np.array(counts)
  cr_means, cr_stds = np.array(cr_means), np.array(cr_stds)
  sqrt_counts = np.sqrt(counts)
  valid = sqrt_counts > 1
  xs, cr_means, cr_stds = xs[valid], cr_means[valid], cr_stds[valid]
  counts, sqrt_counts = counts[valid], sqrt_counts[valid]
  CW = 1.96 * cr_stds / sqrt_counts
  ax.plot(xs, cr_means, color="blue", alpha=0.8)
  ax.fill_between(xs, (cr_means - CW), (cr_means + CW),
    color='blue', alpha=0.2)
  mean, sigma = cr_means.mean(), cr_means.std()
  cr_means_ = cr_means[np.abs(cr_means - mean) < 3 * sigma]
  mean, sigma = cr_means_.mean(), cr_means_.std()
  try:
    ax.set_ylim([mean - 3 * sigma, mean + 3 * sigma])
  except:
    pass


