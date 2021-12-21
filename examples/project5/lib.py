import numpy as np
import pandas as pd


from qlib.contrib.evaluate import risk_analysis
from qlib.backtest import backtest


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

  keys = ['mean', 'std', 'annualized_return', 'max_drawdown']
  items = ['excess_return_with_cost', 'excess_return_without_cost']

  return res, portfolio_metric_dict, indicator_dict


def assign_5label(df):
  ret = df.values.squeeze()
  label = np.zeros_like(ret).astype("int64")
  label[ret > -0.05] = 1
  label[ret > -0.01] = 2
  label[ret > 0.01] = 3
  label[ret > 0.05] = 4
  return pd.DataFrame({"class": label}, index=df.index)



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
  
  # offical configurations - if specified
  if "alpha" in args.name:
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
    max_steps = 8000
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
        "output_dim" : 2,
        "loss" : "br",
        "layers" : hidden_sizes,
        "lr" : 0.002,
        "max_steps" : max_steps,
        "batch_size" : 8192,
        "early_stop" : max_steps,
        "eval_steps" : 20,
        "lr_decay" : 0.96,
        "lr_decay_steps" : 100,
        "optimizer" : "adam",
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
        "model": model,
        "dataset": dataset,
        "topk": 50,
        "n_drop": 5,
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
  Args:
    ax: the figure ax.
    x: (N,) the x-axis of points.
    y: (N,) the y-axis of points.
    div_num: number of bins to compute mean
  """
  ind = x.argsort()
  xmean = x.mean()
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
  ax.plot(xs, [xmean] * len(xs), 'r--', linewidth=1)
  ax.fill_between(xs, (cr_means - CW), (cr_means + CW),
    color='blue', alpha=0.2)
  mean, sigma = cr_means.mean(), cr_means.std()
  cr_means_ = cr_means[np.abs(cr_means - mean) < 3 * sigma]
  mean, sigma = cr_means_.mean(), cr_means_.std()
  try:
    ax.set_ylim([mean - 3 * sigma, mean + 3 * sigma])
  except:
    pass


