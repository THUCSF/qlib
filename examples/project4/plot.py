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
from mlp import simple_backtest


def get_data(df, N=5):
  index_df = df.index.to_frame()
  all_codes = index_df.instrument.unique()
  all_dates = index_df.datetime.unique()
  rng = np.random.RandomState(1)
  selected_codes = rng.choise(all_codes, (N,))
  start_date = rng.randint(0, all_dates.shape[0] - 300)
  selected_dates = all_dates[start_date:]
  return selected_codes, selected_dates


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


def assign_5label(df):
  ret = df.values.squeeze()
  label = np.zeros_like(ret).astype("int64")
  label[ret > -0.05] = 1
  label[ret > -0.01] = 2
  label[ret > 0.01] = 3
  label[ret > 0.05] = 4
  return pd.DataFrame({"class": label}, index=df.index)


def experiment(expr_dir, dataset_config, models, model_cfgs, res_dic, args):
  # model initiaiton
  dataset = init_instance_by_config(dataset_config)
  train_df, val_df, test_df = dataset.prepare(
    ["train", "valid", "test"],
    col_set=["feature", "label"],
    data_key="infer")
  x_train, y_train = train_df["feature"], train_df["label"]
  x_valid, y_valid = val_df["feature"], val_df["label"]
  x_test, y_test = test_df["feature"], test_df["label"]

  for i, model in enumerate(models):
    res, _, _ = simple_backtest(model, dataset, args)
    set_dic(res_dic, model_cfgs[i][0], model_cfgs[i][1],
      res['excess_return_without_cost'].risk['annualized_return'])

    fig = plt.figure(figsize=(30, 7))
    ax = plt.subplot(1, 3, 1)
    y = y_train.values.squeeze()
    pred = model.predict_dataframe(x_train)
    mask = (y >= -0.1) & (y <= 0.1)
    plot_scatter_ci(ax, pred[mask], y[mask])
    ax.set_title(f"Train Pred v.s. Return")

    y = y_valid.values.squeeze()
    pred = model.predict_dataframe(x_valid)
    mask = (y >= -0.1) & (y <= 0.1)
    ax = plt.subplot(1, 3, 2)
    plot_scatter_ci(ax, pred[mask], y[mask])
    ax.set_title(f"Valid Pred v.s. Return")

    y = y_test.values.squeeze()
    pred = model.predict_dataframe(x_test)
    mask = (y >= -0.1) & (y <= 0.1)
    ax = plt.subplot(1, 3, 3)
    plot_scatter_ci(ax, pred[mask], y[mask])
    ax.set_title(f"Test Pred v.s. Return")
    plt.tight_layout()
    plt.savefig(f"{expr_dir}/r{i}_scatter_{args.plot_ds}.png")
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


def dic_mean_std(dic, sort_int=True):
  new_dic = {}

  if sort_int:
    row_keys = [int(k) for k in dic.keys()]
    row_keys.sort()
  else:
    row_keys = dic.keys()

  for row_key in row_keys:
    row_key = str(row_key)
    new_dic[row_key] = {}

    if sort_int:
      col_keys = [int(k) for k in dic[row_key].keys()]
      col_keys.sort()
    else:
      col_keys = dic[row_key].keys()

    for col_key in col_keys:
      col_key = str(col_key)
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
  parser.add_argument("--train-ds", default="china_stock_qlib_adj",
    help="qlib_cn_stock | china_stock_qlib_adj")
  parser.add_argument("--plot-ds", default="china_stock_qlib_adj",
    help="qlib_cn_stock | china_stock_qlib_adj")
  parser.add_argument("--benchmark", default="SH000300", type=str)
  args = parser.parse_args()

  provider_uri = f"../../data/{args.plot_ds}"
  qlib.init(provider_uri=provider_uri, region=REG_CN)

  expr_dir = f"expr/{args.name}"

  model_dirs = glob.glob(f"{expr_dir}/{args.train_ds}/*")
  model_dirs.sort()
  res_dic = {}
  for model_dir in model_dirs:
    expr_names = glob.glob(f"{model_dir}/*.pth")
    expr_names.sort()
    if len(expr_names) == 0:
      continue
    models, model_cfgs = [], []
    for expr_name in expr_names:
      print(expr_name)
      expr_res = torch.load(expr_name, map_location="cpu")
      expr_res["model_config"]["kwargs"]["GPU"] = args.gpu_id
      # load model
      model = init_instance_by_config(expr_res["model_config"])
      model.model.load_state_dict(expr_res["model_param"])
      model.fitted = True
      models.append(model)

      cfgs = expr_name.split("/")[-2]
      cfgs = cfgs[cfgs.rfind("/")+1:].strip().split("_")
      cfgs = [c[1:] for c in cfgs]
      model_cfgs.append(cfgs)
    dataset_config = expr_res["dataset_config"]
    print(f"=> Plot {model_dir}")
    experiment(model_dir, dataset_config, models, model_cfgs, res_dic, args)

  std_dic = dic_mean_std(res_dic)
  with open(f"{expr_dir}/{args.train_ds}/res_{args.plot_ds}.tex", "w") as f:
    f.write(str_latex_table(str_table_single_std(std_dic)))
