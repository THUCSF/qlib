"""Test MLP layers and window size
"""
import sys
sys.path.insert(0, "../..")

import torch, argparse, glob
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('seaborn-poster')
matplotlib.style.use('ggplot')


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
  parser.add_argument("--ds", default="china_stock_qlib_adj",
    help="qlib_cn_stock | china_stock_qlib_adj")
  parser.add_argument("--benchmark", default="SH000300", type=str)
  parser.add_argument("--market", default="csi300", type=str)
  args = parser.parse_args()

  expr_dir = f"expr/{args.name}/{args.ds}/{args.market}"

  model_dirs = glob.glob(f"{expr_dir}/*")
  model_dirs.sort()
  res_dic = {}
  for model_dir in model_dirs:
    expr_names = glob.glob(f"{model_dir}/*.pth")
    expr_names.sort()
    if len(expr_names) == 0:
      continue
    models, model_cfgs = [], []
    for expr_name in expr_names:
      expr_res = torch.load(expr_name, map_location="cpu")
      expr_res["model_config"]["kwargs"]["GPU"] = args.gpu_id
      cfgs = expr_name.split("/")[-2]
      cfgs = cfgs[cfgs.rfind("/")+1:].strip().split("_")
      cfgs = [c[1:] for c in cfgs]
      set_dic(res_dic, cfgs[0], cfgs[1],
        expr_res["eval_result"]["return"])
  print(res_dic)
  std_dic = dic_mean_std(res_dic)
  with open(f"{expr_dir}/res.tex", "w") as f:
    f.write(str_latex_table(str_table_single_std(std_dic)))
