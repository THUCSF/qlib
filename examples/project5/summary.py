"""Summarize the results of MLPs into latex tables.
"""
import sys
sys.path.insert(0, "../..")

import pprint
import torch, argparse, glob, json
import numpy as np
import pandas as pd


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
  parser.add_argument("--benchmark", default="SH000300", type=str)
  args = parser.parse_args()

  label_type = "pc-1"
  for market in ["csi300", "main"]:
    for data_type in ["raw"]:
      expr_dir = f"expr/{market}_{data_type}_{label_type}"
      model_dirs = glob.glob(f"{expr_dir}/*")
      model_dirs.sort()
      model_dirs = [m for m in model_dirs if "." not in m]
      res_dic = {}
      for model_dir in model_dirs:
        model_name = model_dir[model_dir.rfind("/")+1:]
        loss_type, layer, window = model_name.split("_") # e.g. br_l1_w32
        if loss_type not in res_dic:
          res_dic[loss_type] = {}
        layer = layer[1:]
        window = window[1:]
        model_repeat_dirs = glob.glob(f"{model_dir}/*")
        model_repeat_dirs.sort()
        for mrd in model_repeat_dirs:
          mr_name = mrd[mrd.rfind("/")+1:]
          repeat_ind, data_range = mr_name.split("_")
          repeat_ind = repeat_ind[1:]
          data_range = data_range[1:]
          if data_range not in res_dic[loss_type]:
            res_dic[loss_type][data_range] = {}
          try:
            with open(f"{mrd}/result.json", "r") as f:
              res = json.load(f)
          except:
            pass
          set_dic(res_dic[loss_type][data_range], layer, window, res["ER"])
      
      for loss_type in res_dic:
        for data_range in res_dic[loss_type]:
          std_dic = dic_mean_std(res_dic[loss_type][data_range])
          print(market, data_type, loss_type, data_range)
          pprint.pprint(std_dic)
          with open(f"{expr_dir}/{loss_type}_{data_range}.tex", "w") as f:
            f.write(str_latex_table(str_table_single_std(std_dic)))
