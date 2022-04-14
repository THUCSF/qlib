"""Summarize the results of MLPs into latex tables.
"""
import json
import glob
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.style.use("seaborn-poster")
matplotlib.style.use("ggplot")


def str_table_single_std(dic, table_header="", output_std=True):
    row_names = list(dic.keys())
    col_names = list(dic[row_names[0]].keys())
    strs = [[table_header] + [str(c) for c in col_names]]
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
        row_keys = [k for k in dic.keys()]
        row_keys.sort()
    else:
        row_keys = dic.keys()

    for row_key in row_keys:
        row_key = str(row_key)
        new_dic[row_key] = {}

        if sort_int:
            col_keys = [k for k in dic[row_key].keys()]
            col_keys.sort()
        else:
            col_keys = dic[row_key].keys()

        for col_key in col_keys:
            col_key = str(col_key)
            obs = np.array(dic[row_key][col_key])
            mean, std = obs.mean(), obs.std(ddof=1)
            new_dic[row_key][col_key] = {"mean": mean, "std": std}
    return new_dic


def set_dic(dic, key1, key2, key3, val):
    if key1 and key1 not in dic:
        dic[key1] = {}
    if key2 and key2 not in dic[key1]:
        dic[key1][key2] = {}
    if key3:
        dic[key1][key2][key3] = val
    else:
        dic[key1][key2] = val


def plot_monthly(model_repeats, prefix):
    xs = list(model_repeats[0]["monthly_ER"].keys())
    ys = np.array([list(dic["monthly_ER"].values()) for dic in model_repeats])
    means = ys.mean(0)
    mins, maxs = ys.min(0), ys.max(0)
    plt.figure(figsize=(30, 10))
    plt.plot(xs, means)
    plt.fill_between(xs, mins, maxs, alpha=0.2)
    plt.plot(xs, np.zeros(len(xs)), "--")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(f"{prefix}_monthly.png")
    plt.close()


def lastday_month(xs):
    res = []
    for prev, cur in zip(xs[:-1], xs[1:]):
        if prev.split("-")[1] != cur.split("-")[1]:
            res.append(prev)
    return res


def plot_daily(model_repeats, prefix):
    xs = list(model_repeats[0]["daily_return"].keys())
    ys = [np.array(list(dic["daily_return"].values())) for dic in model_repeats]
    ys = np.stack([np.cumprod(1 + y) for y in ys])
    means = ys.mean(0)
    mins, maxs = ys.min(0), ys.max(0)
    plt.figure(figsize=(30, 10))
    plt.plot(xs, means)
    plt.fill_between(xs, mins, maxs, alpha=0.2)
    plt.plot(xs, np.ones(len(xs)), "--")
    plt.xticks(lastday_month(xs), rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(f"{prefix}_daily.png")
    plt.close()


def main():
    model_dirs = glob.glob(f"{args.expr}/*")
    model_dirs.sort()
    model_dirs = [m for m in model_dirs if "." not in m]
    res_dic = {}
    for model_dir in model_dirs:
        model_name = model_dir[model_dir.rfind("/") + 1 :]
        loss_type, layer = model_name.split("_")  # e.g. br_l1_w32
        if loss_type not in res_dic:
            res_dic[loss_type] = {}
        layer = layer[1:]
        model_repeat_dirs = glob.glob(f"{model_dir}/*")
        model_repeat_dirs.sort()
        for mrd in model_repeat_dirs:
            mr_name = mrd[mrd.rfind("/") + 1 :]
            repeat_ind, data_range = mr_name.split("_")
            repeat_ind = repeat_ind[1:]
            data_range = data_range[1:]
            if data_range not in res_dic[loss_type]:
                res_dic[loss_type][data_range] = []
            try:
                with open(f"{mrd}/result.json", "r") as f:
                    res = json.load(f)
            except:
                print(f"!> Not found: {mrd}/result.json")
                continue
            res_dic[loss_type][data_range].append(res)

    num_dic = {"final": {}, "best": {}}
    for loss_type, loss_dic in res_dic.items():
        for data_range, data_dic in loss_dic.items():
            for stop_strategy in ["final", "best"]:
                model_repeats = [m[stop_strategy] for m in data_dic]
                if len(model_repeats) == 0:
                    print(f"!> {loss_type} {data_range} is empty!")
                    continue
                save_prefix = f"{args.expr}/{loss_type}_{data_range}_{stop_strategy}"
                plot_daily(model_repeats, save_prefix)
                set_dic(
                    num_dic,
                    stop_strategy,
                    loss_type,
                    data_range,
                    [m["ERC"] for m in model_repeats],
                )

    for stop_strategy in ["final", "best"]:
        strs = []
        if len(num_dic[stop_strategy]) == 0:
            continue
        std_dic = dic_mean_std(num_dic[stop_strategy])
        table_strs = str_table_single_std(std_dic, table_header="layer")
        s = f"{loss_type} during {data_range}"
        s = "\\multicolumn{" + str(len(table_strs[0]) - 1) + "}" + "{c}" + "{" + s + "}"
        if len(strs) != 0:
            table_strs[0] = ["", s]
        else:
            table_strs = table_strs[:1] + [["", s]] + table_strs[1:]
        strs.extend(table_strs)

        if len(strs) > 0:
            with open(f"{args.expr}/{loss_type}_{stop_strategy}.tex", "w") as f:
                f.write(str_latex_table(strs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="SH000300", type=str)
    parser.add_argument("--expr", default="expr/main_raw_pc-1/rgr-all_l2", type=str)
    args = parser.parse_args()
    main()
