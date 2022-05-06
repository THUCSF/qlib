"""Test LSTM layers
"""
# pylint: disable=wrong-import-position,multiple-imports,import-error,invalid-name,line-too-long
import json, copy, argparse, os, torch, sys, glob
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np

sys.path.insert(0, "../..")
import qlib, lib
from dataset import TSDataset
from qlib.utils import init_instance_by_config
from qlib.config import REG_CN

matplotlib.style.use("seaborn-poster")
matplotlib.style.use("ggplot")


def plot_br(scores, preds, ys, model_dir, model_name, subfix="final"):
    """Plot bayesian regression."""
    names = ["Train", "Test"]
    for i, pred in enumerate(preds):
        plt.figure(figsize=(30, 7))
        ax = plt.subplot(1, 3, 1)
        lib.plot_scatter_ci(ax, scores[i], ys[i])
        ax.set_title(f"{names[i]} Pred (score) v.s. Return")
        ax = plt.subplot(1, 3, 2)
        lib.plot_scatter_ci(ax, pred[:, 0], ys[i])
        ax.set_title(f"{names[i]} Pred (mean) v.s. Return")
        ax = plt.subplot(1, 3, 3)
        lib.plot_scatter_ci(ax, np.exp(pred[:, 1] / 2), ys[i])
        ax.set_title(f"{names[i]} Pred (std) v.s. Return")
        plt.tight_layout()
        plt.savefig(f"{model_dir}/{model_name}/pvr_{names[i]}_{subfix}.png")
        plt.close()


def plot_normal(xs, ys, model_dir, model_name, subfix="final"):
    """Plot normal models."""
    names = ["Train", "Test"]  # j
    plt.figure(figsize=(30, 7))
    for i, name in enumerate(names):
        ax = plt.subplot(1, len(names), i + 1)
        lib.plot_scatter_ci(ax, xs[i], ys[i])
        ax.set_title(f"{name} Pred v.s. Return")
    plt.tight_layout()
    plt.savefig(f"{model_dir}/{model_name}/pvr_{subfix}.png")
    plt.close()


def fetch_label(label_type):
    """Get label expression from type."""
    if label_type == "pc-1":
        return "100 * (Ref($close,-2)/Ref($close,-1) - 1)"


def df_to_tsdf(df):
    """Convert ordinary DataFrame to compatible Time Series DataFrame."""
    df.columns = df.columns.droplevel()
    df = df.reorder_levels([1, 0]).sort_index()
    df = df.reset_index([0, 1])
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # training options
    parser.add_argument(
        "--verbose", default=0, type=int, help="Whether to show verbose information."
    )
    parser.add_argument(
        "--strict-validation",
        default=1,
        type=int,
        help="Whether to ensure strict validation set.",
    )
    parser.add_argument(
        "--n1-epoch", default=50, type=int, help="The total initial training epochs."
    )
    parser.add_argument(
        "--n2-epoch", default=10, type=int, help="The rolling-based training epochs."
    )
    parser.add_argument(
        "--batch-size", default=1024, type=int, help="Training batchsize."
    )
    parser.add_argument(
        "--seq-len", default=64, type=int, help="Training sequence length."
    )
    parser.add_argument("--horizon", default=1, type=int, help="Predicted length.")
    parser.add_argument("--gpu-id", default="0", type=str)
    parser.add_argument(
        "--market", default="main", type=str, choices=["csi300", "main"]
    )
    parser.add_argument("--train-start", default=2011, type=int)
    parser.add_argument("--train-end", default=2013, type=int)
    parser.add_argument(
        "--repeat-ind",
        default=0,
        type=int,
        help="The index of repeats (to distinguish different runs).",
    )
    # architecture options
    parser.add_argument("--hidden-size", default=256, type=int)
    parser.add_argument(
        "--n-layer", default=2, type=int, help="The number of hidden layers."
    )
    parser.add_argument(
        "--loss-type",
        default="rgr-all",
        type=str,
        help="The type of loss function and output format. rgr - regression; cls - classification; br - bayesian regression; mae - mean absolute error",
    )
    parser.add_argument(
        "--data-type",
        default="raw",
        type=str,
        help="The data type and preprocessing method.",
        choices=["raw", "zscorenorm", "alpha158"],
    )
    parser.add_argument(
        "--label-type",
        default="pc-1",
        help="The label for prediction",
        choices=["pc-1"],
    )
    # evaluation
    parser.add_argument("--model-dir", default="expr/main_raw_pc-1", type=str)
    parser.add_argument("--top-k", default=50, type=int)
    parser.add_argument("--n-drop", default=10, type=int)
    parser.add_argument("--eval-only", default="0", type=str)
    parser.add_argument("--benchmark", default="SH000300", type=str)
    parser.add_argument("--test-start", default=2014, type=int)
    parser.add_argument("--test-end", default=2014, type=int)
    args = parser.parse_args()
    lib.set_cuda_devices(args.gpu_id)
    model_dir = glob.glob(f"{args.model_dir}/r{args.repeat_ind}_*-y{args.test_end}")[0]
    print(model_dir)
    model_name = model_dir[model_dir.rfind("/")+1:] 

    provider_uri = "../../data/china_stock_qlib_adj"
    qlib.init(provider_uri=provider_uri, region=REG_CN)
    torch.manual_seed(1997 + args.repeat_ind)
    torch.cuda.manual_seed(1997 + args.repeat_ind)

    train_end = args.train_end
    args.train_end = args.test_end  # to retrieve all data
    args.label_expr = fetch_label(args.label_type)
    task = lib.get_train_config(args)
    dataset = init_instance_by_config(task["dataset"])
    model = init_instance_by_config(task["model"]).cuda()
    learner_config = copy.deepcopy(task["learner"])
    learner_config["kwargs"]["model"] = model
    learner = init_instance_by_config(learner_config).cuda()
    args.train_end = train_end

    full_df = dataset.prepare(
        ["train"], col_set=["feature", "label"], data_key="learn"
    )[0]
    tvcv_names = list(full_df["feature"].columns)
    target_names = list(full_df["label"].columns)
    full_df = df_to_tsdf(full_df)

    learner.cuda()
    st = f"{args.test_start-1}-6-1"
    ed = f"{args.test_start}-12-31"
    full_df = [
        g[(g.datetime >= st) & (g.datetime <= ed)]
        for _, g in full_df.groupby("instrument")
    ]
    full_df = pd.concat(full_df)
    full_df.reset_index(inplace=True)
    full_df.drop(columns="index", inplace=True)
    test_ds = TSDataset(
        full_df,
        seq_len=args.seq_len,
        horizon=args.horizon,
        target_names=target_names,
        input_names=tvcv_names,
    )
    final_signals, best_signals, test_indice = [], [], []
    for i in range(1, 13):
        model_path = f"{model_dir}/model_y{args.test_start}_m{i:02d}"
        # final model
        model.load_state_dict(torch.load(f"{model_path}_final.pth"))
        test_scores, test_preds, test_insts, test_dates, idx = learner.predict_dataset(
            test_ds
        )
        test_dates = pd.Series(test_dates)
        st = f"{args.test_start}-{i}-1"
        ed = f"{args.test_start}-{i+1}-1" if i < 12 else f"{args.test_start+1}-1-1"
        mask = (st <= test_dates) & (test_dates < ed) if i > 1 else (test_dates < ed)
        test_indice.append(idx[mask])
        test_scores, test_preds = test_scores[mask], test_preds[mask]
        final_signals.append(
            pd.Series(test_scores, [test_insts[mask], test_dates[mask]])
        )

        # best model
        best_model_path = glob.glob(f"{model_path}_n=*.ckpt")[0]
        learner.load_state_dict(torch.load(best_model_path)["state_dict"])
        test_scores, test_preds, _, _, _ = learner.predict_dataset(test_ds)
        test_scores, test_preds = test_scores[mask], test_preds[mask]
        best_signals.append(
            pd.Series(test_scores, [test_insts[mask], test_dates[mask]])
        )

    final_signals = pd.concat(final_signals)
    best_signals = pd.concat(best_signals)
    final_signals.index.set_names(["instrument", "datetime"], inplace=True)
    best_signals.index.set_names(["instrument", "datetime"], inplace=True)
    final_signals.to_pickle(f"{model_dir}/final_test_signal.pkl")
    best_signals.to_pickle(f"{model_dir}/best_test_signal.pkl")
    final_report, final_res, _, _, final_month_res = lib.backtest_signal(
        final_signals, args
    )
    best_report, best_res, _, _, best_month_res = lib.backtest_signal(
        best_signals, args
    )

    lib.save_results(
        f"{model_dir}/{model_name}",
        task,
        final_res,
        final_month_res,
        final_report,
        best_res,
        best_month_res,
        best_report,
    )
