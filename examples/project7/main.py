"""Test LSTM layers
"""
# pylint: disable=wrong-import-position,multiple-imports,import-error,invalid-name,line-too-long
import json, copy, argparse, os, torch, sys
import pytorch_lightning.loggers as pl_logger
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
sys.path.insert(0, "../..")
import qlib, lib
from lib import torch2numpy
from dataset import AlignedTSDataset, TSDataset
from qlib.utils import init_instance_by_config
from qlib.config import REG_CN
matplotlib.style.use('seaborn-poster')
matplotlib.style.use('ggplot')


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
        lib.plot_scatter_ci(ax, np.exp(pred[:, 1]/2), ys[i])
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
    """Convert ordinary DataFrame to compatible Time Series DataFrame.
    """
    df.columns = df.columns.droplevel()
    df = df.reorder_levels([1, 0]).sort_index()
    df = df.reset_index([0, 1])
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # training options
    parser.add_argument("--n-epoch", default=100, type=int,
                        help="The total training epochs.")
    parser.add_argument("--batch-size", default=1024, type=int,
                        help="Training batchsize.")
    parser.add_argument("--seq-len", default=64, type=int,
                        help="Training sequence length.")
    parser.add_argument("--gpu-id", default="0", type=str)
    parser.add_argument("--market", default="csi300", type=str,
                        choices=["csi300", "main"])
    parser.add_argument("--train-start", default=2011, type=int)
    parser.add_argument("--train-end", default=2012, type=int)
    parser.add_argument("--repeat-ind", default=0, type=int,
                        help="The index of repeats (to distinguish different runs).")
    # architecture options
    parser.add_argument("--hidden-size", default=256, type=int)
    parser.add_argument("--n-layer", default=1, type=int,
                        help="The number of hidden layers.")
    parser.add_argument("--loss-type", default="rgr", type=str,
                        help="The type of loss function and output format. rgr - regression; cls - classification; br - bayesian regression; mae - mean absolute error",
                        choices=["rgr", "mae", "cls", "br"])
    parser.add_argument("--data-type", default="raw", type=str,
                        help="The data type and preprocessing method.",
                        choices=["raw", "zscorenorm", "alpha158"])
    parser.add_argument("--label-type", default="pc-1",
                        help="The label for prediction",
                        choices=["pc-1"])
    # evaluation
    parser.add_argument("--top-k", default=50, type=int)
    parser.add_argument("--n-drop", default=5, type=int)
    parser.add_argument("--eval-only", default="0", type=str)
    parser.add_argument("--benchmark", default="SH000300", type=str)
    parser.add_argument("--test-start", default=2013, type=int)
    parser.add_argument("--test-end", default=2014, type=int)
    args = parser.parse_args()
    lib.set_cuda_devices(args.gpu_id)

    model_name = f"r{args.repeat_ind}_y{args.train_start}-y{args.test_end}"
    model_dir = f"expr/{args.market}_{args.data_type}_{args.label_type}/{args.loss_type}_l{args.n_layer}"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if os.path.exists(f"{model_dir}/{model_name}/result.json"):
        print(f"=> Skip {model_dir}/{model_name}/result.json.")
        exit(0)

    provider_uri = "../../data/china_stock_qlib_adj"
    qlib.init(provider_uri=provider_uri, region=REG_CN)
    torch.manual_seed(1997 + args.repeat_ind)
    torch.cuda.manual_seed(1997 + args.repeat_ind)

    args.label_expr = fetch_label(args.label_type)
    task = lib.get_train_config(args)
    dataset = init_instance_by_config(task["dataset"])
    model = init_instance_by_config(task["model"]).cuda()
    learner_config = copy.deepcopy(task["learner"])
    learner_config["kwargs"]["model"] = model
    learner = init_instance_by_config(learner_config).cuda()

    train_df = dataset.prepare(["train"],
        col_set=["feature", "label"], data_key="learn")[0]
    tvcv_names = list(train_df["feature"].columns)
    target_names = list(train_df["label"].columns)
    train_df = df_to_tsdf(train_df)
    train_ds = AlignedTSDataset(train_df,
        seq_len=args.seq_len, horizon=1,
        target_names=target_names,
        input_names=tvcv_names)
    train_dl = DataLoader(train_ds,
        batch_size=1, shuffle=True,
        pin_memory=True, num_workers=8)

    logger = pl_logger.TensorBoardLogger(f"{model_dir}/{model_name}")
    trainer = pl.Trainer(
        max_epochs=args.n_epoch,
        gpus=1,
        progress_bar_refresh_rate=1,
        log_every_n_steps=10,
        callbacks=[],
        logger=logger)
    trainer.fit(learner, train_dl)
    torch.save(model.state_dict(), f"{model_dir}/{model_name}/model.pth")

    test_df = dataset.prepare(["test"],
        col_set=["feature", "label"], data_key="infer")[0]
    test_df = df_to_tsdf(test_df)
    start_date = f"{args.test_start-1}-01-01"
    new_df = []
    for g_name, g in tqdm(train_df.groupby("instrument")):
        new_df.append(g[g.datetime >= start_date])
        new_df.append(test_df[test_df.instrument == g_name])
    test_df = pd.concat(new_df)
    test_df.reset_index(inplace=True)
    test_df.drop(columns="index", inplace=True)
    test_ds = TSDataset(test_df,
        seq_len=args.seq_len, horizon=1,
        target_names=target_names,
        input_names=tvcv_names)

    learner.cuda()
    test_scores, test_preds, test_insts, test_dates, test_indice = \
        learner.predict_dataset(test_ds)
    mask = pd.Series(test_dates) > f"{args.test_start}-01-01"
    test_scores, test_preds = test_scores[mask], test_preds[mask]
    test_signal = pd.Series(test_scores, [test_insts[mask], test_dates[mask]])
    test_signal.index.set_names(["instrument", "datetime"], inplace=True)
    test_gt = test_ds.df[test_ds.target_names].values[test_indice]
    test_gt = test_gt[mask].squeeze()
    report, final_res, _, _, month_res = lib.backtest_signal(test_signal, args)

    train_ds = TSDataset(train_df,
        seq_len=args.seq_len, horizon=1,
        target_names=target_names,
        input_names=tvcv_names)
    train_scores, train_preds, _, _, train_indice = \
        learner.predict_dataset(train_ds)
    train_gt = train_ds.df[train_ds.target_names].values
    train_gt = train_gt[train_indice].squeeze()
    if args.loss_type == "br":
        plot_br([train_scores, test_scores],
                [train_preds, test_preds],
                [train_gt, test_gt],
                model_dir, model_name)
    else:
        plot_normal([train_scores, test_scores],
                    [train_gt, test_gt],
                    model_dir, model_name)

    month_ret_key = 'return_total_return'
    month_er_key = 'excess_return_without_cost_total_return'
    month_bench_key = 'bench_return_total_return'
    eval_result = {"final": {
            "ER": float(final_res['ER'].risk['annualized_return']),
            "ERC": float(final_res['ERC'].risk['annualized_return'])
        }, "benchmark": {
            "R": float(final_res['benchmark'].risk['annualized_return']),
        }, "monthly_return": month_res[month_ret_key].to_dict()
         , "monthly_ER": month_res[month_er_key].to_dict()
         , "monthly_bench": month_res[month_bench_key].to_dict()
        }
    config = {
        "learner_config": task["learner"],
        "model_config": task["model"],
        "dataset_config": task["dataset"]}
    path = f"{model_dir}/{model_name}"
    with open(f"{path}/config.json", "w", encoding="ascii") as f:
        json.dump(config, f, indent=2)
    with open(f"{path}/result.json", "w", encoding="ascii") as f:
        json.dump(eval_result, f, indent=2)