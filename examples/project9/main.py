"""Test LSTM layers
"""
# pylint: disable=wrong-import-position,multiple-imports,import-error,invalid-name,line-too-long
import json, copy, argparse, os, torch, sys, glob
import pytorch_lightning.loggers as pl_logger
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
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
    parser.add_argument("--verbose", default=0, type=int,
                        help="Whether to show verbose information.")
    parser.add_argument("--strict-validation", default=0, type=int,
                        help="Whether to ensure strict validation set.")
    parser.add_argument("--n1-epoch", default=50, type=int,
                        help="The total initial training epochs.")
    parser.add_argument("--n2-epoch", default=10, type=int,
                        help="The rolling-based training epochs.")
    parser.add_argument("--batch-size", default=1024, type=int,
                        help="Training batchsize.")
    parser.add_argument("--seq-len", default=64, type=int,
                        help="Training sequence length.")
    parser.add_argument("--horizon", default=1, type=int,
                        help="Predicted length.")
    parser.add_argument("--gpu-id", default="0", type=str)
    parser.add_argument("--market", default="csi300", type=str,
                        choices=["csi300", "main"])
    parser.add_argument("--train-start", default=2011, type=int)
    parser.add_argument("--train-end", default=2013, type=int)
    parser.add_argument("--repeat-ind", default=0, type=int,
                        help="The index of repeats (to distinguish different runs).")
    # architecture options
    parser.add_argument("--hidden-size", default=256, type=int)
    parser.add_argument("--n-layer", default=2, type=int,
                        help="The number of hidden layers.")
    parser.add_argument("--loss-type", default="rgr-all", type=str,
                        help="The type of loss function and output format. rgr - regression; cls - classification; br - bayesian regression; mae - mean absolute error")
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
    parser.add_argument("--test-start", default=2014, type=int)
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

    train_end = args.train_end
    args.train_end = args.test_end # to retrieve all data
    args.label_expr = fetch_label(args.label_type)
    task = lib.get_train_config(args)
    dataset = init_instance_by_config(task["dataset"])
    model = init_instance_by_config(task["model"]).cuda()
    learner_config = copy.deepcopy(task["learner"])
    learner_config["kwargs"]["model"] = model
    learner = init_instance_by_config(learner_config).cuda()
    args.train_end = train_end

    full_df = dataset.prepare(["train"],
        col_set=["feature", "label"], data_key="learn")[0]
    tvcv_names = list(full_df["feature"].columns)
    target_names = list(full_df["label"].columns)
    full_df = df_to_tsdf(full_df)
    valid_mask = full_df[target_names[0]].abs() < 11
    full_df = full_df[valid_mask]
    full_df.reset_index(inplace=True)
    full_df.drop(columns="index", inplace=True)
    aligned_full_ds = AlignedTSDataset(full_df, # AlignedTSDataset
        seq_len=args.seq_len, horizon=args.horizon,
        target_names=target_names,
        input_names=tvcv_names)

    val_end_date = f"{args.train_end+1}-1-1"
    train_end_date = f"{args.train_end}-12-1" \
                        if args.strict_validation else val_end_date
    train_ds = aligned_full_ds.get_split(
        f"{args.train_start}-1-1", train_end_date)
    val_ds = aligned_full_ds.get_split(
        f"{args.train_end}-12-1", val_end_date)
    print("=> Training dataset:")
    train_ds.describe()
    print("=> Validation dataset:")
    val_ds.describe()
    train_dl = DataLoader(train_ds, batch_size=1, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False)

    print(f"=> Initial training on {args.train_start}-{args.train_end}")
    logger = pl_logger.TensorBoardLogger(f"{model_dir}/{model_name}")
    model_prefix = f"model_y{args.test_start}_m01"
    mc = ModelCheckpoint(save_weights_only=True,
      dirpath=f"{model_dir}/{model_name}",
      filename=model_prefix + "_n={epoch}_f={val_metric:.2f}",
      monitor="val_metric", mode="min")
    trainer = pl.Trainer(
        max_epochs=args.n1_epoch,
        gpus=1,
        log_every_n_steps=1,
        callbacks=[mc],
        accumulate_grad_batches=10,
        track_grad_norm=1,
        logger=logger)
    trainer.fit(learner, train_dl, val_dl)
    torch.save(model.state_dict(),
        f"{model_dir}/{model_name}/{model_prefix}_final.pth")

    learner.cuda().eval()
    full_ds = TSDataset(full_df,
                seq_len=args.seq_len, horizon=args.horizon,
                target_names=target_names, input_names=tvcv_names)
    model_path = f"{model_dir}/{model_name}/model_y{args.test_start}_m01"
    # final model
    model.load_state_dict(torch.load(f"{model_path}_final.pth"))
    final_scores, final_preds, insts, dates, indice = \
        learner.predict_dataset(full_ds)
    dates = pd.Series(dates)
    final_signals = pd.Series(final_scores, [insts, dates])
    final_signals.index.set_names(["instrument", "datetime"], inplace=True)
    final_report, final_res, _, _, final_month_res = \
        lib.backtest_signal(final_signals, args)

    # best model
    best_model_path = glob.glob(f"{model_path}_n=*.ckpt")[0]
    learner.load_state_dict(torch.load(best_model_path)["state_dict"])
    best_scores, best_preds, insts, dates, indice = \
        learner.predict_dataset(full_ds)
    dates = pd.Series(dates)
    train_mask = dates < f"{args.test_start}-1-1"
    test_mask = ~train_mask
    best_signals = pd.Series(best_scores, [insts, dates])
    best_signals.index.set_names(["instrument", "datetime"], inplace=True)
    best_report, best_res, _, _, best_month_res = \
        lib.backtest_signal(best_signals, args)

    # split signal between train and test
    label_gt = full_df[target_names].values[indice].squeeze()
    train_gt, test_gt = label_gt[train_mask], label_gt[test_mask]
    train_scores = best_scores[train_mask]
    test_scores = best_scores[test_mask]

    # plot
    if "br" in args.loss_type:
        train_preds = best_preds[train_mask]
        test_preds = best_preds[test_mask]
        plot_br([train_scores, test_scores],
                [train_preds, test_preds],
                [train_gt, test_gt],
                model_dir, model_name)
    else:
        plot_normal([train_scores, test_scores],
                    [train_gt, test_gt],
                    model_dir, model_name)

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
