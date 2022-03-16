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
from torch.utils.data import DataLoader
from tqdm import tqdm
from pytorch_lightning.callbacks import EarlyStopping
sys.path.insert(0, "../..")
import qlib, lib
from lib import torch2numpy
from dataset import TSDataset
from qlib.utils import init_instance_by_config
from qlib.config import REG_CN
matplotlib.style.use('seaborn-poster')
matplotlib.style.use('ggplot')


def plot_br(x_dls, ys, trainer, learner, model_dir, model_name, subfix="final"):
    """Plot bayesian regression."""
    points = []
    xnames = ["mu", "sigma", "score"]  # i
    names = ["Train", "Test"]  # j

    for dl in x_dls:
        out = torch.cat(trainer.predict(learner, dl))
        mu = torch2numpy(out[:, 0])
        sigma = torch2numpy(torch.sigmoid(out[:, 1]))
        score = torch2numpy(learner.pred2score(out))
        points.append((mu, sigma, score))

    for i in range(3):
        plt.figure(figsize=(30, 7))
        for j, name in enumerate(names):
            ax = plt.subplot(1, len(names), j + 1)
            lib.plot_scatter_ci(ax, points[j][i], ys[j].squeeze())
            ax.set_title(f"{name} Pred ({xnames[i]}) v.s. Return")
        plt.tight_layout()
        plt.savefig(f"{model_dir}/{model_name}/pvr_{xnames[i]}_{subfix}.png")
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


def locate_raw_data(batch, ds, df, x_names, y_names, pos=0):
    """Locate the transformed input to the raw dataframe.
    """
    input_data = batch[0]["encoder_cont"][0]
    input_numpy = input_data.detach().cpu().numpy()
    input_df = pd.DataFrame(input_numpy, columns=x_names)
    res_df = ds.x_to_index(batch[0])
    instrument = res_df.instrument[pos]
    time_index = res_df.time_index[pos]
    mask = (df.instrument == instrument) & (df.time_index <= time_index) & (
        df.time_index >= time_index - input_data.shape[0])
    orig_df = df[mask][x_names + y_names]
    return input_df, batch[1][0][pos], orig_df


def predict_dataset(net, ds):
    """Predict scores from a dataset."""
    scores = []
    BS = 1024
    # pinned inputs
    x = torch.Tensor(BS, ds.seq_len, len(ds.input_names)).cuda()
    values = torch.from_numpy(ds.df[ds.input_names].values)
    with torch.no_grad():
        for idx in tqdm(range(len(ds))):
            st, ed = ds.sample_indice[idx]
            x[idx % BS].copy_(values[st:ed], True)
            if (idx + 1) % BS == 0:
                pred = net(x)
                scores.append(torch2numpy(pred.squeeze()))
        if len(ds) % BS != 0:
            pred = net(x[:len(ds) % BS])
            scores.append(torch2numpy(pred.squeeze()))
    target_indice = ds.sample_indice[:, 1] + ds.horizon - 1 # (N,)
    insts = ds.df[ds.inst_index].values[target_indice]
    dates = ds.df[ds.time_index].values[target_indice]
    return np.concatenate(scores), insts, dates, target_indice


def predict_dataloader(net, dl):
    """Predict the scores from a dataloader."""
    scores = []
    with torch.no_grad():
        for batch in tqdm(dl):
            pred = net(batch["input"].cuda())
            scores.append(pred.squeeze().detach().cpu().numpy())
    return np.concatenate(scores)


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
    parser.add_argument("--n-layer", default=2, type=int,
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
    parser.add_argument("--eval-only", default="0", type=str)
    parser.add_argument("--benchmark", default="SH000300", type=str)
    parser.add_argument("--test-start", default=2013, type=int)
    parser.add_argument("--test-end", default=2014, type=int)
    args = parser.parse_args()
    lib.set_cuda_devices(args.gpu_id)

    provider_uri = "../../data/china_stock_qlib_adj"
    qlib.init(provider_uri=provider_uri, region=REG_CN)

    model_name = f"r{args.repeat_ind}_y{args.train_start}-y{args.test_end}"
    model_dir = f"expr/{args.market}_{args.data_type}_{args.label_type}/{args.loss_type}_l{args.n_layer}"
    if not os.path.exists(f"{model_dir}/{model_name}/model.pth"):
        print(f"!> {model_dir}/{model_name}/model.pth not found.")
        exit(0)

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
    train_ds = TSDataset(train_df,
        seq_len=args.seq_len, horizon=1,
        target_names=target_names,
        input_names=tvcv_names)
    train_dl = DataLoader(train_ds,
        batch_size=args.batch_size, shuffle=True,
        pin_memory=True, num_workers=8)

    state_dict = torch.load(f"{model_dir}/{model_name}/model.pth")
    learner.model.load_state_dict(state_dict)
    
    test_df = dataset.prepare(["test"],
        col_set=["feature", "label"], data_key="infer")[0]
    test_df = df_to_tsdf(test_df)
    start_date = f"{args.test_start-1}-01-01"
    new_df = []
    for g_name, g in train_df.groupby("instrument"):
        new_df.append(g[g.datetime > start_date])
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