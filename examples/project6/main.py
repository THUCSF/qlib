"""Test LSTM layers
"""
import sys
sys.path.insert(0, "../..")
import qlib, torch, os, argparse, copy, json, glob
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('seaborn-poster')
matplotlib.style.use('ggplot')

from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_logger
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_forecasting import TimeSeriesDataSet
from qlib.config import REG_CN
from qlib.utils import init_instance_by_config
from lib import *



def plot_br(x_dls, ys, trainer, learner, model_dir, model_name, subfix=""):
  points = []
  xnames = ["mu", "sigma", "score"] # i
  names = ["Train", "Test"] # j

  for i in range(len(x_dls)):
    out = torch.cat(trainer.predict(learner, x_dls[i]))
    mu = torch2numpy(out[:, 0])
    sigma = torch2numpy(torch.sigmoid(out[:, 1]))
    score = torch2numpy(learner.pred2score(out))
    points.append((mu, sigma, score))

  for i in range(3):
    plt.figure(figsize=(30, 7))
    for j in range(len(names)):
      ax = plt.subplot(1, len(names), j + 1)
      plot_scatter_ci(ax, points[j][i], ys[j].squeeze())
      ax.set_title(f"{names[j]} Pred ({xnames[i]}) v.s. Return")
    plt.tight_layout()
    plt.savefig(f"{model_dir}/{model_name}/pvr_{xnames[i]}_{subfix}.png")
    plt.close()


def plot_normal(xs, ys, model_dir, model_name, subfix=""):
  names = ["Train", "Test"] # j
  plt.figure(figsize=(30, 7))
  for i in range(len(names)):
    ax = plt.subplot(1, len(names), i + 1)
    plot_scatter_ci(ax, xs[i], ys[i].squeeze())
    ax.set_title(f"{names[i]} Pred v.s. Return")
  plt.tight_layout()
  plt.savefig(f"{model_dir}/{model_name}/pvr_{subfix}.png")
  plt.close()


def main(args):
  pass


def fetch_label(label_type):
  if label_type == "pc-1":
    return "Ref($close,-2)/Ref($close,-1)-1"


def df_to_tsdf(df):
  """Convert ordinary DataFrame to compatible Time Series DataFrame.
  """
  tvcv_names = list(df["feature"].columns)
  target_names = list(df["label"].columns)
  df.columns = df.columns.droplevel()
  df["time_index"] = 0
  def s_(x):
    x["time_index"] = np.arange(x.shape[0])
    return x
  df = df.groupby("instrument").apply(s_)
  df = df.reset_index([0, 1])
  return df, tvcv_names, target_names


def locate_raw_data(batch, ds, df, x_names, y_names, pos=0):
  """Locate the transformed input to the raw dataframe.
  """
  input_data = batch[0]["encoder_cont"][0]
  input_numpy = input_data.detach().cpu().numpy()
  input_df = pd.DataFrame(input_numpy, columns=x_names)
  res_df = ds.x_to_index(batch[0])
  instrument = res_df.instrument[pos]
  time_index = res_df.time_index[pos]
  mask = (df.instrument == instrument) & (df.time_index <= time_index) & (df.time_index >= time_index - input_data.shape[0])
  orig_df = df[mask][x_names + y_names]
  return input_df, batch[1][0][pos], orig_df


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
  parser.add_argument("--market", default="main", type=str,
    choices=["csi300", "main"])
  parser.add_argument("--train-start", default=2011, type=int)
  parser.add_argument("--train-end", default=2012, type=int)
  parser.add_argument("--repeat-ind", default=0, type=int,
    help="The index of repeats (to distinguish different runs).")
  # architecture options
  parser.add_argument("--hidden-size", default=256, type=int)
  parser.add_argument("--n-layer", default=1, type=int,
    help="The number of MLP hidden layers.")
  parser.add_argument("--loss-type", default="rgr", type=str,
    help="The type of loss function and output format. rgr - regression; cls - classification; br - bayesian regression",
    choices=["rgr", "cls", "br"])
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
  parser.add_argument("--test-end", default=2013, type=int)
  args = parser.parse_args()
  set_cuda_devices(args.gpu_id)

  provider_uri = f"../../data/china_stock_qlib_adj"
  qlib.init(provider_uri=provider_uri, region=REG_CN)

  model_name = f"r{args.repeat_ind}_y{args.train_start}-y{args.test_end}"
  model_dir = f"expr/{args.market}_{args.data_type}_{args.label_type}/{args.loss_type}_l{args.n_layer}"
  if not os.path.exists(model_dir):
    os.makedirs(model_dir)
  
  args.label_expr = fetch_label(args.label_type)
  task = get_train_config(args)
  dataset = init_instance_by_config(task["dataset"])
  model = init_instance_by_config(task["model"]).cuda()
  learner_config = copy.deepcopy(task["learner"])
  learner_config["kwargs"]["model"] = model
  learner = init_instance_by_config(learner_config).cuda()
  train_df = dataset.prepare(["train"],
    col_set=["feature", "label"], data_key="learn")[0]
  train_df, tvcv_names, target_names = df_to_tsdf(train_df)
  train_ds = TimeSeriesDataSet(train_df,
    max_encoder_length=args.seq_len,
    max_prediction_length=1,
    time_idx="time_index",
    target=target_names[0],
    group_ids=["instrument"],
    time_varying_unknown_reals=tvcv_names)

  train_dl = train_ds.to_dataloader(
    train=True, batch_size=args.batch_size, num_workers=2)

  logger = pl_logger.TensorBoardLogger(f"{model_dir}/{model_name}")
  es_cb = EarlyStopping("lr", mode="min", patience=1e3, stopping_threshold=5e-6)
  trainer = pl.Trainer(
    max_epochs=1,
    gpus=1,
    gradient_clip_val=1,
    callbacks=[es_cb],
    logger=logger)
  trainer.fit(learner, train_dl)
  torch.save(model.state_dict(), f"{model_dir}/{model_name}/model.pth")
  learner.cuda()
  test_df = dataset.prepare(["test"],
    col_set=["feature", "label"], data_key="infer")[0]
  test_df, _, _ = df_to_tsdf(test_df)
  start_date = f"{args.test_start - 1}-01-01"
  test_df = train_df[train_df.datetime > start_date].append(test_df)
  test_scores = learner.predict_dataframe(test_df, tvcv_names)
  mask = test_df.datetime > f"{args.test_start}-01-01"
  test_scores = test_scores[mask]
  test_df = test_df[mask]
  test_scores.index = test_df.set_index(["datetime", "instrument"]).index

  train_scores = learner.predict_dataframe(train_df, tvcv_names)
  train_scores.index = train_df.set_index(["datetime", "instrument"]).index

  final_res, _, _ = backtest_signal(test_scores, args)

  if "br" in args.loss_type:
    pass
  else:
    plot_normal(
      [train_scores, test_scores],
      [train_df[args.label_expr], test_df[args.label_expr]],
      model_dir, model_name,
      subfix="final")

  eval_result = {
    "final" : {
      "ER" : float(final_res['ER'].risk['annualized_return']),
      "ERC" : float(final_res['ERC'].risk['annualized_return'])
    }, "benchmark" : {
      "R" : float(final_res['benchmark'].risk['annualized_return']),
    }}
  config = {
    "learner_config" : task["learner"],
    "model_config" : task["model"],
    "dataset_config" : task["dataset"]}
  with open(f"{model_dir}/{model_name}/config.json", "w") as f:
    json.dump(config, f, indent=2)
  with open(f"{model_dir}/{model_name}/result.json", "w") as f:
    json.dump(eval_result, f, indent=2)

