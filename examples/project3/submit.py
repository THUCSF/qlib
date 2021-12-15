"""Easy execution script.
Running scripts at different gpu slots:
python submit.py --gpu 0/1/2/3/...

Running scripts at different multi-gpu slots:
python submit.py --gpu 0,1/2,3/4,5/...

Running scripts with default gpu:
python submit.py --gpu -1
"""
import os, argparse

parser = argparse.ArgumentParser()
parser.add_argument('--func', default="test", type=str)
parser.add_argument('--gpu', default='-1')
args = parser.parse_args()


def mlp_official():
  cmds = []
  cmd = "python mlp.py --name official --repeat-ind {repeat_ind} --n-layer {n_layer} --dataset {ds}"
  for n_layer in [1, 2, 4, 8]:
    for repeat_ind in range(5):
      for ds in ["china_stock_qlib_adj", "qlib_cn_stock"]:
        cmds.append(cmd.format(repeat_ind=repeat_ind,
          n_layer=n_layer, ds=ds))
  return cmds


def mlp_zscorenorm():
  cmds = []
  cmd = "python mlp.py --name zscorenorm --repeat-ind {repeat_ind} --n-layer {n_layer} --win-size {win_size} --dataset {ds}"
  for win_size in [1, 4, 16, 32]:
    for n_layer in [1, 2, 4, 8]:
      for repeat_ind in range(5):
        for ds in ["china_stock_qlib_adj", "qlib_cn_stock"]:
          cmds.append(cmd.format(repeat_ind=repeat_ind,
            n_layer=n_layer, win_size=win_size, ds=ds))
  return cmds


def mlp_raw():
  cmds = []
  cmd = "python mlp.py --name raw --repeat-ind {repeat_ind} --n-layer {n_layer} --win-size {win_size} --dataset {ds}"
  for win_size in [1, 4, 16, 32]:
    for n_layer in [1, 2, 4, 8]:
      for repeat_ind in range(5):
        for ds in ["china_stock_qlib_adj", "qlib_cn_stock"]:
          cmds.append(cmd.format(repeat_ind=repeat_ind,
            n_layer=n_layer, win_size=win_size, ds=ds))
  return cmds


def plot_mlp():
  cmds = []
  cmd = "python plot.py --name {name} --train-ds {ds} --val-ds {ds}"
  for name in ["official"]:
    for ds in ["china_stock_qlib_adj", "qlib_cn_stock"]:
      cmds.append(cmd.format(name=name, ds=ds))
  return cmds


def plot_mlp_cross():
  cmds = []
  cmd = "python plot.py --name {name} --train-ds {tds} --plot-ds {pds}"
  for name in ["raw", "official", "zscorenorm"]:
    for tds in ["china_stock_qlib_adj", "qlib_cn_stock"]:
      for pds in ["china_stock_qlib_adj", "qlib_cn_stock"]:
        cmds.append(cmd.format(name=name, tds=tds, pds=pds))
  return cmds

funcs = {
  "mlp_zscorenorm" : mlp_zscorenorm,
  "mlp_raw" : mlp_raw,
  "plot_mlp" : plot_mlp,
  "plot_mlp_cross" : plot_mlp_cross,
  "mlp_official" : mlp_official,
  }

print(args.gpu)
if args.gpu != "-1":
  gpus = args.gpu.split('/')
  slots = [[] for _ in gpus]
  for i, cmd in enumerate(funcs[args.func]()):
    gpu = gpus[i % len(gpus)]
    slots[i % len(gpus)].append(f"{cmd} --gpu-id {gpu}")
  for s in slots:
    cmd = " && ".join(s) + " &"
    print(cmd)
    os.system(cmd)
elif args.gpu == "-1":
  for cmd in funcs[args.func]():
    print(cmd)
    os.system(cmd)
