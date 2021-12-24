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


def train_mlp():
  cmds = []
  cmd = "python main.py --data-type {data_type} --market {market} --repeat-ind {repeat_ind} --loss-type {loss_type} --n-layer {n_layer} --win-size {win_size} --train-start {train_start} --train-end {train_end} --valid-start {valid_start} --valid-end {valid_end} --test-start {test_start} --test-end {test_end}"

  # raw and zscorenorm
  for train_start in range(2011, 2018):
    train_end = train_start + 2
    valid_start = valid_end = train_end + 1
    test_start = test_end = valid_end + 1

    for data_type in ["raw", "zscorenorm"]:
      for market in ["csi300", "main"]:
        for loss_type in ["rgr", "br", "cls"]:
          if data_type == "raw" and market == "csi300" and loss_type == "rgr":
            continue
          for win_size in [1, 4, 8, 16, 32]:
            for n_layer in [1, 4, 8]:
              for repeat_ind in range(5):
                cmds.append(cmd.format(repeat_ind=repeat_ind,
                  data_type=data_type, market=market, loss_type=loss_type,
                  n_layer=n_layer, win_size=win_size,
                  train_start=train_start, train_end=train_end,
                  valid_start=valid_start, valid_end=valid_end,
                  test_start=test_start, test_end=test_end))

    # alph158
    for market in ["csi300", "main"]:
      for loss_type in ["rgr", "br", "cls"]:
        for n_layer in [1, 4, 8]:
          for repeat_ind in range(5):
            cmds.append(cmd.format(repeat_ind=repeat_ind,
              data_type="alpha158", market=market, loss_type=loss_type,
              n_layer=n_layer, win_size=1,
              train_start=train_start, train_end=train_end,
              valid_start=valid_start, valid_end=valid_end,
              test_start=test_start, test_end=test_end))
  return cmds


funcs = {
  "train_mlp" : train_mlp,
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
