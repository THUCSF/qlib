"""Easy execution script.
Running scripts at different gpu slots:
python submit.py --gpu 0/1/2/3/...

Running scripts at different multi-gpu slots:
python submit.py --gpu 0,1/2,3/4,5/...

Running scripts with default gpu:
python submit.py --gpu -1
"""
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--func", default="test", type=str)
parser.add_argument("--gpu", default="-1")
args = parser.parse_args()


def train_mlp():
    cmds = []
    cmd = "python main.py --data-type {data_type} --market {market} --repeat-ind {repeat_ind} --loss-type {loss_type} --n-layer {n_layer} --win-size {win_size}  --train-end {train_end} --valid-start {valid_start} --valid-end {valid_end} --test-start {test_start} --test-end {test_end}"

    for repeat_ind in range(5):  # 1500 models to be trained
        for train_end in range(2012, 2017):
            valid_start = valid_end = train_end + 2
            test_start = test_end = valid_end + 2
            for data_type in ["raw"]:
                for market in ["csi300", "main"]:
                    for loss_type in ["rgr", "cls"]:
                        for win_size in [1, 4, 8, 16, 32]:
                            for n_layer in [1, 4, 8]:
                                cmds.append(
                                    cmd.format(
                                        repeat_ind=repeat_ind,
                                        data_type=data_type,
                                        market=market,
                                        loss_type=loss_type,
                                        n_layer=n_layer,
                                        win_size=win_size,
                                        train_end=train_end,
                                        valid_start=valid_start,
                                        valid_end=valid_end,
                                        test_start=test_start,
                                        test_end=test_end,
                                    )
                                )

        # alph158
        """
    for market in ["csi300", "main"]:
      for loss_type in ["rgr", "br", "cls"]:
        for n_layer in [1, 4, 8]:
          for repeat_ind in range(5):
            cmds.append(cmd.format(repeat_ind=repeat_ind,
              data_type="alpha158", market=market, loss_type=loss_type,
              n_layer=n_layer, win_size=1,
              train_end=train_end,
              valid_start=valid_start, valid_end=valid_end,
              test_start=test_start, test_end=test_end))
    """
    return cmds


def train_rnn():
    cmds = []
    cmd = "python main.py --market {market} --repeat-ind {repeat_ind} --loss-type {loss_type} --n-layer {n_layer} --train-end {train_end} --test-start {test_start} --test-end {test_end} --top-k 50 --n-drop 10 --n1-epoch 50 --strict-validation 1"
    for repeat_ind in range(5):
        for train_end in range(2013, 2020):  # 4
            test_end = test_start = train_end + 1
            for data_type in ["raw"]:
                for market in ["main"]:
                    for loss_type in ["rgr-all", "rgr-last"]:
                        for n_layer in [2]:
                            cmds.append(
                                cmd.format(
                                    repeat_ind=repeat_ind,
                                    data_type=data_type,
                                    market=market,
                                    loss_type=loss_type,
                                    n_layer=n_layer,
                                    train_end=train_end,
                                    test_start=test_start,
                                    test_end=test_end,
                                )
                            )
    return cmds


def train_rnn_roll():
    cmds = []
    cmd = "python main_roll.py --market {market} --repeat-ind {repeat_ind} --loss-type {loss_type} --n-layer {n_layer} --train-end {train_end} --test-start {test_start} --test-end {test_end} --top-k 50 --n-drop 10 --n1-epoch 50 --n2-epoch 10 --strict-validation 0"
    for repeat_ind in range(5):  # 35 models to be trained
        for train_end in range(2013, 2020):  # 7
            test_end = test_start = train_end + 1
            for data_type in ["raw"]:
                for market in ["main"]:
                    for loss_type in ["rgr-all"]:
                        for n_layer in [2]:
                            cmds.append(
                                cmd.format(
                                    repeat_ind=repeat_ind,
                                    data_type=data_type,
                                    market=market,
                                    loss_type=loss_type,
                                    n_layer=n_layer,
                                    train_end=train_end,
                                    test_start=test_start,
                                    test_end=test_end,
                                )
                            )
    return cmds


def eval_rnn():
    cmds = []
    cmd = "python evaluate.py --market {market} --repeat-ind {repeat_ind} --loss-type {loss_type} --n-layer {n_layer} --train-end {train_end} --test-start {test_start} --test-end {test_end}"
    for repeat_ind in range(1):  # 1500 models to be trained
        for train_end in range(2013, 2021):  # 4
            test_end = test_start = train_end + 1
            for data_type in ["raw"]:
                for market in ["main"]:
                    for loss_type in ["rgr"]:
                        for n_layer in [1]:
                            cmds.append(
                                cmd.format(
                                    repeat_ind=repeat_ind,
                                    data_type=data_type,
                                    market=market,
                                    loss_type=loss_type,
                                    n_layer=n_layer,
                                    train_end=train_end,
                                    test_start=test_start,
                                    test_end=test_end,
                                )
                            )
    return cmds


funcs = {
    "train_mlp": train_mlp,
    "train_rnn": train_rnn,
    "train_rnn_roll": train_rnn_roll,
    "eval_rnn": eval_rnn,
}

print(args.gpu)
if args.gpu != "-1":
    gpus = args.gpu.split("/")
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
