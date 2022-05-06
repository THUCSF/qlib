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


def train_rnn():
    cmds = []
    cmd = "python main.py --market {market} --repeat-ind {repeat_ind} --loss-type {loss_type} --n-layer {n_layer} --train-end {train_end} --test-start {test_start} --test-end {test_end} --top-k 50 --n-drop 10 --n1-epoch 50 --strict-validation 0"

    
    for repeat_ind in range(5):
        for train_end in range(2014, 2020):  # 4
            test_end = test_start = train_end + 1
            for data_type in ["raw"]:
                for market in ["main"]:
                    for loss_type in ["quantile-last"]:
                        for n_layer in [4]:
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

    for train_end in range(2014, 2020):  # 6
        for repeat_ind in range(5):  # 35 models to be trained
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
    cmd = "python evaluate.py --market {market} --repeat-ind {repeat_ind} --loss-type {loss_type} --n-layer {n_layer} --train-start {train_start} --train-end {train_end} --test-start {test_start} --test-end {test_end} --model-dir expr/main_raw_pc-1_50-10/rgr-all_l2"
    for repeat_ind in range(5):  # 1500 models to be trained
        for train_end in range(2013, 2020):  # 4
            test_end = test_start = train_end + 1
            train_start = train_end
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
                                    train_start=train_start,
                                    train_end=train_end,
                                    test_start=test_start,
                                    test_end=test_end,
                                )
                            )
    return cmds


def eval_rnn_roll():
    cmds = []
    cmd = "python evaluate_roll.py --market {market} --repeat-ind {repeat_ind} --loss-type {loss_type} --n-layer {n_layer} --train-end {train_end} --test-start {test_start} --test-end {test_end} --model-dir expr/main_raw_pc-1_rolling/rgr-all_l2"
    for repeat_ind in range(5):  # 1500 models to be trained
        for train_end in range(2014, 2020):  # 4
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

funcs = {
    "train_rnn": train_rnn,
    "train_rnn_roll": train_rnn_roll,
    "eval_rnn": eval_rnn,
    "eval_rnn_roll": eval_rnn_roll
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
