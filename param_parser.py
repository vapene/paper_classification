"""Getting the parameters from the commandline."""

import argparse
from texttable import Texttable


def parameter_parser():

    p = argparse.ArgumentParser()
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--epochs", default=200, type=int)
    p.add_argument("--batch_size", default=80, type=int)
    p.add_argument("--device", default="0")
    p.add_argument('--models', nargs='+', default= ["M3", "M5", "M7", "resnet"])

    return p.parse_args()

def tab_printer(args):
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]] + [[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(t.draw())