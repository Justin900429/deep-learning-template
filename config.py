import os.path as osp
import pprint
from collections.abc import MutableMapping

from colorama import Fore, Style
from tabulate import tabulate
from yacs.config import CfgNode as CN


def create_cfg():
    """
    Setup a config file with yacs, check here: https://github.com/rbgirshick/yacs.
    Please feel free to add more configuration options as needed.
    """
    cfg = CN()
    cfg._BASE_ = None  # This is used to successed the base configuration
    cfg.PROJECT_DIR = None  # Name of the project directory for saving logs, checkpoints, etc.
    cfg.LOG_DIR = "logs"  # Directory for saving logs
    cfg.PROJECT_LOG_WITH = [
        "tensorboard"
    ]  # Log with different trackers. Please check accelerate for more details.

    # ==========  Model   ==========
    cfg.MODEL = CN()
    cfg.MODEL.IN_CHANNELS = 3
    cfg.MODEL.BASE_DIM = 16
    cfg.MODEL.NUM_CLASSES = 10

    # ======= LOSS ========
    cfg.LOSS = CN()
    cfg.LOSS.LABEL_SMOOTHING = 0.0

    # ======= Dataset =======
    cfg.DATA = CN()
    cfg.DATA.ROOT = None

    # ======= Training =======
    cfg.TRAIN = CN()
    cfg.TRAIN.BATCH_SIZE = 32
    cfg.TRAIN.VAL_FREQ = 1  # Validate every epoch, change to suit your needs
    cfg.TRAIN.EPOCHS = 50
    cfg.TRAIN.NUM_WORKERS = 4
    cfg.TRAIN.ACCUM_ITER = 0  # Gradient accumulation controlled with accelerate
    cfg.TRAIN.MIXED_PRECISION = "no"  # Whether to use mixed precision training
    cfg.TRAIN.LR = 0.0003
    cfg.TRAIN.WEIGHT_DECAY = 0.0001
    cfg.TRAIN.LOG_EVERY_STEP = 100  # Log every 100 steps for training
    cfg.TRAIN.RESUME_CHECKPOINT = None  # Path to the checkpoint for resuming training

    # ======= Evaluation =======
    cfg.EVAL = CN()
    cfg.EVAL.NUM_WORKERS = 4
    cfg.EVAL.BATCH_SIZE = cfg.TRAIN.BATCH_SIZE
    cfg.EVAL.LOG_EVERY_STEP = 50  # Log every 50 steps for evaluation

    return cfg


def pretty_print_cfg(cfg: CN) -> str:
    def _indent(s_, num_spaces):
        s = s_.split("\n")
        if len(s) == 1:
            return s_
        first = s.pop(0)
        s = [(num_spaces * " ") + line for line in s]
        s = "\n".join(s)
        s = first + "\n" + s
        return s

    r = ""
    s = []
    for k, v in sorted(cfg.items()):
        seperator = "\n" if isinstance(v, CN) else " "
        attr_str = "{}:{}{}".format(
            str(k),
            seperator,
            pretty_print_cfg(v) if isinstance(v, CN) else pprint.pformat(v),
        )
        attr_str = _indent(attr_str, 2)
        s.append(attr_str)
    r += "\n".join(s)
    return r


def flatten(dictionary, parent_key="", separator="."):
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))

    return dict(items)


def config_to_str(cfg: CN) -> str:
    """This ensure all logger can save the hyperparameters in a readable format"""
    valid_dict = dict()
    for k, v in flatten(dict(cfg)).items():
        valid_dict[k] = pprint.pformat(v) if not isinstance(v, (str, float, int, bool)) else v
    return valid_dict


def show_config(cfg: CN):
    table = tabulate(
        {"Configuration": [pretty_print_cfg(cfg)]},
        headers="keys",
        tablefmt="fancy_grid",
    )
    print(f"{Fore.BLUE}", end="")
    print(table)
    print(f"{Style.RESET_ALL}", end="")


def merge_possible_with_base(cfg: CN, config_path: str):
    with open(config_path, "r") as f:
        new_cfg = cfg.load_cfg(f)
    if "_BASE_" in new_cfg:
        cfg.merge_from_file(osp.join(osp.dirname(config_path), new_cfg._BASE_))
    cfg.merge_from_other_cfg(new_cfg)
