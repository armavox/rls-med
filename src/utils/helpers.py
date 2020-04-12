import json
import logging
import logging.config
import os
import re
import yaml
from argparse import ArgumentParser, Namespace
from datetime import datetime

import numpy as np
import torch


log = logging.getLogger("utils.helpers")


class LoggingFilter(logging.Filter):
    def filter(self, record):
        allow = record.name in logging_config["loggers"]
        return allow


def arguments():
    parser = ArgumentParser()
    parser.add_argument(
        "-c", "--train-config", type=str, help="Path to the yaml file with the training parameters"
    )
    return parser.parse_args()


def config_snapshot(name: str, config: dict, old_config_path: str):
    if os.path.exists(old_config_path):
        with open(old_config_path) as f:
            old_config = json.load(f)
        shared_items = {
            k: old_config[k] for k in old_config if k in config and old_config[k] == config[k]
        }
        if len(shared_items) == len(config):
            return True
        log.warning(f"{name} configs are not similar. Snapshot refreshed")
    log.warning(f"{name} config file doesn't exist. Snapshot created")
    with open(old_config_path, "w") as f:
        json.dump(config, f)
    return False


def load_params_namespace(yaml_path: str) -> Namespace:
    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
                   [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
                   |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
                   |\\.[0-9_]+(?:[eE][-+][0-9]+)?
                   |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
                   |[-+]?\\.(?:inf|Inf|INF)
                   |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))
    with open(yaml_path) as config_file:
        config = yaml.load(config_file, Loader=loader)
        return Namespace(**config)


def make_np(x: torch.Tensor) -> np.ndarray:
    if isinstance(x, torch.autograd.Variable):
        x = x.data
    x = x.cpu().numpy()
    return x


def makedirs(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def random_seed_init(random_seed: bool = None, cuda: bool = False):
    if random_seed:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        if cuda:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def set_logging_config(logging_config_path: str, ws_path: str):
    with open(logging_config_path, "r") as f:
        global logging_config
        logging_config = yaml.safe_load(f.read())
        now = datetime.now().strftime("%Y-%m-%d-%H:%M")
        logdir = os.path.join(ws_path, "artifacts", "logs")
        makedirs(logdir)
        logging_config["handlers"]["file"]["filename"] = f"{logdir}/{now}.log"
        logging.config.dictConfig(logging_config)
