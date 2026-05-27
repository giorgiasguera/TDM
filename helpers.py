import glob
import os
import os.path
import errno
import shutil
import random
import logging
from logging import Logger
from typing import Optional
import numpy as np

import torch
import torch.nn as nn
import yaml

# eccezione sollevata quando il file YAML contiene valori non validi
class ConfigurationError(Exception):
    """ Custom exception for misspecifications of configuration """


def load_config(path: str = "Configs/TDM.yaml") -> dict:
    with open(path, "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg

#scrive la ocnfigurazione nel log
def log_cfg(cfg: dict, logger: Logger, prefix: str = "cfg") -> None:

    for k, v in cfg.items():
        if isinstance(v, dict):
            log_cfg(v, logger, prefix=".".join([prefix, k]))
        else:
            logger.info("{:34s} : {}".format(".".join([prefix, k]), v))


def make_model_dir(model_dir: str, overwrite: bool = False, model_continue: bool = False) -> str:
    if os.path.isdir(model_dir):
        if model_continue: # per riprendere il training
            return model_dir
        if not overwrite:
            raise FileExistsError(
                "Model directory exists and overwriting is disabled.")
        for file in os.listdir(model_dir): #se si può fare l'overwrite rimuove ricorsivamente le vecchie dir e ricomincia da una vuota
            file_path = os.path.join(model_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        shutil.rmtree(model_dir, ignore_errors=True)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    return model_dir

# crea logger che scrive su file e console
def make_logger(model_dir: str, log_file: str = "train.log") -> Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False #fix qua

    formatter = logging.Formatter('%(asctime)s %(message)s')

    fh = logging.FileHandler("{}/{}".format(model_dir, log_file))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)

    logging.getLogger("").addHandler(sh)

    # fix
    if logger.hasHandlers():
      logger.handlers.clear()

    logger.addHandler(fh)
    logger.addHandler(sh) #fix

    logger.info("Text-Driven Diffusion Model for Sign Language Production")

    return logger


#per train uguale
def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def freeze_params(module: nn.Module) -> None:
    for _, p in module.named_parameters():
        p.requires_grad = False


def get_latest_checkpoint(ckpt_dir: str, post_fix: str = "_every") -> Optional[str]:
    list_of_files = glob.glob("{}/*{}.ckpt".format(ckpt_dir, post_fix))
    if list_of_files:
        return max(list_of_files, key=os.path.getctime)
    return None


def load_checkpoint(path: str, use_cuda: bool = True) -> dict:
    assert os.path.isfile(path), "Checkpoint %s not found" % path
    map_location = "cuda" if (use_cuda and torch.cuda.is_available()) else "cpu"
    checkpoint = torch.load(path, map_location=map_location, weights_only=False) #fix
    return checkpoint


def symlink_update(target: str, link_name: str) -> None:
    try:
        os.symlink(target, link_name)
    except FileExistsError as e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(target, link_name)
        else:
            raise e

# (parent_idx, child_idx)
def getSkeletalModelStructure():
    return (
        # Testa e collo
        (1, 0), (1, 1), (1, 2),
        # Braccio sinistro
        (2, 3), (3, 4),
        # Spalla destra → polso destro
        (1, 5), (5, 6), (6, 7), (7, 8),
        # Dita mano destra (dal polso = joint 8)
        (8, 9),  (9, 10),  (10, 11), (11, 12),
        (8, 13), (13, 14), (14, 15), (15, 16),
        (8, 17), (17, 18), (18, 19), (19, 20),
        (8, 21), (21, 22), (22, 23), (23, 24),
        (8, 25), (25, 26), (26, 27), (27, 28),
        # Dita mano sinistra (dal polso = joint 4)
        (4, 29),  (29, 30),  (30, 31),  (31, 32),  (32, 33),
        (29, 34), (34, 35),  (35, 36),  (36, 37),
        (29, 38), (38, 39),  (39, 40),  (40, 41),
        (29, 42), (42, 43),  (43, 44),  (44, 45),
        (29, 46), (46, 47),  (47, 48),  (48, 49),
    )
