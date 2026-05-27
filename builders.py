import torch
import torch.nn as nn

from helpers import ConfigurationError
from typing import Optional, Callable, Generator, Tuple
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, ExponentialLR, LRScheduler

def build_gradient_clipper(config: dict) -> Optional[Callable]:
    """
    Define the function for gradient clipping as specified in configuration.
    If not specified, returns None.

    :param config: training configuration dictionary
    :return: clipping function (in-place) or None
    """
    clip_grad_fun = None
    if "clip_grad_val" in config.keys():
        clip_value = config["clip_grad_val"]
        clip_grad_fun = lambda params: \
            nn.utils.clip_grad_value_(parameters=params,
                                      clip_value=clip_value)
    elif "clip_grad_norm" in config.keys():
        max_norm = config["clip_grad_norm"]
        clip_grad_fun = lambda params: \
            nn.utils.clip_grad_norm_(parameters=params, max_norm=max_norm)

    if "clip_grad_val" in config.keys() and "clip_grad_norm" in config.keys():
        raise ConfigurationError(
            "You can only specify either clip_grad_val or clip_grad_norm.")

    return clip_grad_fun


def build_optimizer(config: dict, parameters: Generator) -> Optimizer:
    """
    Create an optimizer for the given parameters as specified in config.

    :param config: training configuration dictionary
    :param parameters: model parameters
    :return: optimizer
    """
    optimizer_name = config.get("optimizer", "adam").lower()
    learning_rate  = config.get("learning_rate", 1.0e-3) # in sign-idd il default è 3.0e-4
    weight_decay   = config.get("weight_decay", 0)

    if optimizer_name == "adam":
        adam_betas = config.get("adam_betas", (0.9, 0.999))
        optimizer = torch.optim.Adam(parameters, weight_decay=weight_decay,
                                     lr=learning_rate, betas=adam_betas)
    elif optimizer_name == "adagrad":
        optimizer = torch.optim.Adagrad(parameters, weight_decay=weight_decay,
                                        lr=learning_rate)
    elif optimizer_name == "adadelta":
        optimizer = torch.optim.Adadelta(parameters, weight_decay=weight_decay,
                                         lr=learning_rate)
    elif optimizer_name == "rmsprop":
        optimizer = torch.optim.RMSprop(parameters, weight_decay=weight_decay,
                                        lr=learning_rate)
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(parameters, weight_decay=weight_decay,
                                    lr=learning_rate)
    else:
        raise ConfigurationError("Invalid optimizer. Valid options: 'adam', "
                                 "'adagrad', 'adadelta', 'rmsprop', 'sgd'.")
    return optimizer


def build_scheduler(config: dict, optimizer: Optimizer,
                    scheduler_mode: str, hidden_size: int = 0) -> Tuple[Optional[LRScheduler], Optional[str]]:
    """
    Create a learning rate scheduler if specified in config.

    :param config: training configuration dictionary
    :param optimizer: optimizer
    :param scheduler_mode: "min" or "max"
    :param hidden_size: encoder hidden size (for Noam, not used in TDM)
    :return: (scheduler, scheduler_step_at)
    """
    scheduler, scheduler_step_at = None, None
    if "scheduling" in config.keys() and config["scheduling"]:
        if config["scheduling"].lower() == "plateau":
            scheduler = ReduceLROnPlateau(
                optimizer=optimizer,
                mode=scheduler_mode,
                threshold_mode='abs',
                threshold=1e-8,
                factor=config.get("decrease_factor", 0.7), # in sign-idd il default è 0.1
                patience=config.get("patience", 7)) # in sign-idd il default è 10
            scheduler_step_at = "validation"
        elif config["scheduling"].lower() == "decaying":
            scheduler = StepLR(
                optimizer=optimizer,
                step_size=config.get("decaying_step_size", 1))
            scheduler_step_at = "epoch"
        elif config["scheduling"].lower() == "exponential":
            scheduler = ExponentialLR(
                optimizer=optimizer,
                gamma=config.get("decrease_factor", 0.99))
            scheduler_step_at = "epoch"
        elif config["scheduling"].lower() == "noam":
            factor = config.get("learning_rate_factor", 1)
            warmup = config.get("learning_rate_warmup", 4000)
            scheduler = NoamScheduler(hidden_size=hidden_size, factor=factor,
                                      warmup=warmup, optimizer=optimizer)
            scheduler_step_at = "step"
    return scheduler, scheduler_step_at


class NoamScheduler:
    """
    The Noam learning rate scheduler used in "Attention is all you need"
    See Eq. 3 in https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self, hidden_size: int, optimizer: torch.optim.Optimizer,
                 factor: float = 1, warmup: int = 4000):
        """
        Warm-up, followed by learning rate decay.

        :param hidden_size:
        :param optimizer:
        :param factor: decay factor
        :param warmup: number of warmup steps
        """
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.hidden_size = hidden_size
        self._rate = 0

    def step(self):
        """Update parameters and rate"""
        self._step += 1
        rate = self._compute_rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate

    def _compute_rate(self):
        """Implement `lrate` above"""
        step = self._step
        return self.factor * \
            (self.hidden_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))

    #pylint: disable=no-self-use
    def state_dict(self):
        return None