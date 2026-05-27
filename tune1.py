import os
os.environ["WANDB_MODE"] = "disabled"
import optuna
import torch
import wandb
import shutil

from helpers import load_config
from data import build_dataloaders
from vocabulary import Vocabulary
from model import build_model
from training import TrainManager


def objective(trial):
    cfg = load_config("./Configs/TDM.yaml")

    batch_size   = trial.suggest_categorical("batch_size",          [16, 32])
    weight_decay = trial.suggest_categorical("weight_decay",        [0.0, 0.0001])
    num_layers   = trial.suggest_categorical("num_layers",          [2, 4, 6])
    num_heads    = trial.suggest_categorical("num_heads",           [4, 8, 16])
    hidden_size  = trial.suggest_categorical("hidden_size",         [512, 1024])
    sampling_ts  = trial.suggest_categorical("sampling_timesteps",  [3, 5, 10])
    ff_size      = hidden_size * 2

    cfg["training"]["batch_size"]   = batch_size
    cfg["training"]["weight_decay"] = weight_decay

    cfg["model"]["encoder"]["num_layers"]  = num_layers
    cfg["model"]["encoder"]["num_heads"]   = num_heads
    cfg["model"]["encoder"]["hidden_size"] = hidden_size
    cfg["model"]["encoder"]["ff_size"]     = ff_size
    cfg["model"]["encoder"]["embeddings"]["embedding_dim"] = hidden_size

    cfg["model"]["diffusion"]["num_layers"]  = num_layers
    cfg["model"]["diffusion"]["num_heads"]   = num_heads
    cfg["model"]["diffusion"]["hidden_size"] = hidden_size
    cfg["model"]["diffusion"]["ff_size"]     = ff_size
    cfg["model"]["diffusion"]["embeddings"]["embedding_dim"] = hidden_size
    cfg["model"]["diffusion"]["sampling_timesteps"] = sampling_ts

    cfg["training"]["model_dir"] = f"./Models/Tune_{trial.number}"
    cfg["training"]["continue"]  = True
    cfg["training"]["overwrite"] = False

    wandb.init(mode="disabled")

    try:
        loaders   = build_dataloaders(cfg=cfg, num_workers=2)
        src_vocab = Vocabulary.load_or_build(
            vocab_file = cfg["data"]["src_vocab"],
            text_file  = cfg["data"]["train"] + ".text")
        model   = build_model(cfg=cfg, src_vocab=src_vocab)
        cfg["training"]["validation_freq"] = 1000
        trainer = TrainManager(config=cfg, model=model,
                               src_vocab=src_vocab, test=False)
        trainer.max_steps = 40000
        trainer.train_and_validate(
            train_loader=loaders["train"],
            val_loader=loaders["dev"])

    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print(f"Trial {trial.number} fallito per OOM. Pruning.")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise optuna.TrialPruned()
        raise

    finally:
        wandb.finish()   # eseguito sempre, anche in caso di eccezione
         # elimina i checkpoint del trial appena finito
        import shutil
        trial_dir = f"./Models/Tune_{trial.number}"
        if os.path.exists(trial_dir):
            shutil.rmtree(trial_dir)

    return trainer.best_ckpt_score


if __name__ == "__main__":
    study = optuna.create_study(
        study_name     = "TDM_arch",
        storage        = "sqlite:///optuna_arch.db",
        load_if_exists = True,
        direction      = "minimize",
        pruner         = optuna.pruners.MedianPruner(
            n_startup_trials=4,
            n_warmup_steps=12
        )
    )
    study.optimize(objective, n_trials=25)

    print("\nMigliori parametri trovati:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")