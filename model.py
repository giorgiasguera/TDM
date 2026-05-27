import torch.nn as nn
from torch import Tensor

from initialization import initialize_model
from vocabulary import Vocabulary, PAD_IDX, BOS_IDX, EOS_IDX
from text_encoder import TextEncoder
from diffusion import GaussianDiffusion
from denoiser import TDMDenoiser
from loss import TDMLoss

TARGET_PAD = 0.0


class Model(nn.Module):
    """
    Text-Driven Diffusion Model per Sign Language Production.
    """

    def __init__(self,
                 encoder: TextEncoder,
                 diffusion: GaussianDiffusion,
                 src_vocab: Vocabulary,
                 cfg: dict):
        super().__init__()

        self.encoder = encoder
        self.diffusion = diffusion
        self.src_vocab = src_vocab

        self.bos_index = BOS_IDX
        self.pad_index = PAD_IDX
        self.eos_index = EOS_IDX
        self.target_pad = TARGET_PAD

        self.trg_size = cfg["model"].get("trg_size", 150)

    # forward unificato per training e inference 
    def forward(self,
                src: Tensor,
                trg_input: Tensor,
                src_mask: Tensor,
                trg_mask: Tensor,
                is_train: bool) -> Tensor:

        #codifica testo sorgente
        encoder_output = self.encode(src=src, src_mask=src_mask)

        # forward o reverse diffusion
        diffusion_output = self.run_diffusion(
            is_train = is_train,
            encoder_output = encoder_output,
            trg_input = trg_input,
            src_mask = src_mask,
            trg_mask = trg_mask,
        )

        return diffusion_output

    #text_encoder
    def encode(self, src: Tensor, src_mask: Tensor) -> Tensor:
        return self.encoder(src_tokens=src, src_mask=src_mask)

    def run_diffusion(self,
                      is_train: bool,
                      encoder_output : Tensor,
                      trg_input: Tensor,
                      src_mask: Tensor,
                      trg_mask: Tensor) -> Tensor:
        
        return self.diffusion(
            encoder_output = encoder_output,
            input_3d = trg_input,
            src_mask = src_mask,
            trg_mask = trg_mask,
            is_train = is_train,
        )

    def get_loss_for_batch(self, batch: dict, loss_function: TDMLoss) -> Tensor:

        src_tokens = batch["src_tokens"]
        src_mask = batch["src_mask"]
        trg_input = batch["skeletons"]
        trg_mask = batch["trg_mask"]

        # forward in modalità training, ottieni p'0 (diffusion output)
        skel_out = self.forward(
            src = src_tokens,
            trg_input = trg_input,
            src_mask = src_mask,
            trg_mask = trg_mask,
            is_train = True,
        )

        # Calcola la loss combinata tra p'0 e la posa gt
        batch_loss = loss_function(skel_out, trg_input[:, :, :self.trg_size])

        return batch_loss


def build_model(cfg: dict, src_vocab: Vocabulary) -> Model:
    """
    Costruisce il modello dallo YAML
    """
    model_cfg = cfg["model"]
    diff_cfg = model_cfg["diffusion"]

    encoder = TextEncoder.from_config(cfg, src_vocab) # l'embedding sta qua

    denoiser = TDMDenoiser.from_config(cfg)

    diffusion = GaussianDiffusion(
        denoiser = denoiser,
        timesteps = int(diff_cfg.get("timesteps", 1000)),
        sampling_timesteps = int(diff_cfg.get("sampling_timesteps", 5)),
        scale = float(diff_cfg.get("scale", 1.0)),
    )

    #modello completo
    model = Model(
        encoder   = encoder,
        diffusion = diffusion,
        src_vocab = src_vocab,
        cfg       = cfg,
    )

    initialize_model(model, cfg["model"], src_padding_idx=PAD_IDX, trg_padding_idx=0)

    return model
