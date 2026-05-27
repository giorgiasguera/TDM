import math
import torch
import torch.nn as nn
from torch import Tensor

from transformer_layers import TransformerDecoderLayer, PositionalEncoding

# converte il timestep scalare t in un vettore (B, hidden_size),usato nella condizione g
class SinusoidalPositionEmbeddings(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: Tensor) -> Tensor:
        """
        :param time: (B,) timestep interi
        :return: (B, dim) embedding sinusoidale
        """
        device = time.device
        half_dim = self.dim // 2

        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :] # (B, half_dim)
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)  # (B, dim)
        return embeddings


class TDMDenoiser(nn.Module):

    def __init__(self,
                 trg_size:int = 150,
                 hidden_size: int = 1024,
                 ff_size: int = 2048,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 emb_dropout: float = 0.1,
                 freeze: bool = False,
                 **kwargs):
        super().__init__()

        self.trg_size = trg_size
        self.hidden_size = hidden_size

        # linear Embedding Layer
        self.trg_embed = nn.Linear(trg_size, hidden_size)

        # positional Encoding sinusoidale (non apprendibile)
        self.pe = PositionalEncoding(hidden_size)

        # dropout sulle pose dopo PE
        self.emb_dropout = nn.Dropout(p=emb_dropout)

        # dropout sulla condizione g — [7] applicato a encoder_output + time_embed
        self.pos_drop = nn.Dropout(p=emb_dropout)

        # time_mlp: timestep t → embedding della condizione 
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_size),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )

        #stack di TransformerDecoderLayer (qin sign-idd nell' if num_layers == 2)
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                size = hidden_size,
                ff_size = ff_size,
                num_heads = num_heads,
                dropout = dropout,
            )
            for _ in range(num_layers)
        ])

        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.output_layer = nn.Linear(hidden_size, trg_size, bias=False)

        if freeze:
            for p in self.parameters():
                p.requires_grad = False

    # implementa eq 2
    def forward(self,
                encoder_output: Tensor,
                trg_embed: Tensor,
                src_mask: Tensor = None,
                trg_mask: Tensor = None,
                t: Tensor = None) -> Tensor:
        
        time_embed = self.time_mlp(t)[:, None, :]
        condition= encoder_output + time_embed #cond g

        condition= self.pos_drop(condition)

        x = self.trg_embed(trg_embed) # linear
        x = self.pe(x)
        x = self.emb_dropout(x)

        for layer in self.layers:
            x = layer(
                x = x,
                memory = condition,
                src_mask = src_mask,    # padding mask testo 
                trg_mask = trg_mask,    #padding mask pose
            )

        x = self.layer_norm(x)
        output = self.output_layer(x)

        return output


    @classmethod
    def from_config(cls, cfg: dict) -> "TDMDenoiser":

        diff_cfg = cfg["model"]["diffusion"]
        emb_cfg  = diff_cfg["embeddings"]

        return cls(
            trg_size = int(cfg["model"].get("trg_size", 150)),
            hidden_size = int(diff_cfg.get("hidden_size", 1024)),
            ff_size = int(diff_cfg.get("ff_size", 2048)),
            num_layers  = int(diff_cfg.get("num_layers", 4)),
            num_heads = int(diff_cfg.get("num_heads", 8)),
            dropout= float(diff_cfg.get("dropout", 0.1)),
            emb_dropout = float(emb_cfg.get("dropout",0.1)),
        )

    def __repr__(self) -> str:
        return "%s(num_layers=%r, num_heads=%r)" % (
            self.__class__.__name__,
            len(self.layers),
            self.layers[0].self_attn.num_heads,
        )