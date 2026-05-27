#import torch
import torch.nn as nn
from torch import Tensor

from transformer_layers import TransformerEncoderLayer, PositionalEncoding
from vocabulary import Vocabulary, PAD_IDX

class TextEncoder(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        hidden_size : int = 1024,
        ff_size: int = 2048,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        emb_dropout: float = 0.1,
        scale: bool = False,
        pad_idx: int  = PAD_IDX,
        freeze: bool = False,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.scale = scale

        # word embedding
        self.embedding = nn.Embedding(
            vocab_size, hidden_size, padding_idx=pad_idx # padding_idx=pad_idx: il vettore <pad> rimane a zero e non riceve gradiente
        )

        # positional encoding sinusoidale, registrato come buffer, si sposta su GPU con .to(device).
        self.pe = PositionalEncoding(hidden_size)

        # dropout sull'embedding
        self.emb_dropout = nn.Dropout(p=emb_dropout)

        # stack di TransformerEncoderLayer
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                size = hidden_size,
                ff_size = ff_size,
                num_heads = num_heads,
                dropout = dropout,
            )
            for _ in range(num_layers)
        ])

        # layerNorm finale
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)

        if freeze:
            for p in self.parameters():
                p.requires_grad = False


    # processa una sequenza di token ID e restituisce hidden states.
    def forward(self, src_tokens: Tensor, src_mask: Tensor) -> Tensor:
        # word embedding
        x = self.embedding(src_tokens)

        # scaling opzionale serve a bilanciare la norma degli embedding con quella del PE quando hidden_size è grande. 
        #disabilitato
        if self.scale:
            x = x * (self.hidden_size ** 0.5)

        # positional encoding
        x = self.pe(x)

        # dropout embedding
        x = self.emb_dropout(x)

        # stack di TransformerEncoderLayer
        for layer in self.layers:
            x = layer(x, src_mask)

        # layerNorm finale 
        return self.layer_norm(x) # (B, L, hidden_size) primo pezzo condizione g

   #costruisce il TextEncoder leggendo i parametri dallo yaml
    @classmethod
    def from_config(cls, cfg: dict, vocab: Vocabulary) -> "TextEncoder":

        enc_cfg = cfg["model"]["encoder"]
        emb_cfg = enc_cfg["embeddings"]

        return cls(
            vocab_size = len(vocab),
            hidden_size = int(enc_cfg.get("hidden_size", 1024)),
            ff_size = int(enc_cfg.get("ff_size",2048)),
            num_layers = int(enc_cfg.get("num_layers", 4)),
            num_heads = int(enc_cfg.get("num_heads", 8)),
            dropout = float(enc_cfg.get("dropout", 0.1)),
            emb_dropout = float(emb_cfg.get("dropout",0.1)),
            scale = bool(emb_cfg.get("scale", False)),
            pad_idx = PAD_IDX,
        )


    def __repr__(self) -> str:
        return "%s(num_layers=%r, num_heads=%r)" % (
            self.__class__.__name__,
            len(self.layers),
            self.layers[0].attn.num_heads,
        )
