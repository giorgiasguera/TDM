import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads: int, size: int, dropout: float = 0.1):
        super().__init__()
        # l'assert evita errori di reshape in forward
        assert size % num_heads == 0, (
            f"hidden_size ({size}) deve essere divisibile per num_heads ({num_heads})"
        )
        self.head_dim  = size // num_heads # 1024/8 = 128
        self.num_heads = num_heads
        self.size = size
        self.scale = math.sqrt(self.head_dim)

        # proiezioni lineari per query, key, value e output
        self.q_proj = nn.Linear(size, size)
        self.k_proj = nn.Linear(size, size)
        self.v_proj = nn.Linear(size, size)
        self.out_proj = nn.Linear(size, size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, q: Tensor, k: Tensor, v: Tensor,
                mask: Tensor = None) -> Tensor:

        B, L_q, _ = q.shape

        # proiezione e split per teste: (B, L, size) -> (B, num_heads, L, head_dim) (reshape)
        Q = self.q_proj(q).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(k).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(v).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # maschera di padding
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # softmax + dropout sui pesi
        attn = self.dropout(F.softmax(scores, dim=-1))

        # combinazione pesata dei value
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, L_q, self.size)
        return self.out_proj(out)

# serve ad introdurre non linearità tra due strati di attenzione, altrimenti più strati di MHA sarebbero equivalenti a uno solo.
class PositionwiseFeedForward(nn.Module):

    def __init__(self, size: int, ff_size: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(size, ff_size)
        self.fc2 = nn.Linear(ff_size, size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.dropout(F.relu(self.fc1(x))))

# analizza una sequenza di input per estrarre informazioni e relazioni tra gli elementi.
class TransformerEncoderLayer(nn.Module):

    def __init__(self, size: int, ff_size: int,
                 num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = MultiHeadAttention(num_heads, size, dropout)
        self.ffn = PositionwiseFeedForward(size, ff_size, dropout)
        self.norm1 = nn.LayerNorm(size, eps=1e-6)
        self.norm2 = nn.LayerNorm(size, eps=1e-6)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        # pre-norm: LayerNorm prima dell'attenzione
        residual = x
        x_norm = self.norm1(residual)
        # residual connection e dropout: somma il risultato dell'attenzione all'input originale (connessione residua) e applica il dropout per evitare l'overfitting.
        x = self.dropout(self.attn(x_norm, x_norm, x_norm, mask)) + residual

        residual = x
        x = self.dropout(self.ffn(self.norm2(residual))) + residual

        return x

# serve al denoiser per guidare la generazione delle pose tramite condizione g
class TransformerDecoderLayer(nn.Module):

    def __init__(self, size: int, ff_size: int,
                 num_heads: int, dropout: float = 0.1):
        super().__init__()

        # self-attention sulle pose (trg-trg)
        self.self_attn  = MultiHeadAttention(num_heads, size, dropout)
        # cross-attention con condizione g (trg-src)
        self.cross_attn = MultiHeadAttention(num_heads, size, dropout)
        # Feed-Forward
        self.ffn = PositionwiseFeedForward(size, ff_size, dropout)

        self.norm1 = nn.LayerNorm(size, eps=1e-6)   # prima del self-attn
        self.norm2 = nn.LayerNorm(size, eps=1e-6)   # prima del cross-attn
        self.norm3 = nn.LayerNorm(size, eps=1e-6)   # prima della FFN

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor, memory: Tensor,
                src_mask: Tensor = None,
                trg_mask: Tensor = None) -> Tensor:
        
        # self-attention sulle pose.Le pose si "guardano" tra loro. Usa trg_mask per escludere i frame di padding.
        residual = x
        x = self.dropout(
            self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x),
                           mask=trg_mask)
        ) + residual

        #cross-attention con condizione g. Le pose "interrogano" la condizione g. Usa src_mask per escludere i token <pad> del testo.
        # Q = pose (cosa cerchiamo)
        # K = V = memory = g (dove cerchiamo: testo + timestep)
        residual = x
        x = self.dropout(
            self.cross_attn(self.norm2(x), memory, memory,
                            mask=src_mask)
        ) + residual

        # Feed-Forward.
        # trasformazione non lineare per-posizione, identica a quella nell'encoder.
        residual = x
        x = self.dropout(
            self.ffn(self.norm3(x))
        ) + residual

        return x


class PositionalEncoding(nn.Module):
    """
    Positional Encoding sinusoidale. Serve a fornire informazioni sulla posizione dei token in una sequenza.
    Non apprendibile — registrato come buffer, spostato su GPU con .to(device).
    """
    def __init__(self, size: int, max_len: int = 512, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, size)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, size, 2).float() * (-math.log(10000.0) / size)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, L, size) -> (B, L, size) con PE sinusoidale sommato"""
        return self.dropout(x + self.pe[:, : x.size(1)])
