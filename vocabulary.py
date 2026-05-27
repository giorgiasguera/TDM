import torch
from torch import Tensor
from pathlib import Path
from collections import Counter
from typing import List, Optional

# Costanti token speciali
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]

PAD_IDX = 0
UNK_IDX = 1
BOS_IDX = 2
EOS_IDX = 3


class Vocabulary:
    """
    Vocabolario bidirezionale token <-> indice intero.
    """

    def __init__(self):
        self.token2idx: dict = {}
        self.idx2token: list = []

    # Costruzione 
    @classmethod
    def build(
        cls,
        text_file: str,
        vocab_file: Optional[str] = None,
        min_freq: int = 1,
    ) -> "Vocabulary":

        vocab = cls()

        lines = Path(text_file).read_text(encoding="utf-8").strip().split("\n")
        counter = Counter()
        for line in lines:
            counter.update(cls._tokenize(line))

        # Token speciali sempre in testa, poi le parole per frequenza decrescente
        all_tokens = SPECIAL_TOKENS + [
            tok for tok, freq in counter.most_common()
            if freq >= min_freq
        ]

        vocab.token2idx = {tok: i for i, tok in enumerate(all_tokens)}
        vocab.idx2token = all_tokens

        if vocab_file:
            Path(vocab_file).parent.mkdir(parents=True, exist_ok=True)
            Path(vocab_file).write_text(
                "\n".join(all_tokens), encoding="utf-8"
            )
            print(f"[Vocabulary] Salvato: {vocab_file}  ({len(vocab)} tipi)")

        print(
            f"[Vocabulary] Costruito da {text_file}: "
            f"{len(vocab)} tipi (min_freq={min_freq})"
        )
        return vocab

    @classmethod
    def load(cls, vocab_file: str) -> "Vocabulary":
        """
        Carica un vocabolario precedentemente salvato su disco.
        """
        vocab =cls()
        tokens = Path(vocab_file).read_text(encoding="utf-8").strip().split("\n")
        vocab.token2idx = {tok: i for i, tok in enumerate(tokens)}
        vocab.idx2token = tokens
        print(f"[Vocabulary] Caricato: {vocab_file}  ({len(vocab)} tipi)")
        return vocab

    @classmethod
    def load_or_build(
        cls,
        vocab_file: str,
        text_file: str,
        min_freq: int = 1,
    ) -> "Vocabulary":
        """
        Carica il vocabolario se esiste su disco, altrimenti lo costruisce
        e lo salva.
        """
        if Path(vocab_file).exists():
            return cls.load(vocab_file)
        else:
            return cls.build(text_file, vocab_file=vocab_file, min_freq=min_freq)

    
    # Tokenizzazione
    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """
        Tokenizzazione a livello di parola: lowercase + split su spazi.
        """
        return text.lower().strip().split()


    # Encode (da frase a lista di ID)
    def encode(self, text: str) -> List[int]:
        """
        Converte una frase in lista di interi.
        Aggiunge <bos> all'inizio e <eos> alla fine.
        Le parole fuori vocabolario diventano <unk>.
        """
        tokens = self._tokenize(text)
        return (
            [BOS_IDX]
            + [self.token2idx.get(tok, UNK_IDX) for tok in tokens] # .get() restituisce l'indice del token, o UNK_IDX se non trovato
            + [EOS_IDX]
        )

    # Decode
    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """
        Converte una lista di interi in frase leggibile.
        """
        special = {PAD_IDX, BOS_IDX, EOS_IDX} if skip_special else set()
        return " ".join(
            self.idx2token[i]
            for i in ids
            if i not in special and 0 <= i < len(self.idx2token)
        )

    # Encode del batch con padding e maschere (bridge tra modello e dataloader)
    def batch_encode(
        self,
        texts: List[str],
        device: torch.device = torch.device("cpu"),
    ) -> tuple[Tensor, Tensor]:
        """
        Converte una lista di frasi in due tensor pronti per il TextEncoder.
        """
        encoded = [self.encode(t) for t in texts]
        lengths = [len(seq) for seq in encoded]
        max_len = max(lengths) #determio la frase più lunga per paddare le altre

        # Padding a destra con PAD_IDX delle sequenze piu' corte fino a max_len
        padded = [
            seq + [PAD_IDX] * (max_len - len(seq))
            for seq in encoded
        ]

        src_tokens = torch.tensor(padded, dtype=torch.long,   device=device) # (B, L) 
        src_mask   = (src_tokens != PAD_IDX).unsqueeze(1) # (B, 1, L) formato atteso da MultiHeadAttention: 1=token reale, 0=padding

        return src_tokens, src_mask


    def __len__(self) -> int:
        return len(self.idx2token)

    def __contains__(self, token: str) -> bool:
        return token in self.token2idx

    def __repr__(self) -> str:
        return f"Vocabulary(size={len(self)}, special={SPECIAL_TOKENS})"
