# OPTUNA
# modificato: build_dataloaders
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
import yaml

# Load configuration from TDM.yaml
def load_config(yaml_path: str) -> dict:
    """ 
    :param yaml_path: path to TDM.yaml
    :return configuration dictionary

    """
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


class PHOENIXDataset(Dataset):
    """
    Load PHOENIX14T dataset for the Text-to-Pose (TDM) task.

    Expected structure on disk for each split (e.g., train):
        <split_dir>/train.text — a sentence per line (free text)
        <split_dir>/train.skels — a skeleton sequence per line (flat values)
        <split_dir>/train.files — a video ID per line

    Operative parameters are read directly from TDM.yaml:
        data.src → "text"  
        data.trg  → "skels" 
        data.skip_frames → temporal downsampling 
        data.max_sent_length → longer samples are discarded 
        model.trg_size → vector dimension of each frame (e.g., 150 = 50 joints × 3D)
    """

    def __init__(self, split_path: str, cfg: dict):
        """
        : param split_path: ex. "./Data/phoenix14t/train"
        : param cfg: dictionary from TDM.yaml
        """
        data_cfg = cfg["data"]
        model_cfg = cfg["model"]

        #var locali
        self.skip_frames = int(data_cfg.get("skip_frames", 1))
        self.max_len = int(data_cfg.get("max_sent_length", 300))
        self.frame_size = int(model_cfg.get("trg_size", 150)) # 50 joint × 3D
        self.num_joints = self.frame_size // 3 # 50

        base = Path(split_path)
        split_name = base.name # "train"|"dev"|"test"
        split_dir = base.parent # directory contenente i file

        # testo libero (source)
        text_file = split_dir / f"{split_name}.text"
        self.texts = text_file.read_text(encoding="utf-8").strip().split("\n") #legge il file, rimuove spazi iniziali/finali (strip), splitta per linee

        # file IDs
        files_file = split_dir / f"{split_name}.files"
        self.file_ids = files_file.read_text(encoding="utf-8").strip().split("\n")

        # skeleton (target)
        skels_file = split_dir / f"{split_name}.skels"
        raw_skels = skels_file.read_text(encoding="utf-8").strip().split("\n") 

        #per ogni riga del file .skels, crea un tensor (T, frame_size) e lo aggiunge alla lista skeletons_raw. Se skip_frames > 1, prende solo ogni skip_frames-esimo frame (downsampling temporale).
        skeletons_raw = []
        for line in raw_skels:
            vals = list(map(float, line.strip().split())) #crea una lista di token float
            actual_frame_size = 151 #fix
            T = len(vals) // actual_frame_size #calcola il numero di frame (T) nella lista
            seq = torch.tensor(
                vals[: T * actual_frame_size], dtype=torch.float32
            ).view(T, actual_frame_size)
            seq = seq[:, :self.frame_size]
            
            if self.skip_frames > 1:
                seq = seq[:: self.skip_frames] #downsampling temporale: prendi solo ogni skip_frames-esimo frame
            skeletons_raw.append(seq)
            

        # check lunghezze (stesso numero di righe nei tre file)
        assert len(self.texts) == len(self.file_ids) == len(skeletons_raw), (
            f"Numero di righe disallineato in {split_name}: "
            f"text={len(self.texts)}, files={len(self.file_ids)}, "
            f"skels={len(skeletons_raw)}"
        )

        # filtraggio per max_sent_length (scarta i sample dove la sequenza skeleton supera max_len frame oppure il testo e' una stringa vuota (riga vuota nel .text)
        # (Il testo libero è generalmente molto più corto dello skeleton)
        kept_indices = [
            i for i, (t, s) in enumerate(zip(self.texts, skeletons_raw))
            if s.shape[0] <= self.max_len and t.strip() != ""
        ]
        discarded = len(skeletons_raw) - len(kept_indices)
        if discarded:
            print(
                f"[PHOENIXDataset/{split_name}] Scartati {discarded} sample "
                f"con T > {self.max_len} frame."
            )

        # costruisce una lista di indici da tenere (troppo padding diventa pesante)
        self.texts = [self.texts[i] for i in kept_indices]
        self.file_ids = [self.file_ids[i] for i in kept_indices]
        self.skeletons = [skeletons_raw[i] for i in kept_indices]

        print(
            f"[PHOENIXDataset/{split_name}] Caricati {len(self.texts)} sample. "
            f"frame_size={self.frame_size}, skip_frames={self.skip_frames}."
        )

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        """
        Restituisce un singolo sample. Lo skeleton è un tensor (T, 150).
        """
        return {
            "text": self.texts[idx], # str — frase in linguaggio naturale
            "file_id": self.file_ids[idx], # str — ID video
            "skeleton": self.skeletons[idx], # Tensor (T, 150)
        }

# padding per batch di sequenze a lunghezza variabile
def collate_fn(batch: list) -> dict:
    """
    Aggrega una lista di sample in un mini-batch.
    Le sequenze skeleton hanno lunghezze diverse: vengono paddare con 0
    fino alla lunghezza massima nel batch.

    Returns:
        texts : list[str] — frasi (il tokenizer le gestirà)
        file_ids : list[str] — ID video
        skeletons : Tensor (B, T_max, 150) — skeleton paddati
        skel_lens : Tensor (B,) — lunghezze originali (per masked loss)
    """
    
    texts = [s["text"] for s in batch]
    file_ids = [s["file_id"] for s in batch]
    skels = [s["skeleton"] for s in batch] 

    
    skel_lens = torch.tensor([s.shape[0] for s in skels], dtype=torch.long) #lunghezze originali di ogni sequenza skeleton usate per la masked loss durante l'addestramento
    skels_pad = pad_sequence(skels, batch_first=True, padding_value=0.0) #padda le sequenze skeleton con 0 fino alla lunghezza massima T_max del batch, restituendo un tensor (B, T_max, 150)
    # (B, T_max, 150)

    return {
        "texts": texts, #list[str]
        "file_ids": file_ids,
        "skeletons": skels_pad, # (B, T_max, 150)
        "skel_lens": skel_lens, # (B,)
    }

# crea i tre DataLoader da config
def build_dataloaders(
    yaml_path: str,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> dict:
    """
    Legge TDM.yaml e costruisce i DataLoader per train, dev e test.

    Returns:
        dict con chiavi "train", "dev", "test", ognuno un DataLoader.
    """
    cfg=load_config(yaml_path)
    data_cfg=cfg["data"]
    train_cfg=cfg["training"]

    batch_size=int(train_cfg.get("batch_size", 16))
    shuffle=bool(train_cfg.get("shuffle", True))
    use_cuda=bool(train_cfg.get("use_cuda", True))

    splits = {
        "train": data_cfg["train"],
        "dev":   data_cfg["dev"],
        "test":  data_cfg["test"],
    }

    loaders = {}
    for split_name, split_path in splits.items():
        dataset = PHOENIXDataset(split_path=split_path, cfg=cfg)
        loaders[split_name] = DataLoader(
            dataset,
            batch_size = batch_size,
            shuffle = (shuffle and split_name == "train"),
            collate_fn = collate_fn,
            num_workers = num_workers, # se num_workers > 0, i DataLoader useranno processi separati per caricare i dati in parallelo (consigliato se il dataset è grande e il caricamento è un collo di bottiglia)
            pin_memory = (pin_memory and use_cuda), 
            drop_last = (split_name == "train"),  # drop_last=True scarta l'ultimo batch se è più piccolo di batch_size 
        )

    return loaders