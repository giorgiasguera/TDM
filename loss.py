import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple

from helpers import getSkeletalModelStructure

# usato per il padding delle sequenze di pose target.
TARGET_PAD = 0.0

# gli do un batch di pose (B, N, 150) e mi restituisce tensore lunghezza (B, N, 50) e direzione normalizzata (B, N, 150) di ogni osso
def get_length_direct(trg: Tensor) -> Tuple[Tensor, Tensor]:
    # (B, N, 150) -> (B, N, 50, 3): riorganizza il tensore in una struttura dove ogni joint è un vettore 3D
    trg_reshaped = trg.view(trg.shape[0], trg.shape[1], 50, 3)

    # Lista di 50 tensori (B, N, 3): uno per ogni joint
    trg_list = trg_reshaped.split(1, dim=2)
    # trg_list_squeeze[j] = coordinate 3D del joint j per tutti i batch e frame
    trg_list_squeeze = [t.squeeze(dim=2) for t in trg_list]

    skeletons = getSkeletalModelStructure() # 50 coppie (parent_idx, child_idx)

    length = []
    direct = []

    for parent_idx, child_idx in skeletons:
        # vettore da child a parent
        diff = trg_list_squeeze[parent_idx] - trg_list_squeeze[child_idx]

        # Lunghezza dell'osso con norma L2
        result_length = torch.norm(diff, p=2, dim=2, keepdim=True)

        # direzione normalizzata
        result_direct = diff / (result_length + torch.finfo(result_length.dtype).tiny)

        length.append(result_length)
        direct.append(result_direct)

    lengths = torch.stack(length, dim=-1).squeeze()
    directs = torch.stack(direct, dim=2).view(trg.shape[0], trg.shape[1], -1)

    return lengths, directs


class TDMLoss(nn.Module):
    """
    L = Ljoint + lambda * Lbone
    """

    def __init__(self, cfg: dict, target_pad: float = TARGET_PAD):
        super(TDMLoss, self).__init__() 

        self.loss = cfg["training"]["loss"].lower()
        self.bone_loss = cfg["training"]["bone_loss"].lower()

        if self.loss == "l1":
            self.criterion = nn.L1Loss() #gemini dice di mettere reduction='none' e modificare anche il forward
        elif self.loss == "mse":
            self.criterion = nn.MSELoss()
        else:
            print("Loss not found - revert to default L1 loss")
            self.criterion = nn.L1Loss()

        if self.bone_loss == "l1":
            self.criterion_bone = nn.L1Loss()
        elif self.bone_loss == "mse":
            self.criterion_bone = nn.MSELoss()
        else:
            print("Loss not found - revert to default MSE loss")
            self.criterion_bone = nn.MSELoss()

        model_cfg = cfg["model"]

        self.target_pad = target_pad
        self.loss_scale = model_cfg.get("loss_scale", 1.0)
        self.lambda_bone = model_cfg["diffusion"].get("lambda_bone", 0.1)

    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        """
        L = Ljoint + lambda * Lbone su un batch
        """
        # padding mask(B, N, 150): true dove i dati sono reali, false dove sono padding
        loss_mask = (targets != self.target_pad)

        # azzera predizioni e target sui frame di padding
        preds_masked = preds * loss_mask
        targets_masked = targets * loss_mask

        # lunghezze e direzione ossa
        preds_masked_length, preds_masked_direct = get_length_direct(preds_masked)
        targets_masked_length,targets_masked_direct = get_length_direct(targets_masked)

        # maschera sulle lunghezze e direzioni (le prime 50 colonne per le lunghezze etutte le 150 colonne per le direzioni)
        preds_masked_length = preds_masked_length * loss_mask[:, :, :50]
        targets_masked_length = targets_masked_length * loss_mask[:, :, :50]
        preds_masked_direct = preds_masked_direct * loss_mask[:, :, :150]
        targets_masked_direct = targets_masked_direct * loss_mask[:, :, :150]

        # eq loss combinata
        loss = (self.criterion(preds_masked, targets_masked) + 
               self.lambda_bone * self.criterion_bone(preds_masked_direct,
                                                      targets_masked_direct))

        if self.loss_scale != 1.0:
            loss = loss * self.loss_scale

        return loss
