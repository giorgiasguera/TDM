import torch
import numpy as np

from model import Model
from dtw import calculate_dtw
from vocabulary import Vocabulary

# Validate the model on a DataLoader split and return scores and sequences.
def validate_on_data(model: Model, data_loader,
                     src_vocab: Vocabulary,
                     loss_function: torch.nn.Module = None,
                     max_output_length: int = None,
                     eval_metric: str = "dtw",
                     type: str = "val",
                     BT_model=None):

    device = next(model.parameters()).device
    model.eval()

    valid_hypotheses = [] # lista che accumula le pose generate dal modello
    valid_references = [] # lista che accumula le pose corrispondenti
    valid_inputs = [] # lista di frasi sorgenti decodificate dal vocabolario (usate per il log). 
    file_paths = [] # lista degli ID video per associare ogni sequenza generata al video originale
    all_dtw_scores = [] 

    valid_loss = 0
    total_ntokens = 0 # contatori del numero totale di frame reali
    total_nseqs = 0 #contatori delle sequenze processate

    with torch.no_grad():
        for batch in data_loader:

            # preparazione batch (logica di _prepare_batch) in trainmanager
            src_tokens, src_mask = src_vocab.batch_encode(
                batch["texts"], device=device)

            skeletons = batch["skeletons"].to(device)
            skel_lens = batch["skel_lens"]
            skel_lens = batch["skel_lens"].to(device) #io
            B, N, _ = skeletons.shape
            idx = torch.arange(N, device=device).unsqueeze(0)
            trg_mask = (idx < skel_lens.unsqueeze(1)).unsqueeze(1)

            targets = skeletons #pose GT

            if loss_function is not None:
                batch_aug = {
                    "src_tokens": src_tokens,
                    "src_mask": src_mask,
                    "skeletons": skeletons,
                    "trg_mask": trg_mask,
                }
                batch_loss = model.get_loss_for_batch( #esegue il forward diffusion su GT per calcolare la loss=ljoint+lbone
                    batch=batch_aug, loss_function=loss_function)
                valid_loss += batch_loss
                total_ntokens += int(skel_lens.sum())
                total_nseqs += B

            # Inference DDIM (is_train=False)
            output = model.forward(
                src = src_tokens,
                trg_input= skeletons, # usato solo per shape, la lunghezza della sequenza da generare
                src_mask = src_mask,
                trg_mask = trg_mask,
                is_train = False,
            )

            # raccolta risultati
            valid_references.extend(targets) # targets è skeletons, le pose GT
            valid_hypotheses.extend(output)
            file_paths.extend(batch["file_ids"])

            # Converte gli ID del testo in parole
            for i in range(B):
                tokens = [src_vocab.idx2token[src_tokens[i][j].item()]
                          for j in range(src_tokens.shape[1])]
                valid_inputs.append(tokens)

            # calcolo DTW per questo batch
            # Converti in numpy e togli il padding tagliando con skel_lens lunghezza originale
            refs_np = [targets[i, :int(skel_lens[i]), :].cpu().numpy()
                       for i in range(B)]
            hyps_np = [output[i, :int(skel_lens[i]), :].cpu().numpy()
                       for i in range(B)]
            dtw_score = calculate_dtw(refs_np, hyps_np)
            all_dtw_scores.extend(dtw_score)

    # DTW media 
    current_valid_score = np.mean(all_dtw_scores)

    return (current_valid_score, valid_loss, valid_references,
            valid_hypotheses, valid_inputs, all_dtw_scores, file_paths)
