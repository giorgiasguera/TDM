#NUOVA VERSIONE (video solo su wandb)
import os
import pickle
import shutil
import queue
import time
import numpy as np
import torch
import pandas as pd

from torch import Tensor

from builders import build_gradient_clipper, build_optimizer, build_scheduler
from helpers import (load_config, set_seed, load_checkpoint, log_cfg,
                     make_model_dir, make_logger, ConfigurationError,
                     get_latest_checkpoint, symlink_update)
from data import build_dataloaders
from model import build_model, Model
from vocabulary import Vocabulary
from loss import TDMLoss, TARGET_PAD
from prediction import validate_on_data
from plot_videos import plot_video, alter_DTW_timing
#from torch.utils.tensorboard import SummaryWriter
import wandb

#per le metriche
from metrics import mpjpe, mpjae, fid


class TrainManager:
    """
    Manages training loop, validation, checkpointing.
    """

    def __init__(self, model: Model, config: dict, src_vocab: Vocabulary, test: bool = False):

        train_config = config["training"]
        model_dir = train_config["model_dir"]

        # If model continue, continues from the latest checkpoint
        model_continue = train_config.get("continue", True)

        # If the directory has not been created, can't continue from anything
        if not os.path.isdir(model_dir):
            model_continue = False
        if test:
            model_continue = True

        # Files for logging and storing
        self.model_dir = make_model_dir(train_config["model_dir"],
                                        overwrite=train_config.get("overwrite", False),
                                        model_continue=model_continue)

        # build logger
        self.logger = make_logger(model_dir=self.model_dir)
        self.logging_freq = train_config.get("logging_freq", 100)
        # build validation files
        self.valid_report_file = "{}/validations.txt".format(self.model_dir)
        #self.tb_writer = SummaryWriter(log_dir=self.model_dir+"/tensorboard/")

        # model
        self.model = model
        self.src_vocab = src_vocab
        self.pad_index = self.model.pad_index
        self.bos_index = self.model.bos_index
        self._log_parameters_list()
        self.target_pad = TARGET_PAD

        # New loss - depending on config
        self.loss = TDMLoss(cfg=config, target_pad=self.target_pad)

        # normalization
        self.normalization = "batch"

        # Optimization
        self.learning_rate_min = train_config.get("learning_rate_min", 1.0e-8)
        self.clip_grad_fun = build_gradient_clipper(config=train_config)
        self.optimizer = build_optimizer(config=train_config, parameters=model.parameters())

        # Validation & early stopping 
        self.validation_freq = train_config.get("validation_freq", 1000)
        self.ckpt_best_queue = queue.Queue(maxsize=train_config.get("keep_last_ckpts", 1))
        self.ckpt_queue = queue.Queue(maxsize=1)

        # TODO - Include Back Translation (!!!)
        self.eval_metric = train_config.get("eval_metric", "dtw").lower()
        if self.eval_metric not in ["dtw", "loss"]: #!!!
            raise ConfigurationError("Invalid setting for 'eval_metric', valid options: 'dtw', 'loss'") #!!!!!

        self.early_stopping_metric = train_config.get("early_stopping_metric", "dtw")
        
        # if we schedule after BLEU/chrf, we want to maximize it, else minimize
        # early_stopping_metric decides on how to find the early stopping point:
        # ckpts are written when there's a new high/low score for this metric
        if self.early_stopping_metric in ["loss", "dtw"]:
            self.minimize_metric = True
        else:
            raise ConfigurationError("Invalid setting for 'early_stopping_metric', "
                                    "valid options: 'loss', 'dtw'.")

        # learning rate scheduling
        self.scheduler, self.scheduler_step_at = build_scheduler(
            config=train_config,
            scheduler_mode="min" if self.minimize_metric else "max",
            optimizer=self.optimizer,
            hidden_size=config["model"]["encoder"]["hidden_size"])

        # Data & batch handling 
        self.shuffle = train_config.get("shuffle", True)
        self.epochs = train_config["epochs"]
        self.batch_size = train_config["batch_size"]
        self.batch_multiplier = train_config.get("batch_multiplier", 1)
        
        # generation
        self.max_output_length = train_config.get("max_output_length", None)

        # CPU / GPU 
        self.use_cuda = train_config["use_cuda"]
        if self.use_cuda:
            self.model.cuda()
            self.loss.cuda()

        # Tinitialize training statistics 
        self.steps = 0
        # stop training if this flag is True by reaching learning rate minimum
        self.stop = False
        self.total_tokens = 0
        self.best_ckpt_iteration = 0
        # initial values for best scores
        self.best_ckpt_score = np.inf if self.minimize_metric else -np.inf
        # comparison function for scores (decide se un nuovo score è migliore del best_ckpt_score)
        self.is_best = lambda score: score < self.best_ckpt_score \
            if self.minimize_metric else score > self.best_ckpt_score

        # Checkpoint restart 
        # If continuing
        if model_continue:
            post_fix = "_best" if test else "_every"
            ckpt = get_latest_checkpoint(model_dir, post_fix=post_fix)
            if ckpt is None:
                self.logger.info("Can't find checkpoint in directory %s", model_dir)
            else:
                self.logger.info("Continuing model from %s", ckpt)
                self.init_from_checkpoint(ckpt)

        self.skip_frames = config["data"].get("skip_frames", 1)

        # -----
        # Inizializza wandb passandogli l'intero file YAML come configurazione
        if not test:
            wandb.login(key="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") #per non fare sempre login
            wandb.init(
                project="TDM", # Il nome della cartella principale sul sito
                name="train_i_5",  # Il nome specifico di questo run
                config=config # Salva tutti i tuoi iperparametri
            )
        # -----

    # nuova
    def _prepare_batch(self, batch: dict) -> dict:
        """
        Converte il batch fornito dal dataLoader nel formato atteso da
        model.get_loss_for_batch(), aggiungendo src_tokens, src_mask
        e trg_mask al dizionario.
        """
        device = next(self.model.parameters()).device

        # src_tokens (B, L) e src_mask (B, 1, L)
        src_tokens, src_mask = self.src_vocab.batch_encode(batch["texts"], device=device)

        # trg_mask (B, 1, N): True dove il frame è reale, False dove è padding
        # Si costruisce confrontando l'indice di colonna con la lunghezza reale
        skeletons= batch["skeletons"].to(device)
        skel_lens= batch["skel_lens"]
        skel_lens= batch["skel_lens"].to(device) #HO AGGIUNTO QUESTO
        B, N, _ = skeletons.shape
        idx= torch.arange(N, device=device).unsqueeze(0)  # (1, N)
        trg_mask= (idx < skel_lens.unsqueeze(1)).unsqueeze(1)  # (B, 1, N)

        batch["src_tokens"] = src_tokens
        batch["src_mask"] = src_mask
        batch["skeletons"] = skeletons
        batch["trg_mask"] = trg_mask

        return batch


    def _save_checkpoint(self, type: str = "every") -> None:
        """
        Save a checkpoint.
        """
        # Define model path
        model_path = "{}/{}_{}.ckpt".format(self.model_dir, self.steps, type)
        # Define State
        state = {
            "steps": self.steps,
            "total_tokens": self.total_tokens,
            "best_ckpt_score": self.best_ckpt_score,
            "best_ckpt_iteration":self.best_ckpt_iteration,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict()
                                  if self.scheduler is not None else None,
        }
        torch.save(state, model_path)

        # If this is the best checkpoint
        if type == "best":
            if self.ckpt_best_queue.full():
                to_delete = self.ckpt_best_queue.get() # delete oldest ckpt
                try:
                    os.remove(to_delete)
                except FileNotFoundError:
                    self.logger.warning("Wanted to delete old checkpoint %s but "
                                        "file does not exist.", to_delete)
            self.ckpt_best_queue.put(model_path)
            
            best_path = "{}/best.ckpt".format(self.model_dir)
            try:
                # create/modify symbolic link for best checkpoint
                symlink_update("{}_best.ckpt".format(self.steps), best_path)
            except OSError:
                # overwrite best.ckpt
                torch.save(state, best_path)

        # If this is just the checkpoint at every validation
        elif type == "every":
            if self.ckpt_queue.full():
                to_delete = self.ckpt_queue.get() # delete oldest ckpt
                try:
                    os.remove(to_delete)
                except FileNotFoundError:
                    self.logger.warning("Wanted to delete old checkpoint %s but "
                                        "file does not exist.", to_delete)
            self.ckpt_queue.put(model_path)
            
            every_path = "{}/every.ckpt".format(self.model_dir)
            try:
                # create/modify symbolic link for best checkpoint
                symlink_update("{}_every.ckpt".format(self.steps), every_path)
            except OSError:
                # overwrite every.ckpt
                torch.save(state, every_path)

    # Initialise from a checkpoint
    def init_from_checkpoint(self, path: str) -> None:
        """
        Initialize model and optimizer from a checkpoint.
        """
        # Find last checkpoint
        model_checkpoint = load_checkpoint(path=path, use_cuda=self.use_cuda)

        # restore model and optimizer parameters
        self.model.load_state_dict(model_checkpoint["model_state"])
        self.optimizer.load_state_dict(model_checkpoint["optimizer_state"])

        if model_checkpoint["scheduler_state"] is not None and \
                self.scheduler is not None:
            # Load the scheduler state
            self.scheduler.load_state_dict(model_checkpoint["scheduler_state"])

        # restore counts
        self.steps = model_checkpoint["steps"]
        self.total_tokens = model_checkpoint["total_tokens"]
        self.best_ckpt_score = model_checkpoint["best_ckpt_score"]
        self.best_ckpt_iteration = model_checkpoint["best_ckpt_iteration"]

        # move parameters to cuda
        if self.use_cuda:
            self.model.cuda()


    # Train and validate function
    def train_and_validate(self, train_loader, val_loader) -> None:
        # no iteratore torchtext
        val_step = 0

        # Loop through epochs
        for epoch_no in range(self.epochs):
            self.logger.info("EPOCH %d", epoch_no + 1)

            if self.scheduler is not None and self.scheduler_step_at == "epoch":
                self.scheduler.step(epoch=epoch_no)

            self.model.train()

            # Reset statistics for each epoch.
            start = time.time()
            total_valid_duration = 0
            start_tokens = self.total_tokens
            count = self.batch_multiplier - 1
            epoch_loss = 0

            for batch in train_loader:
                # reactivate training
                self.model.train()

                # [3] crea il batch usando _prepare_batch invece di Batch()
                batch = self._prepare_batch(batch)

                update = count == 0
                batch_loss = self._train_batch(batch, update=update)

                #self.tb_writer.add_scalar("train/train_batch_loss", batch_loss,self.steps)
                # wandb
                wandb.log({"train/batch_loss": batch_loss}, step=self.steps)

                count = self.batch_multiplier if update else count
                count -= 1
                epoch_loss += batch_loss.detach().cpu().numpy()

                if self.scheduler is not None and self.scheduler_step_at == "step" and update:
                    self.scheduler.step()

                # log learning progress
                if self.steps % self.logging_freq == 0 and update:
                    elapsed = time.time() - start - total_valid_duration
                    elapsed_tokens = self.total_tokens - start_tokens
                    self.logger.info(
                        "Epoch %3d Step: %8d Batch Loss: %12.6f "
                        "Tokens per Sec: %8.0f, Lr: %.6f",
                        epoch_no + 1, self.steps, batch_loss,
                        elapsed_tokens / elapsed,
                        self.optimizer.param_groups[0]["lr"])
                    start = time.time()
                    total_valid_duration = 0
                    start_tokens = self.total_tokens

                # validate on the entire dev set
                if self.steps % self.validation_freq == 0 and update:

                    valid_start_time = time.time()

                    valid_score, valid_loss, valid_references, valid_hypotheses, \
                        valid_inputs, all_dtw_scores, valid_file_paths = \
                        validate_on_data(
                            model=self.model,
                            data_loader=val_loader,
                            src_vocab=self.src_vocab,
                            loss_function=self.loss,
                            eval_metric=self.eval_metric,
                            type="val",
                        )

                    val_step += 1 

                    # Tensorboard writer
                    #self.tb_writer.add_scalar("valid/valid_loss", valid_loss, self.steps)
                    #self.tb_writer.add_scalar("valid/valid_score", valid_score, self.steps)
                    # wandb
                    wandb.log({
                        "valid/valid_loss": valid_loss,
                        "valid/valid_score_dtw": valid_score,
                        "learning_rate": self.optimizer.param_groups[0]["lr"]
                     }, step=self.steps)

                    if self.early_stopping_metric == "loss":
                        ckpt_score = valid_loss
                    else:  # dtw
                        ckpt_score = valid_score #qua vedi sign-idd

                    new_best  = False
                    self.best = False
                    if self.is_best(ckpt_score):
                        self.best = True
                        self.best_ckpt_score = ckpt_score
                        self.best_ckpt_iteration = self.steps
                        self.logger.info(
                            'Hooray! New best validation result [%s]!',
                            self.early_stopping_metric)
                        if self.ckpt_queue.maxsize > 0:
                            self.logger.info("Saving new checkpoint.")
                            new_best = True
                            self._save_checkpoint(type="best")

                        # Display these sequences, in this index order
                        display = list(range(0, len(valid_hypotheses), int(np.ceil(len(valid_hypotheses) / 13.15))))
                        self.produce_validation_video(
                            output_joints=valid_hypotheses,
                            inputs=valid_inputs,
                            references=valid_references,
                            model_dir=self.model_dir,
                            steps=self.steps,
                            display=display,
                            type="val_inf",
                            file_paths=valid_file_paths,
                        )
                    
                    self._save_checkpoint(type="every")

                    if self.scheduler is not None and self.scheduler_step_at == "validation":
                        self.scheduler.step(ckpt_score)

                    # append to validation report
                    self._add_report(
                        valid_score=valid_score, valid_loss=valid_loss,
                        eval_metric=self.eval_metric,
                        new_best=new_best, report_type="val")

                    valid_duration = time.time() - valid_start_time
                    total_valid_duration += valid_duration
                    self.logger.info(
                        'Validation result at epoch %3d, step %8d: '
                        'Val DTW Score: %6.2f, loss: %8.4f, duration: %.4fs',
                        epoch_no + 1, self.steps, valid_score,
                        valid_loss, valid_duration)

                if self.stop:
                    break

            if self.stop:
                self.logger.info(
                    'Training ended since minimum lr %f was reached.',
                    self.learning_rate_min)
                break

            self.logger.info('Epoch %3d: total training loss %.5f', epoch_no + 1, epoch_loss)
        else:
            self.logger.info('Training ended after %3d epochs.', epoch_no + 1)

        self.logger.info('Best validation result at step %8d: %6.2f %s.',
                         self.best_ckpt_iteration, self.best_ckpt_score,
                         self.early_stopping_metric)

        #self.tb_writer.close()  # close Tensorboard writer

        #wandb
        wandb.finish()
    

    # Produce the video of Phoenix MTC joints
    def produce_validation_video(self, output_joints, inputs, references, display, model_dir, type, steps="", file_paths=None, dtw_file=None):

        # If not at test
        if type != "test":
            dir_name = model_dir + "/videos/Step_{}/".format(steps)
            if not os.path.exists(model_dir + "/videos/"):
                os.mkdir(model_dir + "/videos/")

        # If at test time
        elif type == "test":
            dir_name = model_dir + "/test_videos/"

        # Create model video folder if not exist
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

        # For sequence to display
        for i in display:

            seq = output_joints[i].detach().cpu().numpy() if hasattr(output_joints[i], 'detach') else output_joints[i] #passo a alter_dtw_timing già sequenze numpy
            ref_seq = references[i].detach().cpu().numpy() if hasattr(references[i], 'detach') else references[i]

            #fix 
            # Taglia i frame di padding dalla GT (righe tutte zero)
            real_len = int((ref_seq.any(axis=1)).sum())   # conta i frame non-zero
            ref_seq = ref_seq[:real_len]
            seq = seq[:real_len]

            input = inputs[i]
            # Write gloss label
            gloss_label = input[0]
            if input[1] != "<eos>":
                gloss_label += "_" + input[1]
            if input[2] != "<eos>":
                gloss_label += "_" + input[2]

            # Alter the dtw timing of the produced sequence, and collect the DTW score
            timing_hyp_seq, ref_seq_count, dtw_score = alter_DTW_timing(seq, ref_seq)

            video_ext = "{}_{}.mp4".format(gloss_label, "{0:.2f}".format(float(dtw_score)).replace(".", "_"))

            if file_paths is not None:
                sequence_ID = file_paths[i]
            else:
                sequence_ID = None

            print(sequence_ID + '    dtw: ' + '{0:.2f}'.format(float(dtw_score)))

            if dtw_file is not None:
                dtw_file.writelines(sequence_ID + ' ' + '{0:.2f}'.format(float(dtw_score)) + '\n')

            # Plot this sequences video
            # if "<" not in video_ext:
            plot_video(joints=timing_hyp_seq,
                       file_path=dir_name,
                       video_name=video_ext,
                       references=ref_seq_count,
                       skip_frames=self.skip_frames,
                       sequence_ID=sequence_ID)
            
            # Ricostruiamo il percorso esatto in cui plot_video ha appena salvato l'MP4
            video_path = "{}/{}.mp4".format(dir_name, sequence_ID.split(".")[0])

            if os.path.exists(video_path):
                # carica video su wandb associandolo allo step corrente
                if wandb.run is not None:
                    log_kwargs = {"commit": False}
                    if type != "test":
                        log_kwargs["step"] = self.steps

                    wandb.log({
                        f"Video_Validazione/{gloss_label}": wandb.Video(
                            video_path, 
                            #fps=25 // self.skip_frames, 
                            format="mp4",
                            caption=f"Gloss: {gloss_label} | DTW: {dtw_score:.2f}"
                        )
                    }, **log_kwargs)
                
                os.remove(video_path) #rimuove il video dopo averlo caricato su wandb


    # Save the skeletons of Phoenix
    def save_skels(self, output_joints, display, model_dir, type, file_paths=None):
        # ipdb.set_trace()

        picklefile = open(model_dir + "/phoenix14t.skels.%s" % type, "wb")

        csvIn = pd.read_csv(model_dir + "/csv/%s_phoenix2014t.csv" % type, sep='|',encoding='utf-8')
        pickle_list = []

        for i in display:
            name = file_paths[i]
            video = name[len(os.path.dirname(name))+1:]
            signer = csvIn[csvIn['id']==video]['signer'].item()
            gloss = csvIn[csvIn['id']==video]['annotation'].item()
            text = csvIn[csvIn['id']==video]['translation'].item()
            seq = output_joints[i].detach().cpu().numpy()
            sign = torch.tensor(seq, dtype = torch.float32)

            dict_num = {'name': name, 'signer': signer, 'gloss': gloss, 'text': text, 'sign': sign}

            pickle_list.append(dict_num)

        pickle.dump(pickle_list, picklefile)
        print("The skeletons of %s date have been save." % type)


    # Train the batch and return the loss
    def _train_batch(self, batch: dict, update: bool = True) -> Tensor:
        """
        Train the model on one batch and return the loss.
        """
        # Get loss from this batch — is_train rimosso (sempre True qui)
        batch_loss = self.model.get_loss_for_batch(
            batch=batch, loss_function=self.loss)

        # normalize batch loss [4]
        if self.normalization == "batch":
            normalizer = len(batch["texts"]) 
        elif self.normalization == "tokens":
            normalizer = int(batch["skel_lens"].sum())
        else:
            raise NotImplementedError("Only normalize by 'batch' or 'tokens'")

        norm_batch_loss     = batch_loss / normalizer
        # division needed since loss.backward sums the gradients until updated
        norm_batch_multiply = norm_batch_loss / self.batch_multiplier

        # compute gradients
        norm_batch_multiply.backward()

        if self.clip_grad_fun is not None:
            # clip gradients (in-place)
            self.clip_grad_fun(params=self.model.parameters())

        if update:
            # make gradient step
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            # increment step counter
            self.steps += 1

        # increment token counter
        self.total_tokens += int(batch["skel_lens"].sum())

        return norm_batch_loss


    #controlla se lr ha raggiunto il minimo
    def _add_report(self, valid_score: float, valid_loss: float, eval_metric: str, new_best: bool = False, report_type: str = "val") -> None:
        """
        Append validation results to the report file and check early stopping.
        """
        current_lr = -1
        # ignores other param groups for now
        for param_group in self.optimizer.param_groups:
            current_lr = param_group['lr']

        if current_lr < self.learning_rate_min:
            self.stop = True

        if report_type == "val":
            with open(self.valid_report_file, 'a') as opened_file:
                opened_file.write(
                    "Steps: {} Loss: {:.5f}| DTW: {:.3f}|"
                    " LR: {:.6f} {}\n".format(
                        self.steps, valid_loss, valid_score,
                        current_lr, "*" if new_best else ""))


    def _log_parameters_list(self) -> None:
        """
        Write all model parameters (name, shape) to the log.
        """
        model_parameters = filter(lambda p: p.requires_grad,
                                  self.model.parameters())
        n_params = sum([np.prod(p.size()) for p in model_parameters])
        self.logger.info("Total params: %d", n_params)
        trainable_params = [n for (n, p) in self.model.named_parameters()
                            if p.requires_grad]
        self.logger.info("Trainable parameters: %s", sorted(trainable_params))
        assert trainable_params


def train(cfg_file: str, ckpt: str = None) -> None:
    """
    Entry point per il training.
    """
    # Load the config file
    cfg = load_config(cfg_file)

    # Set the random seed
    set_seed(seed=cfg["training"].get("random_seed", 42))

    # HO CAMBIATO QUA 
    loaders = build_dataloaders(yaml_path=cfg_file, num_workers=2) #num_workers=2 invece che 4 è per colab
    train_loader = loaders["train"]
    dev_loader = loaders["dev"]

    # Vocabolario (costruito dentro build_dataloaders, ricostruiamo)
    src_vocab = Vocabulary.load_or_build(
        vocab_file=cfg["data"]["src_vocab"],
        text_file=cfg["data"]["train"] + ".text")

    model = build_model(cfg=cfg, src_vocab=src_vocab)

    if ckpt is not None:
        use_cuda = cfg["training"].get("use_cuda", True)
        model_ckpt = load_checkpoint(ckpt, use_cuda=use_cuda)
        # Build model and load parameters from the checkpoint
        model.load_state_dict(model_ckpt["model_state"])

    # for training management, e.g. early stopping and model selection
    trainer = TrainManager(config=cfg, model=model, src_vocab=src_vocab, test=False)

    # Store copy of original training config in model dir
    shutil.copy2(cfg_file, trainer.model_dir + "/TDM.yaml")
    # Log all entries of config
    log_cfg(cfg, trainer.logger)

    # Train the model
    trainer.train_and_validate(train_loader=train_loader, val_loader=dev_loader)


def test(cfg_file: str, ckpt: str = None) -> None:
    """
    Entry point per la valutazione su test/dev set.
    """
    # Load the config file
    cfg = load_config(cfg_file)

    # Load the model directory and checkpoint
    model_dir = cfg["training"]["model_dir"]

    # when checkpoint is not specified, take latest (best) from model dir
    if ckpt is None:
        ckpt = get_latest_checkpoint(model_dir, post_fix="_best")
        if ckpt is None:
            raise FileNotFoundError("No checkpoint found in directory {}.".format(model_dir))

    use_cuda = cfg["training"].get("use_cuda", True)
    eval_metric = cfg["training"]["eval_metric"] #serve la chiamata a validate_on_Data
    max_output_length = cfg["training"].get("max_output_length", None)

    # load the data
    # HO CAMBIATO QUA
    loaders = build_dataloaders(yaml_path=cfg_file, num_workers=0)
    dev_loader  = loaders["dev"]
    test_loader = loaders["test"]
    
    src_vocab = Vocabulary.load_or_build(
        vocab_file=cfg["data"]["src_vocab"],
        text_file=cfg["data"]["train"] + ".text")

    # Load model state from disk
    model_ckpt = load_checkpoint(ckpt, use_cuda=use_cuda)
    # Build model and load parameters into it
    model = build_model(cfg=cfg, src_vocab=src_vocab)
    model.load_state_dict(model_ckpt["model_state"])

    # If cuda, set model as cuda
    if use_cuda:
        model.cuda()

    # Set up trainer to produce videos
    # TrainManager in test mode (per _validate e logging)
    trainer = TrainManager(model=model, config=cfg, src_vocab=src_vocab, test=True)

    wandb.login(key="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")
    wandb.init(project="TDM", name="inference_i_5", config=cfg)

    # For each of the required data, produce results
    for data_set_name, data_loader in [("dev", dev_loader), ("test", test_loader)]:
        current_valid_score, valid_loss, valid_references, valid_hypotheses, \
            valid_inputs, all_dtw_scores, valid_file_paths = \
            validate_on_data(
                model = model,
                data_loader = data_loader,
                src_vocab = src_vocab,
                loss_function = trainer.loss,
                eval_metric = eval_metric,
                type = data_set_name,
            )
        
        #metriche
        try:
            # MPJPE e MPJAE prendono direttamente le coordinate 3D degli scheletri
            mpjpe_score = mpjpe(references=valid_references, hypotheses=valid_hypotheses)
            mpjae_score = mpjae(references=valid_references, hypotheses=valid_hypotheses)
            # FID
            fid_score = fid(references=valid_references, hypotheses=valid_hypotheses)

            # Stampa i risultati nel terminale
            print(f"[{data_set_name.upper()}] MPJPE: {mpjpe_score:.2f} | MPJAE: {mpjae_score:.2f} | FID: {fid_score:.2f}")

            # Salva i risultati nel file di log del trainer
            if hasattr(trainer, 'logger') and trainer.logger:
                trainer.logger.info(f"{data_set_name} - MPJPE: {mpjpe_score:.4f} | MPJAE: {mpjae_score:.4f} | FID: {fid_score:.4f}")

            # Manda i risultati alla dashboard di WandB per i grafici
            if wandb.run is not None:
                wandb.log({
                    f"{data_set_name}/MPJPE": mpjpe_score,
                    f"{data_set_name}/MPJAE": mpjae_score,
                    f"{data_set_name}/FID": fid_score
                })
        except Exception as e:
            print(f"Errore durante il calcolo delle metriche su {data_set_name}: {e}")

        if not os.path.exists(os.path.join(model_dir, 'test_videos')):
            os.mkdir(os.path.join(model_dir, 'test_videos'))

        dtw_file = open(os.path.join(model_dir, 'test_videos', data_set_name+'_dtw.txt'),'w')
        dtw_file.writelines('DTW Score of %s set: %.3f\n' %(data_set_name, current_valid_score))

        print('DTW Score of %s set: %.3f' %(data_set_name, current_valid_score))
        
        trainer.logger.info('%4s DTW: %.4f\t loss: %.4f', data_set_name, current_valid_score, valid_loss)
        # Set which sequences to produce video for
        display = list(range(len(valid_hypotheses)))

        #trainer.save_skels(
            #output_joints = valid_hypotheses,
            #display = display,
            #model_dir = model_dir,
            #type = data_set_name,
            #file_paths = valid_file_paths,
        #)

        # Produce videos for the produced hypotheses
        trainer.produce_validation_video(
            output_joints = valid_hypotheses,
            inputs = valid_inputs,
            references = valid_references,
            model_dir = model_dir,
            steps = trainer.steps,
            display = display,
            type = "test",
            file_paths = valid_file_paths,
            dtw_file=dtw_file,
        )

    wandb.finish()
