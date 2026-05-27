import math
import torch
import torch.nn.functional as F

from collections import namedtuple
from torch import nn

__all__ = ["GaussianDiffusion"]

# Utility functions
ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


# Cosine noise schedule (genera il parametro β_t che serve ad aggiunge una quantità di rumore controllata nel forward process)
def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64) # [0, 1, 2, ..., T]
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2 
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

# forward q_sample e reverse ddim_sample
class GaussianDiffusion(nn.Module):

    # istanzia tutti i coefficienti
    def __init__(
        self,
        denoiser,
        timesteps: int = 1000,
        sampling_timesteps: int= 5,
        scale: float = 1.0, #qua
    ):
        super().__init__()

        self.denoiser = denoiser

        self.num_timesteps= int(timesteps)
        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= self.num_timesteps

        self.is_ddim_sampling = self.sampling_timesteps < self.num_timesteps 
        self.ddim_sampling_eta = 1.   # eta 1 campionamento stocastico

        self.scale = scale

        # calcolo dei coefficienti del schedule, scritto tutto nella sezione forward process del paper di signidd
        betas = cosine_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0) #ā_t = ∏ᵢ₌₁ᵗ (1-βᵢ)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)  # ā_{t-1}

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev',alphas_cumprod_prev)

        self.register_buffer('sqrt_alphas_cumprod',
                             torch.sqrt(alphas_cumprod)) #percentuale di posa pulita che rimane
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             torch.sqrt(1. - alphas_cumprod)) #percentuale di rumore aggiunto 
        self.register_buffer('log_one_minus_alphas_cumprod',
                             torch.log(1. - alphas_cumprod))

        # coefficienti usati in ddim_sample
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             torch.sqrt(1. / alphas_cumprod - 1))

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod) # β̃t
        self.register_buffer('posterior_variance', posterior_variance)
            # Below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
                             torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
                             betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

    # register_buffer:tutti i coefficienti sono registrati come buffer PyTorch (non parametri, non aggiornati da ottimizzatore, ma salvati con il modello e spostati su device insieme ai parametri).
   
    # forward process
    def q_sample(self, x_start, t, noise=None):

        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod,
                                             t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod,
                                                   t, x_start.shape)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    # Predizione rumore da xt e x0 (usata in ddim_sample per ricavare il rumore predetto a partire dalla posa predetta))
    def predict_noise_from_start(self, x_t, t, x0):

        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    # predizionedel rumore da predic_noise_from_start e della posa pulita dal denoiser
    def model_predictions(self, x, encoder_output, t, src_mask, trg_mask):

        pred_x_start = self.denoiser(
            encoder_output = encoder_output,
            trg_embed = x,
            src_mask = src_mask,
            trg_mask = trg_mask,
            t = t,
        )
        pred_noise = self.predict_noise_from_start(x, t, pred_x_start)
        return ModelPrediction(pred_noise, pred_x_start)


    # Reverse process (inferenza)
    def ddim_sample(self, encoder_output, trg_len, src_mask, trg_mask=None):

        batch = encoder_output.shape[0]
        device = encoder_output.device
        shape = (batch, trg_len, 150)

        total_timesteps = self.num_timesteps
        sampling_timesteps = self.sampling_timesteps
        eta = self.ddim_sampling_eta #eta è coefficiente di stocasticità del campionamento ddim: eta=0 deterministico, eta=1 stocastico 

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        
        img = torch.randn(shape, device=device) #img corrisponde a p_T, il punto di partenza del processo di campionamento DDIM (p_T ~ N(0, I))

        preds_all = []
        x_start = None

        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)

            preds = self.model_predictions(x=img, # chiama model_prediction e questa chiama il denoiser
                                                 encoder_output=encoder_output,
                                                 t=time_cond,
                                                 src_mask=src_mask,
                                                 trg_mask=trg_mask) # predice la posa pulita e il rumore
            pred_noise = preds.pred_noise.float()
            x_start= preds.pred_x_start
            preds_all.append(x_start)

            if time_next < 0:   # ultimo step: restituisce x_start direttamente
                img = x_start
                continue

            # Coefficienti DDIM 
            alpha      = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) *
                           (1 - alpha_next) / (1 - alpha)).sqrt() # quanta stocasticità iniettare 
            c = (1 - alpha_next - sigma ** 2).sqrt() # coefficiente del rumore predetto

            noise = torch.randn_like(img)

            img = (x_start * alpha_next.sqrt() +
                   c * pred_noise +
                   sigma * noise)

        return preds_all

    # fornisco le pose originali e ottengo pose rumorose, timetep e rumore
    def prepare_targets(self, targets):

        device = targets.device
        B = targets.shape[0]

        # Campiona un timestep t per ogni sample del batch
        t = torch.randint(0, self.num_timesteps, (B,), device=device, dtype=torch.long)

        # il campionamento casuale del timestep serve al denoiser per imparare implicitamente a stimare p0 
        # a qualsiasi livello di corruzione

        noise   = torch.randn_like(targets) # genera rumore puro
        x_start = targets * self.scale
        x_poses = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_poses = x_poses / self.scale

        return x_poses, noise, t


    # Forward (training + inference)
    def forward(self, encoder_output, input_3d, src_mask, trg_mask, is_train):

        if not is_train:
            results = self.ddim_sample(
                encoder_output = encoder_output,
                trg_len = input_3d.shape[1],
                src_mask = src_mask,
                trg_mask = trg_mask,
            )
            # Restituisce la predizione dell'ultimo step DDIM
            return results[self.sampling_timesteps - 1]

        # training 
        x_poses, noises, t = self.prepare_targets(input_3d)
        x_poses = x_poses.float()

        t = t.squeeze(-1) if t.dim() > 1 else t

        pred_pose = self.denoiser(
            encoder_output = encoder_output,
            trg_embed = x_poses,
            src_mask = src_mask,
            trg_mask = trg_mask,
            t = t,
        )
        return pred_pose