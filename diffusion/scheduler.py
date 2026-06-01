import torch
import math
import torch.nn.functional as F


def make_beta_schedule(
    timesteps,
    beta_start=1e-6,
    beta_end=0.02,
    scheduler_type="cosine",
    cosine_s=0.008,
    max_beta=0.999,
):
    """
    Generates a beta schedule for diffusion models.

    Supported scheduler types:
    - "linear": Linear interpolation from beta_start to beta_end
    - "cosine": Cosine noise schedule from IDDPM paper

    Args:
        timesteps (int): Number of diffusion steps
        beta_start (float): Min beta value
        beta_end (float): Max beta value for a linear schedule.
        scheduler_type (str): "linear" or "cosine"
        cosine_s (float): Offset used by the IDDPM cosine schedule.
        max_beta (float): Maximum beta used to stabilize a cosine schedule.

    Returns:
        torch.Tensor: Beta schedule of shape (timesteps,)
    """
    if scheduler_type == "linear":
        return torch.linspace(beta_start, beta_end, timesteps)

    elif scheduler_type == "cosine":
        steps = torch.arange(timesteps + 1, dtype=torch.float32)
        alphas_bar = torch.cos(
            (steps / timesteps + cosine_s) / (1 + cosine_s) * math.pi / 2
        ) ** 2
        alphas_bar = alphas_bar / alphas_bar[0]
        betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
        return torch.clip(betas, beta_start, max_beta)

    else:
        raise ValueError(f"Unknown scheduler_type: {scheduler_type}")


def compute_alpha_schedule(betas):
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return alphas, alpha_bars


def q_sample(x0, t, alpha_bars, noise=None):
    if noise is None:
        noise = torch.randn_like(x0)
    sqrt_alpha_bar = alpha_bars[t].sqrt().view(-1, 1, 1, 1)
    sqrt_one_minus = (1 - alpha_bars[t]).sqrt().view(-1, 1, 1, 1)
    return sqrt_alpha_bar * x0 + sqrt_one_minus * noise


def compute_snr(t, alpha_bars):
    alpha_bar = alpha_bars[t]
    return alpha_bar / (1 - alpha_bar).clamp_min(1e-12)


def min_snr_weight(t, alpha_bars, gamma=5.0):
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    snr = compute_snr(t, alpha_bars)
    safe_snr = snr.clamp_min(1e-12)
    return safe_snr.clamp(max=gamma) / safe_snr


def get_timestep_embedding(timesteps, dim):
    device = timesteps.device
    half_dim = dim // 2
    emb = torch.exp(-math.log(10000) * torch.arange(half_dim, device=device).float() / half_dim)
    emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb
