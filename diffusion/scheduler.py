import torch
import math
import torch.nn.functional as F


def make_beta_schedule(timesteps, beta_start=1e-6, beta_end=0.02, scheduler_type="cosine"):
    """
    Generates a beta schedule for diffusion models.

    Supported scheduler types:
    - "linear": Linear interpolation from beta_start to beta_end
    - "cosine": Cosine noise schedule from IDDPM paper

    Args:
        timesteps (int): Number of diffusion steps
        beta_start (float): Min beta value
        beta_end (float): Max beta value
        scheduler_type (str): "linear" or "cosine"

    Returns:
        torch.Tensor: Beta schedule of shape (timesteps,)
    """
    if scheduler_type == "linear":
        return torch.linspace(beta_start, beta_end, timesteps)

    elif scheduler_type == "cosine":
        steps = torch.arange(timesteps + 1, dtype=torch.float32)
        alphas_bar = torch.cos((steps / timesteps + 0.001) / 1.001 * math.pi / 2) ** 2
        alphas_bar = alphas_bar / alphas_bar[0]
        betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
        return torch.clip(betas, beta_start, beta_end)

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


def get_timestep_embedding(timesteps, dim):
    device = timesteps.device
    half_dim = dim // 2
    emb = torch.exp(-math.log(10000) * torch.arange(half_dim, device=device).float() / half_dim)
    emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb
