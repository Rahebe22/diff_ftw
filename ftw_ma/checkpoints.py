from pathlib import Path

import torch


DIFFUSION_ENCODER_FORMAT = "ftw_diffusion_encoder_v1"


def load_diffusion_encoder_checkpoint(encoder, path: str | Path) -> dict:
    """Load an exported diffusion EfficientNet encoder into a target encoder."""
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("Diffusion encoder checkpoint must contain a dictionary")
    if checkpoint.get("format") != DIFFUSION_ENCODER_FORMAT:
        raise ValueError(
            f"Expected checkpoint format '{DIFFUSION_ENCODER_FORMAT}', "
            f"got '{checkpoint.get('format')}'"
        )
    state_dict = checkpoint.get("encoder_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("Diffusion encoder checkpoint is missing encoder_state_dict")
    encoder.load_state_dict(state_dict, strict=True)
    return checkpoint
