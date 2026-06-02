import torch

from diffusion.scheduler import (
    compute_alpha_schedule,
    compute_snr,
    make_beta_schedule,
    min_snr_weight,
)
from ftw_ma.checkpoints import (
    DIFFUSION_ENCODER_FORMAT,
    load_diffusion_encoder_checkpoint,
)


def test_cosine_schedule_reaches_nearly_pure_noise():
    betas = make_beta_schedule(1000, scheduler_type="cosine")
    _, alpha_bars = compute_alpha_schedule(betas)

    assert betas.shape == (1000,)
    assert torch.all(betas > 0)
    assert torch.all(betas <= 0.999)
    assert torch.all(alpha_bars[1:] < alpha_bars[:-1])
    assert alpha_bars[-1] < 1e-6


def test_compute_snr_decreases_with_timestep():
    betas = make_beta_schedule(1000, scheduler_type="cosine")
    _, alpha_bars = compute_alpha_schedule(betas)
    t = torch.tensor([0, 100, 500, 999])
    snr = compute_snr(t, alpha_bars)

    assert torch.all(snr[1:] < snr[:-1])


def test_min_snr_weight_downweights_only_high_snr_examples():
    betas = make_beta_schedule(1000, scheduler_type="cosine")
    _, alpha_bars = compute_alpha_schedule(betas)
    t = torch.tensor([0, 999])
    weights = min_snr_weight(t, alpha_bars, gamma=5.0)

    assert weights[0] < 1
    assert torch.isclose(weights[1], torch.tensor(1.0))


def test_min_snr_weight_handles_zero_snr():
    alpha_bars = torch.tensor([0.0])
    weights = min_snr_weight(torch.tensor([0]), alpha_bars, gamma=5.0)

    assert torch.equal(weights, torch.ones(1))


def test_diffusion_encoder_checkpoint_loads_strictly(tmp_path):
    source = torch.nn.Sequential(
        torch.nn.Conv2d(4, 8, kernel_size=3, padding=1),
        torch.nn.BatchNorm2d(8),
    )
    target = torch.nn.Sequential(
        torch.nn.Conv2d(4, 8, kernel_size=3, padding=1),
        torch.nn.BatchNorm2d(8),
    )
    path = tmp_path / "encoder_ema.pt"
    torch.save(
        {
            "format": DIFFUSION_ENCODER_FORMAT,
            "backbone": "efficientnet-b7",
            "in_channels": 4,
            "source": "ema",
            "encoder_state_dict": source.state_dict(),
        },
        path,
    )

    checkpoint = load_diffusion_encoder_checkpoint(target, path)

    assert checkpoint["source"] == "ema"
    for source_value, target_value in zip(
        source.state_dict().values(), target.state_dict().values()
    ):
        assert torch.equal(source_value, target_value)


def test_timestep_conditioning_keeps_smp_encoder_unchanged():
    smp = __import__("segmentation_models_pytorch")
    from ftw_ma.diffusion_task import FTWEfficientNetDiffusionModel

    baseline = smp.Unet(
        encoder_name="efficientnet-b0",
        encoder_weights=None,
        in_channels=4,
        classes=4,
    )
    conditioned = FTWEfficientNetDiffusionModel(
        in_channels=4,
        backbone="efficientnet-b0",
        weights=None,
        time_embedding_dim=16,
        time_condition_dim=32,
    )

    baseline_state = baseline.encoder.state_dict()
    conditioned_state = conditioned.model.encoder.state_dict()
    assert baseline_state.keys() == conditioned_state.keys()
    for key in baseline_state:
        assert baseline_state[key].shape == conditioned_state[key].shape


def test_unused_efficientnet_classification_tail_is_frozen():
    from ftw_ma.diffusion_task import FTWEfficientNetDiffusionModel

    model = FTWEfficientNetDiffusionModel(
        in_channels=4,
        backbone="efficientnet-b0",
        weights=None,
        time_embedding_dim=16,
        time_condition_dim=32,
    )

    for name in ("_conv_head", "_bn1"):
        module = getattr(model.model.encoder, name)
        assert all(not parameter.requires_grad for parameter in module.parameters())


def test_timestep_conditioned_unet_forward_pass():
    from ftw_ma.diffusion_task import FTWEfficientNetDiffusionModel

    model = FTWEfficientNetDiffusionModel(
        in_channels=4,
        backbone="efficientnet-b0",
        weights=None,
        time_embedding_dim=16,
        time_condition_dim=32,
    )
    model.eval()

    with torch.inference_mode():
        output = model(
            torch.randn(2, 4, 32, 32),
            torch.tensor([0, 999]),
        )

    assert output.shape == (2, 4, 32, 32)


def test_timestep_conditioned_unet_has_no_unused_trainable_parameters():
    from ftw_ma.diffusion_task import FTWEfficientNetDiffusionModel

    model = FTWEfficientNetDiffusionModel(
        in_channels=4,
        backbone="efficientnet-b0",
        weights=None,
        time_embedding_dim=16,
        time_condition_dim=32,
    )
    output = model(
        torch.randn(2, 4, 32, 32),
        torch.tensor([0, 999]),
    )

    output.mean().backward()

    unused = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    assert unused == []
