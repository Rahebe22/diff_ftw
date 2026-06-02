import os
from typing import Any

import lightning as L
import segmentation_models_pytorch as smp
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from torchvision.utils import make_grid, save_image

from diffusion.scheduler import (
    compute_alpha_schedule,
    get_timestep_embedding,
    make_beta_schedule,
    min_snr_weight,
    q_sample,
)
from .checkpoints import DIFFUSION_ENCODER_FORMAT


def get_noise_prediction_loss(name: str):
    name = name.lower()
    if name == "mse":
        return F.mse_loss
    if name == "l1":
        return F.l1_loss
    raise ValueError(f"Unknown noise prediction loss '{name}'. Use 'mse' or 'l1'.")


class TimestepFiLM(torch.nn.Module):
    """Apply timestep-dependent scale and shift to one U-Net feature map."""

    def __init__(self, channels: int, embedding_dim: int) -> None:
        super().__init__()
        self.affine = torch.nn.Linear(embedding_dim, channels * 2)
        torch.nn.init.zeros_(self.affine.weight)
        torch.nn.init.zeros_(self.affine.bias)

    def forward(self, feature: Tensor, embedding: Tensor) -> Tensor:
        scale, shift = self.affine(embedding).chunk(2, dim=1)
        shape = (feature.size(0), feature.size(1), 1, 1)
        return feature * (1 + scale.view(shape)) + shift.view(shape)


class FTWEfficientNetDiffusionModel(torch.nn.Module):
    """SMP EfficientNet U-Net with multiscale timestep conditioning."""

    def __init__(
        self,
        in_channels: int,
        backbone: str = "efficientnet-b7",
        weights: bool | str | None = True,
        time_embedding_dim: int = 128,
        time_condition_dim: int = 512,
        model_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.time_embedding_dim = time_embedding_dim
        self._active_time_condition = None
        self._decoder_hook_handles = []

        unet_kwargs = dict(model_kwargs or {})
        decoder_channels = tuple(
            unet_kwargs.get("decoder_channels", (256, 128, 64, 32, 16))
        )
        self.model = smp.Unet(
            encoder_name=backbone,
            encoder_weights="imagenet" if weights is True else None,
            in_channels=in_channels,
            classes=in_channels,
            **unet_kwargs,
        )
        self.time_mlp = torch.nn.Sequential(
            torch.nn.Linear(time_embedding_dim, time_condition_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(time_condition_dim, time_condition_dim),
        )
        self.encoder_time_conditioners = torch.nn.ModuleList(
            [
                TimestepFiLM(channels, time_condition_dim)
                for channels in self.model.encoder.out_channels[1:]
            ]
        )
        self.decoder_time_conditioners = torch.nn.ModuleList(
            [
                TimestepFiLM(channels, time_condition_dim)
                for channels in decoder_channels
            ]
        )
        if len(self.decoder_time_conditioners) != len(self.model.decoder.blocks):
            raise ValueError(
                "decoder_channels must describe every SMP U-Net decoder block"
            )
        self._register_decoder_time_hooks()

    def _register_decoder_time_hooks(self) -> None:
        for handle in self._decoder_hook_handles:
            handle.remove()
        self._decoder_hook_handles = []
        for index, block in enumerate(self.model.decoder.blocks):
            handle = block.register_forward_hook(self._make_decoder_hook(index))
            self._decoder_hook_handles.append(handle)

    def _make_decoder_hook(self, index: int):
        def apply_conditioning(module, inputs, output):
            if self._active_time_condition is None:
                return output
            return self.decoder_time_conditioners[index](
                output, self._active_time_condition
            )

        return apply_conditioning

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        t_emb = get_timestep_embedding(t, self.time_embedding_dim).to(x.device)
        time_condition = self.time_mlp(t_emb)
        features = self.model.encoder(x)
        if len(features) != len(self.encoder_time_conditioners) + 1:
            raise RuntimeError("Unexpected number of EfficientNet feature maps")
        features = [
            features[0],
            *[
                conditioner(feature, time_condition)
                for conditioner, feature in zip(
                    self.encoder_time_conditioners, features[1:]
                )
            ],
        ]
        self._active_time_condition = time_condition
        try:
            decoder_output = self.model.decoder(features)
        finally:
            self._active_time_condition = None
        return self.model.segmentation_head(decoder_output)


class FTWDiffusionSSLTask(L.LightningModule):
    """Diffusion self-supervised pretraining task for FTW imagery."""

    def __init__(
        self,
        in_channels: int = 4,
        lr: float = 2e-4,
        weight_decay: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.999),
        timesteps: int = 1000,
        noise_schedule: str = "cosine",
        beta_start: float = 1e-6,
        beta_end: float = 0.02,
        cosine_s: float = 0.008,
        max_beta: float = 0.999,
        loss: str = "mse",
        min_snr_gamma: float | None = 5.0,
        centered_inputs: bool = True,
        ema_decay: float = 0.9999,
        use_ema_for_validation: bool = True,
        val_timestep_bins: int = 10,
        export_encoder_on_train_end: bool = True,
        model: str = "ftw_efficientnet",
        backbone: str = "efficientnet-b7",
        weights: bool | str | None = True,
        scheduler: str = "cosinewarm",
        t_0: int = 30,
        t_mult: int = 2,
        t_max: int = 100,
        eta_min: float = 1e-5,
        model_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self._printed_train_batch = False
        self._printed_val_batch = False
        self._best_val_sample = None
        self._val_loss_sum = None
        self._val_loss_count = None
        self._val_timestep_loss_sum = None
        self._val_timestep_count = None
        self.criterion = get_noise_prediction_loss(loss)

        if not 0 <= ema_decay < 1:
            raise ValueError("ema_decay must be in [0, 1)")
        if val_timestep_bins < 1:
            raise ValueError("val_timestep_bins must be positive")

        print("[DiffusionTask] Initializing FTW diffusion SSL task")
        print(
            f"[DiffusionTask] in_channels={in_channels}, "
            f"timesteps={timesteps}, noise_schedule={noise_schedule}"
        )
        print(
            f"[DiffusionTask] model={model}, backbone={backbone}, "
            f"weights={weights}"
        )
        print(
            f"[DiffusionTask] loss={loss}, min_snr_gamma={min_snr_gamma}, "
            f"ema_decay={ema_decay}, centered_inputs={centered_inputs}"
        )

        unet_kwargs = dict(model_kwargs or {})
        time_embedding_dim = unet_kwargs.pop("time_embedding_dim", 128)
        time_condition_dim = unet_kwargs.pop("time_condition_dim", 512)
        print(f"[DiffusionTask] model_kwargs={unet_kwargs}")
        if model != "ftw_efficientnet":
            raise ValueError(
                f"Unknown diffusion model '{model}'. Use 'ftw_efficientnet'."
            )
        self.model = FTWEfficientNetDiffusionModel(
            in_channels=in_channels,
            backbone=backbone,
            weights=weights,
            time_embedding_dim=time_embedding_dim,
            time_condition_dim=time_condition_dim,
            model_kwargs=unet_kwargs,
        )
        self.ema_model = FTWEfficientNetDiffusionModel(
            in_channels=in_channels,
            backbone=backbone,
            weights=None,
            time_embedding_dim=time_embedding_dim,
            time_condition_dim=time_condition_dim,
            model_kwargs=unet_kwargs,
        )
        self.ema_model.load_state_dict(self.model.state_dict())
        self.ema_model.requires_grad_(False)
        self.ema_model.eval()
        self.register_buffer("ema_updates", torch.zeros((), dtype=torch.long))

        betas_tensor = make_beta_schedule(
            timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            scheduler_type=noise_schedule,
            cosine_s=cosine_s,
            max_beta=max_beta,
        )
        _, alpha_bars = compute_alpha_schedule(betas_tensor)
        self.register_buffer("alpha_bars", alpha_bars)
        print("[DiffusionTask] Noise schedule ready")

    def forward(self, x: Tensor, t: Tensor, use_ema: bool = False) -> Tensor:
        denoiser = self.ema_model if use_ema else self.model
        return denoiser(x, t)

    def on_fit_start(self) -> None:
        os.makedirs(self.trainer.default_root_dir, exist_ok=True)
        for logger in self.loggers:
            log_dir = getattr(logger, "log_dir", None)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)

    def _prepare_images(self, images: Tensor) -> Tensor:
        if self.hparams.centered_inputs:
            return images.mul(2).sub(1)
        return images

    def _model_to_image(self, images: Tensor) -> Tensor:
        if self.hparams.centered_inputs:
            return images.add(1).div(2).clamp(0, 1)
        return images.clamp(0, 1)

    def _prediction_losses(self, pred_noise: Tensor, noise: Tensor) -> Tensor:
        losses = self.criterion(pred_noise, noise, reduction="none")
        return losses.view(losses.size(0), -1).mean(dim=1)

    def _min_snr_weights(self, t: Tensor) -> Tensor:
        if self.hparams.min_snr_gamma is None:
            return torch.ones_like(t, dtype=self.alpha_bars.dtype)
        return min_snr_weight(t, self.alpha_bars, self.hparams.min_snr_gamma)

    def _shared_step(
        self,
        batch: dict[str, Tensor],
        split: str,
        return_artifacts: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        x0 = self._prepare_images(batch["image"])
        if split == "train" and not self._printed_train_batch:
            print(f"[DiffusionTask] First train batch image shape={tuple(x0.shape)}")
            print(
                "[DiffusionTask] First train batch value range="
                f"({x0.min().item():.4f}, {x0.max().item():.4f})"
            )
            self._printed_train_batch = True
        if split == "val" and not self._printed_val_batch:
            print(
                f"[DiffusionTask] First validation batch image "
                f"shape={tuple(x0.shape)}"
            )
            print(
                "[DiffusionTask] First validation batch value range="
                f"({x0.min().item():.4f}, {x0.max().item():.4f})"
            )
            self._printed_val_batch = True

        batch_size = x0.size(0)
        t = torch.randint(0, self.hparams.timesteps, (batch_size,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = q_sample(x0, t, self.alpha_bars, noise)
        use_ema = split != "train" and self.hparams.use_ema_for_validation
        pred_noise = self(x_t, t, use_ema=use_ema)
        raw_losses = self._prediction_losses(pred_noise, noise)
        min_snr_weights = self._min_snr_weights(t)
        loss = (raw_losses * min_snr_weights).mean()
        self._log_loss(split, loss, min_snr_weights.mean(), batch_size)
        if not return_artifacts:
            return loss

        artifacts = {
            "x0": x0,
            "x_t": x_t,
            "t": t,
            "noise": noise,
            "pred_noise": pred_noise,
            "raw_losses": raw_losses,
        }
        return loss, artifacts

    def _log_loss(
        self,
        split: str,
        loss: Tensor,
        mean_min_snr_weight: Tensor,
        batch_size: int,
    ) -> None:
        if split == "train":
            self.log(
                "train/loss",
                loss,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                sync_dist=False,
                batch_size=batch_size,
            )
            self.log(
                "train/min_snr_weight",
                mean_min_snr_weight,
                on_step=True,
                on_epoch=False,
                sync_dist=False,
                batch_size=batch_size,
            )
        else:
            self.log(
                "val/loss",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
                batch_size=batch_size,
            )
        self.log(
            f"{split}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        loss, artifacts = self._shared_step(batch, "val", return_artifacts=True)
        raw_losses = artifacts["raw_losses"].detach()
        batch_size = raw_losses.size(0)
        self._val_loss_sum += loss.detach() * batch_size
        self._val_loss_count += batch_size
        self._accumulate_timestep_losses(artifacts["t"], raw_losses)
        self._update_best_val_sample(artifacts, batch_idx)
        if (
            batch_idx == 0
            and self.trainer.is_global_zero
            and self.logger
            and hasattr(self.logger, "experiment")
        ):
            self._log_reconstruction_grid(batch)
        return loss

    def on_validation_epoch_start(self) -> None:
        self._best_val_sample = None
        self._val_loss_sum = torch.zeros((), device=self.device)
        self._val_loss_count = torch.zeros((), device=self.device)
        bins = self.hparams.val_timestep_bins
        self._val_timestep_loss_sum = torch.zeros(bins, device=self.device)
        self._val_timestep_count = torch.zeros(bins, device=self.device)

    def _accumulate_timestep_losses(self, t: Tensor, losses: Tensor) -> None:
        bins = self.hparams.val_timestep_bins
        bin_indices = (t * bins // self.hparams.timesteps).clamp(max=bins - 1)
        self._val_timestep_loss_sum.scatter_add_(0, bin_indices, losses)
        self._val_timestep_count.scatter_add_(
            0, bin_indices, torch.ones_like(losses)
        )

    @staticmethod
    def _distributed_sum(value: Tensor) -> Tensor:
        value = value.clone()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
        return value

    def on_validation_epoch_end(self) -> None:
        loss_sum = self._distributed_sum(self._val_loss_sum)
        loss_count = self._distributed_sum(self._val_loss_count)
        timestep_loss_sum = self._distributed_sum(self._val_timestep_loss_sum)
        timestep_count = self._distributed_sum(self._val_timestep_count)
        mean_val_loss = (loss_sum / loss_count.clamp_min(1)).item()

        bins = self.hparams.val_timestep_bins
        bin_width = (self.hparams.timesteps + bins - 1) // bins
        bin_messages = []
        for index in range(bins):
            start = index * bin_width
            end = min(self.hparams.timesteps - 1, (index + 1) * bin_width - 1)
            bin_loss = timestep_loss_sum[index] / timestep_count[index].clamp_min(1)
            self.log(
                f"val/loss_t_{start:04d}_{end:04d}",
                bin_loss,
                on_step=False,
                on_epoch=True,
                sync_dist=False,
            )
            bin_messages.append(f"{start:04d}-{end:04d}={bin_loss.item():.6f}")

        if self.trainer.is_global_zero:
            print(
                f"[DiffusionTask] Mean validation loss for epoch "
                f"{self.current_epoch + 1}: {mean_val_loss:.6f}"
            )
            print("[DiffusionTask] Validation MSE by timestep: " + ", ".join(bin_messages))
        self._save_best_val_sample(mean_val_loss)

    def _update_best_val_sample(
        self,
        artifacts: dict[str, Tensor],
        batch_idx: int,
    ) -> None:
        losses = artifacts["raw_losses"]
        best_idx = torch.argmin(losses)
        best_loss = losses[best_idx].detach()
        if (
            self._best_val_sample is not None
            and best_loss.item() >= self._best_val_sample["loss"]
        ):
            return

        t = artifacts["t"][best_idx]
        alpha_bar = self.alpha_bars[t].view(1, 1, 1, 1)
        x0 = artifacts["x0"][best_idx:best_idx + 1].detach()
        x_t = artifacts["x_t"][best_idx:best_idx + 1].detach()
        pred = artifacts["pred_noise"][best_idx:best_idx + 1].detach()
        true_noise = artifacts["noise"][best_idx:best_idx + 1].detach()
        recon = (x_t - (1 - alpha_bar).sqrt() * pred) / alpha_bar.sqrt()
        self._best_val_sample = {
            "loss": best_loss.item(),
            "batch_idx": int(batch_idx),
            "sample_idx": int(best_idx.item()),
            "t": int(t.item()),
            "x0": x0.cpu(),
            "x_t": x_t.cpu(),
            "true_noise": true_noise.cpu(),
            "pred_noise": pred.cpu(),
            "recon": recon.cpu(),
        }

    def _save_best_val_sample(self, mean_val_loss: float) -> None:
        if not self.trainer.is_global_zero or self._best_val_sample is None:
            return
        sample = self._best_val_sample
        save_dir = os.path.join(self.trainer.default_root_dir, "best_samples")
        os.makedirs(save_dir, exist_ok=True)
        epoch = self.current_epoch + 1
        base_name = (
            f"epoch{epoch}_best_sample{sample['sample_idx']}"
            f"_batch{sample['batch_idx']}_t{sample['t']}"
        )
        save_image(
            self._model_to_image(sample["x0"]),
            os.path.join(save_dir, f"{base_name}_x0.png"),
        )
        save_image(
            self._model_to_image(sample["x_t"]),
            os.path.join(save_dir, f"{base_name}_xt.png"),
        )
        save_image(
            self._noise_to_image(sample["true_noise"]),
            os.path.join(save_dir, f"{base_name}_truenoise.png"),
        )
        save_image(
            self._noise_to_image(sample["pred_noise"]),
            os.path.join(save_dir, f"{base_name}_prednoise.png"),
        )
        save_image(
            self._model_to_image(sample["recon"]),
            os.path.join(save_dir, f"{base_name}_reconstructed.png"),
        )
        print(
            "[DiffusionTask] Saved best validation sample diagnostic "
            f"for epoch {epoch}: timestep={sample['t']}, "
            f"best_sample_loss={sample['loss']:.6f}, "
            f"mean_val_loss={mean_val_loss:.6f}, dir={save_dir}"
        )

    @staticmethod
    def _noise_to_image(noise: Tensor) -> Tensor:
        return ((noise.clamp(-3, 3) + 3) / 6).clamp(0, 1)

    def _log_reconstruction_grid(self, batch: dict[str, Tensor]) -> None:
        x0 = self._prepare_images(batch["image"][:4])
        t = torch.randint(0, self.hparams.timesteps, (x0.size(0),), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = q_sample(x0, t, self.alpha_bars, noise)
        pred_noise = self(
            x_t, t, use_ema=self.hparams.use_ema_for_validation
        )
        alpha_bar = self.alpha_bars[t].view(-1, 1, 1, 1)
        recon = (x_t - (1 - alpha_bar).sqrt() * pred_noise) / alpha_bar.sqrt()
        grid = make_grid(
            torch.cat(
                [
                    self._model_to_image(x0[:, :3]),
                    self._model_to_image(x_t[:, :3]),
                    self._model_to_image(recon[:, :3]),
                ],
                dim=0,
            ),
            nrow=x0.size(0),
        )
        experiment = self.logger.experiment
        if hasattr(experiment, "add_image"):
            experiment.add_image("diffusion/x0_xt_reconstruction", grid, self.global_step)

    @torch.no_grad()
    def on_before_zero_grad(self, optimizer) -> None:
        decay = self.hparams.ema_decay
        for ema_param, param in zip(
            self.ema_model.parameters(), self.model.parameters()
        ):
            ema_param.lerp_(param.detach(), 1 - decay)
        for ema_buffer, buffer in zip(
            self.ema_model.buffers(), self.model.buffers()
        ):
            ema_buffer.copy_(buffer)
        self.ema_updates.add_(1)
        self.ema_model.eval()

    def export_encoder_checkpoint(self, path: str, use_ema: bool = True) -> str:
        """Export only EfficientNet encoder weights for segmentation transfer."""
        source = self.ema_model if use_ema else self.model
        encoder_state_dict = {
            key: value.detach().cpu()
            for key, value in source.model.encoder.state_dict().items()
        }
        payload = {
            "format": DIFFUSION_ENCODER_FORMAT,
            "backbone": self.hparams.backbone,
            "in_channels": self.hparams.in_channels,
            "source": "ema" if use_ema else "online",
            "encoder_state_dict": encoder_state_dict,
        }
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        torch.save(payload, path)
        return path

    def on_train_end(self) -> None:
        if (
            self.hparams.export_encoder_on_train_end
            and self.trainer.is_global_zero
        ):
            path = os.path.join(
                self.trainer.default_root_dir, "encoder_ema.pt"
            )
            self.export_encoder_checkpoint(path, use_ema=True)
            print(f"[DiffusionTask] Exported EMA EfficientNet encoder to {path}")

    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            betas=tuple(self.hparams.betas),
        )
        print(f"[DiffusionTask] Optimizer ready: AdamW lr={self.hparams.lr}")
        if self.hparams.scheduler == "cosinewarm":
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=self.hparams.t_0,
                T_mult=self.hparams.t_mult,
                eta_min=self.hparams.eta_min,
            )
        else:
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.hparams.t_max,
                eta_min=self.hparams.eta_min,
            )
        print(f"[DiffusionTask] LR scheduler ready: {self.hparams.scheduler}")
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
