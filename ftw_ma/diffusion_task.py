from typing import Any

import lightning as L
import os
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from torchvision.utils import make_grid, save_image

from diffusion.scheduler import (
    compute_alpha_schedule,
    get_timestep_embedding,
    make_beta_schedule,
    q_sample,
)


def get_noise_prediction_loss(name: str):
    name = name.lower()
    if name == "mse":
        return F.mse_loss
    if name == "l1":
        return F.l1_loss
    raise ValueError(f"Unknown noise prediction loss '{name}'. Use 'mse' or 'l1'.")


class FTWEfficientNetDiffusionModel(torch.nn.Module):
    """FTW EfficientNet model adapted for diffusion noise prediction."""

    def __init__(
        self,
        in_channels: int,
        backbone: str = "efficientnet-b7",
        weights: bool | str | None = True,
        time_embedding_dim: int = 128,
        model_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.time_embedding_dim = time_embedding_dim
        self.time_mlp = torch.nn.Sequential(
            torch.nn.Linear(time_embedding_dim, in_channels),
            torch.nn.SiLU(),
            torch.nn.Linear(in_channels, in_channels),
        )
        self.model = smp.Unet(
            encoder_name=backbone,
            encoder_weights="imagenet" if weights is True else None,
            in_channels=in_channels,
            classes=in_channels,
            **(model_kwargs or {}),
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        t_emb = get_timestep_embedding(t, self.time_embedding_dim).to(x.device)
        t_bias = self.time_mlp(t_emb).view(x.size(0), x.size(1), 1, 1)
        return self.model(x + t_bias)


class FTWDiffusionSSLTask(L.LightningModule):
    """Diffusion self-supervised pretraining task for FTW imagery."""

    def __init__(
        self,
        in_channels: int = 4,
        lr: float = 2e-4,
        weight_decay: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.999),
        timesteps: int = 1000,
        noise_schedule: str = "linear",
        beta_start: float = 1e-6,
        beta_end: float = 0.02,
        loss: str = "mse",
        loss_weights: dict[str, float] | None = None,
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
        self._val_loss_sum = 0.0
        self._val_loss_count = 0

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
            f"[DiffusionTask] loss={loss}, lr={lr}, "
            f"weight_decay={weight_decay}, scheduler={scheduler}"
        )

        model_kwargs = model_kwargs or {}
        print(f"[DiffusionTask] model_kwargs={model_kwargs}")
        if model == "ftw_efficientnet":
            self.model = FTWEfficientNetDiffusionModel(
                in_channels=in_channels,
                backbone=backbone,
                weights=weights,
                time_embedding_dim=model_kwargs.pop("time_embedding_dim", 128),
                model_kwargs=model_kwargs,
            )
        else:
            raise ValueError(
                f"Unknown diffusion model '{model}'. Use 'ftw_efficientnet'."
            )

        self.criterion = get_noise_prediction_loss(loss)

        betas_tensor = make_beta_schedule(
            timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            scheduler_type=noise_schedule,
        )
        _, alpha_bars = compute_alpha_schedule(betas_tensor)
        self.register_buffer("alpha_bars", alpha_bars)
        print("[DiffusionTask] Noise schedule ready")

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        return self.model(x, t)

    def on_fit_start(self) -> None:
        os.makedirs(self.trainer.default_root_dir, exist_ok=True)
        for logger in self.loggers:
            log_dir = getattr(logger, "log_dir", None)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)

    def _shared_step(
        self,
        batch: dict[str, Tensor],
        split: str,
        return_artifacts: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        x0 = batch["image"]
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
        pred_noise = self(x_t, t)
        loss = self.criterion(pred_noise, noise)
        self.log(
            f"{split}/loss",
            loss,
            on_step=split == "train",
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            f"{split}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        if not return_artifacts:
            return loss

        artifacts = {
            "x0": x0,
            "x_t": x_t,
            "t": t,
            "noise": noise,
            "pred_noise": pred_noise,
        }
        return loss, artifacts

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        loss, artifacts = self._shared_step(batch, "val", return_artifacts=True)
        self._val_loss_sum += float(loss.detach().cpu())
        self._val_loss_count += 1
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
        self._val_loss_sum = 0.0
        self._val_loss_count = 0

    def _update_best_val_sample(
        self,
        artifacts: dict[str, Tensor],
        batch_idx: int,
    ) -> None:
        pred_noise = artifacts["pred_noise"]
        noise = artifacts["noise"]
        batch_size = pred_noise.size(0)
        losses = F.mse_loss(pred_noise, noise, reduction="none").view(
            batch_size, -1
        ).mean(dim=1)
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
        pred = pred_noise[best_idx:best_idx + 1].detach()
        true_noise = noise[best_idx:best_idx + 1].detach()
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

    def on_validation_epoch_end(self) -> None:
        mean_val_loss = (
            self._val_loss_sum / self._val_loss_count
            if self._val_loss_count
            else float("nan")
        )
        if self.trainer.is_global_zero:
            print(
                f"[DiffusionTask] Mean validation loss for epoch "
                f"{self.current_epoch + 1}: {mean_val_loss:.6f}"
            )

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
            sample["x0"].clamp(0, 1),
            os.path.join(save_dir, f"{base_name}_x0.png"),
        )
        save_image(
            sample["x_t"].clamp(0, 1),
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
            sample["recon"].clamp(0, 1),
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
        x0 = batch["image"][:4]
        t = torch.randint(0, self.hparams.timesteps, (x0.size(0),), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = q_sample(x0, t, self.alpha_bars, noise)
        pred_noise = self(x_t, t)
        alpha_bar = self.alpha_bars[t].view(-1, 1, 1, 1)
        recon = (x_t - (1 - alpha_bar).sqrt() * pred_noise) / alpha_bar.sqrt()
        grid = make_grid(
            torch.cat([x0[:, :3], x_t[:, :3], recon[:, :3]], dim=0).clamp(0, 1),
            nrow=x0.size(0),
        )
        experiment = self.logger.experiment
        if hasattr(experiment, "add_image"):
            experiment.add_image("diffusion/x0_xt_reconstruction", grid, self.global_step)

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
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
