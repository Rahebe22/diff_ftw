"""Lightweight per-rank progress diagnostics for long DDP runs."""

from __future__ import annotations

import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lightning.pytorch.callbacks import Callback


class DDPProgressDiagnostics(Callback):
    """Write each DDP rank's current training phase to a separate log file."""

    def __init__(
        self,
        output_dir: str,
        log_every_n_steps: int = 500,
        heartbeat_seconds: float = 60,
    ) -> None:
        if log_every_n_steps < 1:
            raise ValueError("log_every_n_steps must be positive")
        if heartbeat_seconds <= 0:
            raise ValueError("heartbeat_seconds must be positive")
        self.output_dir = Path(output_dir)
        self.log_every_n_steps = log_every_n_steps
        self.heartbeat_seconds = heartbeat_seconds
        self._lock = threading.Lock()
        self._state: dict[str, Any] = {
            "phase": "not_started",
            "epoch": -1,
            "global_step": 0,
            "batch_idx": -1,
        }
        self._stop_heartbeat = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None
        self._log_path: Path | None = None
        self._rank = -1

    def _start(self, trainer) -> None:
        if self._heartbeat_thread is not None:
            return
        self._rank = int(trainer.global_rank)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = self.output_dir / f"rank_{self._rank}.log"
        self._log_path.write_text("", encoding="utf-8")
        self._append(trainer, "fit_start", force=True)
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            args=(trainer,),
            name=f"ddp-progress-rank-{self._rank}",
            daemon=True,
        )
        self._heartbeat_thread.start()

    def _stop(self, trainer, phase: str) -> None:
        self._set_state(trainer, phase, force=True)
        self._stop_heartbeat.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=5)
            self._heartbeat_thread = None

    def _heartbeat_loop(self, trainer) -> None:
        while not self._stop_heartbeat.wait(self.heartbeat_seconds):
            self._append(trainer, "heartbeat", force=True)

    def _set_state(
        self,
        trainer,
        phase: str,
        batch_idx: int | None = None,
        force: bool = False,
    ) -> None:
        with self._lock:
            self._state.update(
                {
                    "phase": phase,
                    "epoch": int(trainer.current_epoch),
                    "global_step": int(trainer.global_step),
                }
            )
            if batch_idx is not None:
                self._state["batch_idx"] = int(batch_idx)
        if force or self._should_log(batch_idx):
            self._append(trainer, phase, force=True)

    def _should_log(self, batch_idx: int | None) -> bool:
        return batch_idx is not None and batch_idx % self.log_every_n_steps == 0

    def _append(self, trainer, event: str, force: bool = False) -> None:
        if self._log_path is None:
            return
        with self._lock:
            state = dict(self._state)
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        line = (
            f"{timestamp} pid={os.getpid()} rank={self._rank} event={event} "
            f"phase={state['phase']} epoch={state['epoch']} "
            f"global_step={state['global_step']} batch_idx={state['batch_idx']}\n"
        )
        with self._log_path.open("a", encoding="utf-8") as stream:
            stream.write(line)
            stream.flush()

    def on_fit_start(self, trainer, pl_module) -> None:
        self._start(trainer)

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        self._set_state(trainer, "train_epoch_start", force=True)

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        self._set_state(trainer, "train_epoch_end", force=True)

    def on_train_batch_start(
        self, trainer, pl_module, batch, batch_idx: int
    ) -> None:
        self._set_state(trainer, "train_batch_start", batch_idx)

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx: int
    ) -> None:
        self._set_state(trainer, "train_batch_end", batch_idx)

    def on_before_backward(self, trainer, pl_module, loss) -> None:
        self._set_state(trainer, "before_backward")

    def on_after_backward(self, trainer, pl_module) -> None:
        self._set_state(trainer, "after_backward")

    def on_before_optimizer_step(self, trainer, pl_module, optimizer) -> None:
        self._set_state(trainer, "before_optimizer_step")

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        self._set_state(trainer, "validation_epoch_start", force=True)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        self._set_state(trainer, "validation_epoch_end", force=True)

    def on_validation_batch_start(
        self, trainer, pl_module, batch, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        self._set_state(trainer, "validation_batch_start", batch_idx)

    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._set_state(trainer, "validation_batch_end", batch_idx)

    def on_exception(self, trainer, pl_module, exception: BaseException) -> None:
        self._set_state(
            trainer,
            f"exception:{type(exception).__name__}:{exception}",
            force=True,
        )

    def on_fit_end(self, trainer, pl_module) -> None:
        self._stop(trainer, "fit_end")

    def teardown(self, trainer, pl_module, stage: str) -> None:
        if self._heartbeat_thread is not None:
            self._stop(trainer, f"teardown:{stage}")
