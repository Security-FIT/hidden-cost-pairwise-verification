import math
import time
from contextlib import nullcontext

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import det_curve
from torch.nn import CrossEntropyLoss

from trainers.BaseTrainer import BaseTrainer
from trainers.utils import bootstrap_eer_confidence_interval, interspeech_eer_confidence_interval

DEFAULT_FORENSICS_FPRS = (0.0001, 0.001, 0.01, 0.05)


def _parse_epoch_mult_schedule(schedule: str) -> list[tuple[int, float]]:
    points: list[tuple[int, float]] = []
    for entry in schedule.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if ":" not in entry:
            raise ValueError(
                "Invalid LR epoch multiplier entry. Use 'epoch:mult' pairs, e.g. 11:0.1,21:0.01."
            )
        epoch_str, mult_str = entry.split(":", 1)
        epoch = int(epoch_str.strip())
        mult = float(mult_str.strip())
        if epoch < 1:
            raise ValueError("LR epoch multipliers must use epochs >= 1.")
        points.append((epoch, mult))
    if not points:
        raise ValueError("LR epoch multiplier schedule is empty.")
    points.sort(key=lambda item: item[0])
    return points


def _tpr_at_fpr_targets(
    fpr: np.ndarray, fnr: np.ndarray, thresholds: np.ndarray, targets: tuple[float, ...]
) -> list[dict[str, float | bool | None]]:
    results: list[dict[str, float | bool | None]] = []
    tpr = 1.0 - fnr
    for target in targets:
        if fpr.size == 0:
            results.append({"fpr_target": target, "fpr": None, "tpr": None, "threshold": None, "met_target": False})
            continue
        finite = np.isfinite(fpr) & np.isfinite(tpr) & np.isfinite(thresholds)
        if not np.any(finite):
            results.append({"fpr_target": target, "fpr": None, "tpr": None, "threshold": None, "met_target": False})
            continue
        fpr_f = fpr[finite]
        tpr_f = tpr[finite]
        thr_f = thresholds[finite]
        valid = np.where(fpr_f <= target)[0]
        if valid.size > 0:
            idx = int(valid[np.nanargmax(fpr_f[valid])])
            met = True
        else:
            idx = int(np.nanargmin(fpr_f))
            met = False
        fpr_val = None if math.isnan(fpr_f[idx]) else float(fpr_f[idx])
        tpr_val = None if math.isnan(tpr_f[idx]) else float(tpr_f[idx])
        thr_val = None if math.isnan(thr_f[idx]) else float(thr_f[idx])
        results.append(
            {"fpr_target": target, "fpr": fpr_val, "tpr": tpr_val, "threshold": thr_val, "met_target": met}
        )
    return results


class BaseFFTrainer(BaseTrainer):
    def __init__(self, model, device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__(model, device)

        # Mabye TODO??? Add class weights for the loss function - maybe not necessary since we have weighted sampler
        self.lossfn = CrossEntropyLoss()  # Should also try with BCELoss
        lr = getattr(model, "default_lr", 1e-3)
        weight_decay = getattr(model, "default_weight_decay", 0.0)
        param_groups = None
        if hasattr(model, "get_param_groups"):
            try:
                param_groups = model.get_param_groups()
            except Exception:
                param_groups = None
        if param_groups:
            self.optimizer = torch.optim.Adam(param_groups, lr=lr, weight_decay=weight_decay)
        else:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.device = device

        self.model = model.to(device)
        self.scheduler = None

        # A statistics tracker dict for the training and validation losses, accuracies and EERs
        self.statistics = {
            "train_losses": [],
            "train_accuracies": [],
            "val_losses": [],
            "val_accuracies": [],
            "val_eers": [],
            "val_eer_cis": [],
            "val_epochs": [],
        }
        self._cumulative_gpu_seconds = 0.0
        self._use_amp_train = False
        self._use_amp_eval = False
        self._amp_dtype = torch.bfloat16
        self._grad_scaler: torch.cuda.amp.GradScaler | None = None
        self._max_train_batches: int | None = None
        self._grad_accum_steps: int = 1
        self._grad_accum_counter: int = 0
        # For tiny microbatches, guard against spiky updates
        self.grad_clip_norm: float | None = 1.0
        self._clip_batch_threshold: int = 8  # only clip when microbatch < this
        self._last_batch_size: int | None = None
        self._lr_ramp_enabled: bool = False
        self._lr_ramp_start_mult: float = 1.0
        self._lr_ramp_target_mult: float = 1.0
        self._lr_ramp_steps: int = 0
        self._lr_ramp_step_count: int = 0
        self._lr_ramp_base_lrs: list[float] | None = None
        self._epoch_lr_multipliers: list[tuple[int, float]] | None = None
        self._epoch_lr_base_lrs: list[float] | None = None

    def _clear_cuda_cache(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def configure_epoch_lr_multipliers(self, schedule: str | None) -> None:
        if schedule:
            self._epoch_lr_multipliers = _parse_epoch_mult_schedule(schedule)
        else:
            self._epoch_lr_multipliers = None

    def _apply_epoch_lr_multiplier(self, epoch: int) -> None:
        if not self._epoch_lr_multipliers:
            return
        if self._epoch_lr_base_lrs is None:
            self._epoch_lr_base_lrs = [group["lr"] for group in self.optimizer.param_groups]
        mult = self._epoch_lr_multipliers[0][1]
        for epoch_point, epoch_mult in self._epoch_lr_multipliers:
            if epoch >= epoch_point:
                mult = epoch_mult
            else:
                break
        for base_lr, group in zip(self._epoch_lr_base_lrs, self.optimizer.param_groups):
            group["lr"] = base_lr * mult
        print(f"[lr] epoch={epoch} mult={mult:.4f}")

    def train(
        self,
        train_dataloader,
        val_dataloader,
        numepochs=50,
        start_epoch=1,
        validation_interval: int = 5,
        compute_confidence: bool = True,
        bootstrap_iters: int = 1000,
        bootstrap_random_state: int | None = 0,
        stop_on_plateau: bool = False,
        patience: int | None = None,
        min_delta: float = 0.001,
    ):
        """
        Common training loop

        Train the model on the given dataloader for the given number of epochs
        Uses the optimizer and loss function defined in the constructor

        param train_dataloader: Dataloader loading the training data
        param val_dataloader: Dataloader loading the validation/dev data
        param numepochs: Number of epochs to train for (default 50 for Stage 1)
        param start_epoch: Epoch to start from (1-indexed)
        param validation_interval: Run dev validation every N epochs (default 5)
        param compute_confidence: Whether to compute EER CIs (Interspeech framework with bootstrap fallback)
        param bootstrap_iters: Number of bootstrap samples for CI estimation
        param bootstrap_random_state: Seed for bootstrap fallback
        param stop_on_plateau: Toggle early stopping (disabled by default for Stage 1)
        param patience: Number of validations to wait for improvement when stop_on_plateau is True
        param min_delta: Absolute EER improvement required to reset plateau counter
        """
        best_eer = math.inf
        best_val_loss = math.inf
        best_val_loss_for_plateau = math.inf
        epochs_without_improvement = 0
        # Configure scheduler if the model exposes one
        if self.scheduler is None and hasattr(self.model, "build_scheduler"):
            try:
                self.scheduler = self.model.build_scheduler(self.optimizer, numepochs)
            except Exception:
                self.scheduler = None

        for epoch in range(start_epoch, start_epoch + numepochs):  # 1-indexed epochs
            epoch_start = time.time()
            print(f"Starting epoch {epoch} with {len(train_dataloader)} batches")

            self.model.train()  # Set model to training mode
            self._apply_epoch_lr_multiplier(epoch)
            if hasattr(train_dataloader, "set_epoch"):
                train_dataloader.set_epoch(epoch)

            accuracies, losses = self.train_epoch(train_dataloader)
            if hasattr(train_dataloader, "get_last_epoch_stats"):
                stats = train_dataloader.get_last_epoch_stats()
                if stats:
                    if "stream_ratios" in stats:
                        ratios = stats.get("stream_ratios", {})
                        per_batch = stats.get("stream_per_batch", {})
                        batches = stats.get("stream_batches", {})
                        mix_str = ",".join(
                            f"{name}:{ratio:.3f}" for name, ratio in ratios.items()
                        )
                        msg = (
                            "[curriculum] "
                            f"epoch={stats.get('epoch')} "
                            f"steps={stats.get('steps_per_epoch')} "
                            f"mix={mix_str}"
                        )
                        if per_batch:
                            per_batch_str = ",".join(
                                f"{name}:{count}" for name, count in per_batch.items()
                            )
                            msg += f" per_batch={per_batch_str}"
                        if batches:
                            batches_str = ",".join(
                                f"{name}:{count}" for name, count in batches.items()
                            )
                            msg += f" batches={batches_str}"
                    else:
                        msg = (
                            "[curriculum] "
                            f"epoch={stats.get('epoch')} "
                            f"p_hard={stats.get('p_hard'):.3f} "
                            f"hard_batches={stats.get('hard_batches')} "
                            f"easy_batches={stats.get('easy_batches')} "
                            f"steps={stats.get('steps_per_epoch')}"
                        )
                        if "pos_per_batch" in stats and "neg_per_batch" in stats:
                            msg += (
                                f" pos_per_batch={stats.get('pos_per_batch')} "
                                f"neg_per_batch={stats.get('neg_per_batch')}"
                            )
                        if "easy_label_ratio" in stats and "hard_label_ratio" in stats:
                            msg += (
                                " label_ratio="
                                f"easy:{stats.get('easy_label_ratio'):.3f},"
                                f"hard:{stats.get('hard_label_ratio'):.3f},"
                                f"overall:{stats.get('overall_label_ratio'):.3f}"
                            )
                    print(msg)

            # Save epoch statistics
            epoch_accuracy = np.mean(accuracies)
            epoch_loss = np.mean(losses)
            print(
                f"Epoch {epoch} finished,",
                f"training loss: {np.mean(losses)},",
                f"training accuracy: {np.mean(accuracies)}",
            )

            self.statistics["train_losses"].append(epoch_loss)
            self.statistics["train_accuracies"].append(epoch_accuracy)
            self._log_ffattn_params(epoch)
            train_phase_seconds = time.time() - epoch_start

            # Every epoch
            # plot losses and accuracy and save the model
            # validate on the validation set (incl. computing EER)
            # self._plot_loss_accuracy(
            #     self.statistics["train_losses"],
            #     self.statistics["train_accuracies"],
            #     f"Training epoch {epoch}",
            # )
            self.save_model(f"{type(self.model).__name__}_{epoch}.pt")

            epoch_seconds = time.time() - epoch_start
            if torch.cuda.is_available():
                gpu_count = max(torch.cuda.device_count(), 1)
                gpu_seconds = train_phase_seconds * gpu_count
                self._cumulative_gpu_seconds += gpu_seconds
                device_name = torch.cuda.get_device_name(0)
                print(
                    f"Epoch {epoch} GPU time: {gpu_seconds / 3600:.4f} GPU-hours "
                    f"(elapsed {epoch_seconds / 60:.2f} min across {gpu_count} GPU(s), device {device_name})"
                )
                print(
                    f"Cumulative GPU time: {self._cumulative_gpu_seconds / 3600:.4f} GPU-hours"
                )
            else:
                print(f"Epoch {epoch} elapsed time: {epoch_seconds / 60:.2f} min (CPU execution)")

            should_validate = (
                validation_interval is None
                or validation_interval <= 0
                or ((epoch - start_epoch + 1) % validation_interval == 0)
            )
            if should_validate:
                self._clear_cuda_cache()
                val_metrics = self._evaluate_validation(
                    val_dataloader,
                    save_scores=False,
                    plot_det=False,
                    subtitle=str(epoch),
                    compute_confidence=compute_confidence,
                    bootstrap_iters=bootstrap_iters,
                    bootstrap_random_state=bootstrap_random_state,
                )
                self._clear_cuda_cache()

                val_loss = val_metrics["loss"]
                val_accuracy = val_metrics["accuracy"]
                eer = val_metrics["eer"]
                eer_ci = val_metrics.get("eer_ci")

                print(f"Validation loss: {val_loss}, validation accuracy: {val_accuracy*100}%")
                print(f"Validation EER: " + ("None" if eer is None else f"{eer*100}%"))
                if eer_ci:
                    print(f"Validation EER 95% CI: [{eer_ci[0]*100:.2f}%, {eer_ci[1]*100:.2f}%]")
                if val_metrics.get("forensics_tail"):
                    print("Validation forensics tail (TPR@FPR):")
                    for op in val_metrics["forensics_tail"]:
                        tpr_str = "n/a" if op["tpr"] is None else f"{op['tpr']*100:.2f}%"
                        fpr_str = "n/a" if op["fpr"] is None else f"{op['fpr']*100:.3f}%"
                        prefix = "<=" if op.get("met_target") else "≈"
                        print(
                            f"  FPR{prefix}{op['fpr_target']*100:.2f}% (achieved {fpr_str}) "
                            f"-> TPR={tpr_str} (thr={op['threshold']})"
                        )

                self.statistics["val_epochs"].append(epoch)
                self.statistics["val_losses"].append(val_loss)
                self.statistics["val_accuracies"].append(val_accuracy)
                self.statistics["val_eers"].append(eer)
                self.statistics["val_eer_cis"].append(eer_ci)

                if math.isfinite(val_loss) and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    # Keep a simple, stable name for downstream scripts.
                    self.save_model("best_model.pth")
                    print(f"Saved new best_model.pth with dev loss {val_loss}.")

                num_classes = getattr(self.model, "num_classes", 2)
                if num_classes == 2:
                    improved = eer is not None and (eer + min_delta < best_eer)
                    if improved:
                        best_eer = eer
                        epochs_without_improvement = 0
                    else:
                        epochs_without_improvement += 1
                else:
                    improved = math.isfinite(val_loss) and (val_loss + min_delta < best_val_loss_for_plateau)
                    if improved:
                        best_val_loss_for_plateau = val_loss
                        epochs_without_improvement = 0
                    else:
                        epochs_without_improvement += 1

                if stop_on_plateau and patience is not None and epochs_without_improvement >= patience:
                    metric_name = "dev EER" if num_classes == 2 else "dev loss"
                    print(
                        f"Stopping at epoch {epoch} - no {metric_name} improvement for {patience} validation(s)."
                    )
                    break

            # Step LR scheduler once per epoch
            if self.scheduler is not None:
                self.scheduler.step()

        # self._plot_loss_accuracy(
        #     self.statistics["val_losses"], self.statistics["val_accuracies"], "Validation"
        # )
        # self._plot_eer(self.statistics["val_eers"], "Validation")

    def train_epoch(self, train_dataloader) -> tuple[list[float], list[float]]:
        """
        Train the model for one epoch on the given dataloader

        return: Tuple(list of accuracies, list of losses)
        """
        raise NotImplementedError("Child classes should implement the train_epoch method")

    def set_amp_eval(self, enabled: bool, dtype: torch.dtype | None = None) -> None:
        """
        Enable mixed-precision (AMP) for validation/evaluation passes.
        """
        if enabled:
            resolved_dtype = self._resolve_amp_dtype(dtype, "evaluation")
            if resolved_dtype is None:
                self._use_amp_eval = False
                return
            self._amp_dtype = resolved_dtype
            self._use_amp_eval = True
        else:
            self._use_amp_eval = False

    def set_amp_train(self, enabled: bool, dtype: torch.dtype | None = None) -> None:
        """
        Enable mixed-precision (AMP) for training passes.
        """
        if enabled:
            resolved_dtype = self._resolve_amp_dtype(dtype, "training")
            if resolved_dtype is None:
                self._use_amp_train = False
                self._configure_grad_scaler()
                return
            self._amp_dtype = resolved_dtype
            self._use_amp_train = True
        else:
            self._use_amp_train = False
        self._configure_grad_scaler()

    def set_max_train_batches(self, max_batches: int | None):
        """
        Limit the number of training batches processed each epoch.
        """
        if max_batches is None or max_batches <= 0:
            self._max_train_batches = None
        else:
            self._max_train_batches = max_batches

    def set_grad_accum_steps(self, steps: int | None):
        """
        Configure gradient accumulation steps (train-time microbatching).
        """
        if steps is None or steps < 1:
            self._grad_accum_steps = 1
        else:
            self._grad_accum_steps = steps

    def configure_lr_ramp(
        self,
        enabled: bool,
        start_mult: float,
        target_mult: float,
        ramp_steps: int,
    ) -> None:
        """
        Configure a linear LR ramp (per optimizer step) relative to the base LR.
        """
        if not enabled:
            self._lr_ramp_enabled = False
            self._lr_ramp_base_lrs = None
            return
        if start_mult <= 0 or target_mult <= 0:
            raise ValueError("LR ramp multipliers must be positive.")
        if ramp_steps < 0:
            raise ValueError("LR ramp steps must be >= 0.")
        self._lr_ramp_enabled = True
        self._lr_ramp_start_mult = float(start_mult)
        self._lr_ramp_target_mult = float(target_mult)
        self._lr_ramp_steps = int(ramp_steps)
        self._lr_ramp_step_count = 0
        self._lr_ramp_base_lrs = [group["lr"] for group in self.optimizer.param_groups]
        initial_mult = (
            self._lr_ramp_start_mult if self._lr_ramp_steps > 0 else self._lr_ramp_target_mult
        )
        self._apply_lr_multiplier(initial_mult)
        print(
            "LR ramp enabled: "
            f"{self._lr_ramp_start_mult}x -> {self._lr_ramp_target_mult}x "
            f"over {self._lr_ramp_steps} optimizer step(s)."
        )

    def _train_batch_iterator(self, dataloader):
        if self._max_train_batches is None:
            yield from dataloader
        else:
            for idx, batch in enumerate(dataloader):
                if idx >= self._max_train_batches:
                    break
                yield batch

    def _train_batch_total(self, dataloader):
        if self._max_train_batches is not None:
            try:
                return min(self._max_train_batches, len(dataloader))
            except TypeError:
                return self._max_train_batches
        try:
            return len(dataloader)
        except TypeError:
            return None

    def _reset_grad_accumulation(self):
        """
        Clear gradients and reset accumulation counter.
        """
        self._grad_accum_counter = 0
        self.optimizer.zero_grad(set_to_none=True)

    def _eval_autocast_context(self):
        if self._use_amp_eval and torch.cuda.is_available():
            return torch.amp.autocast(device_type="cuda", dtype=self._amp_dtype)
        return nullcontext()

    def _train_autocast_context(self):
        if self._use_amp_train and torch.cuda.is_available():
            return torch.amp.autocast(device_type="cuda", dtype=self._amp_dtype)
        return nullcontext()

    def _resolve_amp_dtype(self, dtype: torch.dtype | None, context: str) -> torch.dtype | None:
        if not torch.cuda.is_available():
            print(f"AMP {context} requested but CUDA is unavailable; running in full precision.")
            return None
        requested = dtype or self._amp_dtype
        if requested not in (torch.float16, torch.bfloat16):
            print(
                f"Unsupported AMP dtype {requested}; defaulting to torch.bfloat16 "
                f"for {context} autocast."
            )
            requested = torch.bfloat16
        if requested == torch.bfloat16:
            bf16_supported = getattr(torch.cuda, "is_bf16_supported", None)
            if not (callable(bf16_supported) and bf16_supported()):
                print(
                    f"bfloat16 AMP requested for {context} but the current GPU lacks support; "
                    "falling back to float16."
                )
                requested = torch.float16
        return requested

    def _configure_grad_scaler(self) -> None:
        if self._use_amp_train and self._amp_dtype == torch.float16 and torch.cuda.is_available():
            self._grad_scaler = torch.cuda.amp.GradScaler()
        else:
            self._grad_scaler = None

    def set_grad_clip_norm(self, max_norm: float | None) -> None:
        """
        Configure gradient clipping (global norm). Use None or <=0 to disable.
        """
        if max_norm is None or max_norm <= 0:
            self.grad_clip_norm = None
        else:
            self.grad_clip_norm = max_norm

    def _record_batch_size(self, batch_size: int | None) -> None:
        """Store the last seen microbatch size for conditional clipping."""
        self._last_batch_size = batch_size

    def _apply_grad_clip(self) -> None:
        if self.grad_clip_norm is None:
            return
        if self._last_batch_size is not None and self._last_batch_size >= self._clip_batch_threshold:
            return
        if self._grad_scaler is not None:
            # Unscale first when using AMP so clipping is meaningful
            self._grad_scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

    def _compute_grad_norms(self) -> dict[str, float]:
        head_grads = []
        proc_grads = []
        for name, p in self.model.named_parameters():
            if not p.requires_grad or p.grad is None:
                continue
            if name.startswith("feature_processor"):
                proc_grads.append(p.grad.flatten())
            else:
                head_grads.append(p.grad.flatten())
        norms = {}
        norms["head"] = torch.linalg.norm(torch.cat(head_grads)).item() if head_grads else float("nan")
        norms["proc"] = torch.linalg.norm(torch.cat(proc_grads)).item() if proc_grads else float("nan")
        return norms

    def _step_optimizer(self, loss: torch.Tensor) -> dict[str, float] | None:
        """
        Backward and (optionally) optimizer step with gradient accumulation.
        """
        scaled_loss = loss / self._grad_accum_steps
        if self._grad_scaler is not None:
            self._grad_scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        self._grad_accum_counter += 1
        if self._grad_accum_counter % self._grad_accum_steps == 0:
            self._apply_grad_clip()
            grad_norms = self._compute_grad_norms()
            self._apply_lr_ramp_for_step()
            if self._grad_scaler is not None:
                self._grad_scaler.step(self.optimizer)
                self._grad_scaler.update()
            else:
                self.optimizer.step()
            self._register_optimizer_step()
            self.optimizer.zero_grad(set_to_none=True)
            return grad_norms
        return None

    def _finalize_optimizer_step(self) -> None:
        """
        If gradients remain (batch count not divisible by accum steps), apply a final step.
        """
        if self._grad_accum_counter % self._grad_accum_steps != 0:
            self._apply_grad_clip()
            self._apply_lr_ramp_for_step()
            if self._grad_scaler is not None:
                self._grad_scaler.step(self.optimizer)
                self._grad_scaler.update()
            else:
                self.optimizer.step()
            self._register_optimizer_step()
            self.optimizer.zero_grad(set_to_none=True)
        self._grad_accum_counter = 0

    def _apply_lr_multiplier(self, mult: float) -> None:
        if self._lr_ramp_base_lrs is None:
            self._lr_ramp_base_lrs = [group["lr"] for group in self.optimizer.param_groups]
        for group, base_lr in zip(self.optimizer.param_groups, self._lr_ramp_base_lrs):
            group["lr"] = base_lr * mult

    def _lr_multiplier_for_step(self, step_index: int) -> float:
        if self._lr_ramp_steps <= 0:
            return self._lr_ramp_target_mult
        if step_index >= self._lr_ramp_steps:
            return self._lr_ramp_target_mult
        if self._lr_ramp_steps == 1:
            return self._lr_ramp_target_mult
        progress = (step_index - 1) / (self._lr_ramp_steps - 1)
        return self._lr_ramp_start_mult + (
            (self._lr_ramp_target_mult - self._lr_ramp_start_mult) * progress
        )

    def _apply_lr_ramp_for_step(self) -> None:
        if not self._lr_ramp_enabled:
            return
        step_index = self._lr_ramp_step_count + 1
        mult = self._lr_multiplier_for_step(step_index)
        self._apply_lr_multiplier(mult)

    def _register_optimizer_step(self) -> None:
        if self._lr_ramp_enabled:
            self._lr_ramp_step_count += 1

    def _log_ffattn_params(self, epoch: int) -> None:
        """
        Print key FFAttn scalars so we can track their evolution during training.
        """
        model = getattr(self, "model", None)
        if model is None or not type(model).__name__.startswith("FFAttn"):
            return

        def _to_float(val: float | torch.Tensor) -> float:
            if isinstance(val, torch.Tensor):
                return float(val.detach().item())
            return float(val)

        log_entry: dict[str, float] = {"epoch": float(epoch)}
        if hasattr(model, "attn_gain"):
            attn_cap = _to_float(getattr(model, "attn_cap", 1.0))
            attn_gain_raw = _to_float(getattr(model, "attn_gain"))
            attn_gain_scaled = _to_float(torch.sigmoid(getattr(model, "attn_gain").detach()) * attn_cap)
            log_entry.update(
                {
                    "attn_gain_raw": attn_gain_raw,
                    "attn_gain_scaled": attn_gain_scaled,
                    "attn_cap": attn_cap,
                }
            )
        if hasattr(model, "atten_beta"):
            beta_raw = _to_float(getattr(model, "atten_beta"))
            beta_sigmoid = _to_float(torch.sigmoid(getattr(model, "atten_beta").detach()))
            log_entry.update({"atten_beta_raw": beta_raw, "atten_beta_sigmoid": beta_sigmoid})

        if len(log_entry) > 1:
            formatted = " ".join([f"{k}={v:.4f}" for k, v in log_entry.items() if k != "epoch"])
            print(f"[epoch {epoch}] FFAttn params: {formatted}")
            self.statistics.setdefault("ffattn_params", []).append(log_entry)

    def _evaluate_validation(
        self,
        val_dataloader,
        save_scores=False,
        plot_det: bool = False,
        subtitle: str = "",
        compute_confidence: bool = False,
        bootstrap_iters: int = 1000,
        bootstrap_random_state: int | None = 0,
    ) -> dict:
        """
        Run validation and return metrics, optionally with a bootstrap CI for EER.
        """
        self.model.eval()

        with torch.no_grad():
            val_outputs = self.val_epoch(val_dataloader, save_scores)

        # Compatibility: allow val_epoch to optionally return condition identifiers
        if len(val_outputs) == 6:
            losses, labels, scores, predictions, file_names, conditions = val_outputs
        else:
            losses, labels, scores, predictions, file_names = val_outputs
            conditions = None

        if save_scores:
            score_path = self._resolve_output_path(f"{type(self.model).__name__}_{subtitle}_scores.txt")
            with open(score_path, "w") as f:
                for file_name, score, label in zip(file_names, scores, labels):
                    f.write(f"{file_name},{score},{'nan' if math.isnan(label) else int(label)}\n")

        val_loss = np.mean(losses).astype(float)
        val_accuracy = np.mean(np.array(labels) == np.array(predictions))
        num_classes = getattr(self.model, "num_classes", 2)
        if num_classes != 2:
            eer = None
        elif None in labels or any(map(lambda x: math.isnan(x), labels)):
            print("Skipping EER calculation due to missing labels")
            eer = None
        elif len(set(labels)) == 1:
            print("Skipping EER calculation due to all labels being the same")
            eer = None
        else:
            eer = self.calculate_EER(labels, scores, plot_det=plot_det, det_subtitle=subtitle)

        forensics_tail = None
        if num_classes == 2 and eer is not None:
            det_fpr, det_fnr, det_thresholds = det_curve(labels, scores, pos_label=0)
            forensics_tail = _tpr_at_fpr_targets(
                det_fpr,
                det_fnr,
                det_thresholds,
                DEFAULT_FORENSICS_FPRS,
            )

        eer_ci = None
        if compute_confidence and eer is not None:
            interspeech_eer, interspeech_ci = interspeech_eer_confidence_interval(
                labels,
                scores,
                n_bootstrap=bootstrap_iters,
                conditions=conditions,
            )
            if interspeech_ci is not None:
                eer = interspeech_eer
                eer_ci = interspeech_ci
            else:
                eer_ci = bootstrap_eer_confidence_interval(
                    labels,
                    scores,
                    n_bootstrap=bootstrap_iters,
                    random_state=bootstrap_random_state,
                    conditions=conditions,
                )

        return {
            "loss": val_loss,
            "accuracy": val_accuracy,
            "eer": eer,
            "eer_ci": eer_ci,
            "labels": labels,
            "scores": scores,
            "predictions": predictions,
            "file_names": file_names,
            "conditions": conditions,
            "forensics_tail": forensics_tail,
        }

    def val(
        self,
        val_dataloader,
        save_scores=False,
        plot_det=False,
        subtitle="",
        compute_confidence: bool = False,
        bootstrap_iters: int = 1000,
        bootstrap_random_state: int | None = 0,
    ) -> tuple[float, float, float | None]:
        """
        Common validation loop

        Validate the model on the given dataloader and return the loss, accuracy and EER

        param val_dataloader: Dataloader loading the validation/dev data

        return: Tuple(loss, accuracy, EER)
        """

        metrics = self._evaluate_validation(
            val_dataloader,
            save_scores=save_scores,
            plot_det=plot_det,
            subtitle=subtitle,
            compute_confidence=compute_confidence,
            bootstrap_iters=bootstrap_iters,
            bootstrap_random_state=bootstrap_random_state,
        )

        return metrics["loss"], metrics["accuracy"], metrics["eer"]

    def val_epoch(
        self, val_dataloader, save_scores=False
    ) -> tuple[list[float], list[float], list[float], list[int], list[str]]:
        """
        Validate the model for one epoch on the given dataloader

        return: Tuple(list of losses, list of labels, list of scores, list of predictions, list of file names)
        """
        raise NotImplementedError("Child classes should implement the val_epoch method")

    def eval(self, eval_dataloader, subtitle: str = ""):
        """
        Common evaluation code

        Evaluate the model on the given dataloader and print the loss, accuracy and EER

        param eval_dataloader: Dataloader loading the test data
        """

        # Reuse code from val() to evaluate the model on the eval set
        self._clear_cuda_cache()
        eval_loss, eval_accuracy, eer = self.val(
            eval_dataloader, save_scores=True, plot_det=True, subtitle=subtitle
        )
        self._clear_cuda_cache()
        print(f"Eval loss: {eval_loss}, eval accuracy: {eval_accuracy*100}%")
        print(f"Eval EER: {eer*100 if eer else None}%")

    def _plot_loss_accuracy(self, losses, accuracies, subtitle: str = ""):
        """
        Plot the loss and accuracy and save the graph to a file
        """
        plt.figure(figsize=(12, 6))
        plt.plot(losses, label="Loss")
        plt.plot(accuracies, label="Accuracy")
        plt.legend()
        plt.title(f"{type(self.model).__name__} Loss and Accuracy" + f" - {subtitle}" if subtitle else "")
        plt.xlabel("Epoch")
        plt.ylabel("Loss/Accuracy")
        loss_acc_path = self._resolve_output_path(f"{type(self.model).__name__}_loss_acc_{subtitle}.png")
        plt.savefig(loss_acc_path)

    def _plot_eer(self, eers, subtitle: str = ""):
        """
        Plot the EER and save the graph to a file
        """
        plt.figure(figsize=(12, 6))
        plt.plot(eers, label="EER")
        plt.legend()
        plt.title(f"{type(self.model).__name__} EER" + f" - {subtitle}" if subtitle else "")
        plt.xlabel("Epoch")
        plt.ylabel("EER")
        eer_path = self._resolve_output_path(f"{type(self.model).__name__}_EER_{subtitle}.png")
        plt.savefig(eer_path)

    def finetune(self, train_dataloader, val_dataloader, numepochs=5, finetune_ssl=False):
        """
        Fine-tune the model on the given dataloader for the given number of epochs.
        TODO: Maybe do finetuning based on steps instead of epochs?

        param train_dataloader: Dataloader loading the training data
        param val_dataloader: Dataloader loading the validation/dev data
        param numepochs: Number of epochs to fine-tune for
        param finetune_ssl: Whether to fine-tune the SSL extractor
        """

        self.model.extractor.finetune = finetune_ssl
        # Use the optimizer but with a smaller learning rate
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-6)
        self.model.train()  # Set model to training mode
        self.statistics = {  # Reset statistics
            "train_losses": [],
            "train_accuracies": [],
            "val_losses": [],
            "val_accuracies": [],
            "val_eers": [],
            "val_eer_cis": [],
            "val_epochs": [],
        }

        for epoch in range(1, numepochs + 1):
            print(f"Starting epoch {epoch} with {len(train_dataloader)} batches")

            accuracies, losses = self.train_epoch(train_dataloader)

            # Save epoch statistics
            epoch_accuracy = np.mean(accuracies)
            epoch_loss = np.mean(losses)
            print(
                f"Finetuning epoch {epoch} finished,",
                f"Finetuning training loss: {np.mean(losses)},",
                f"Finetuning training accuracy: {np.mean(accuracies)}",
            )

            self.statistics["train_losses"].append(epoch_loss)
            self.statistics["train_accuracies"].append(epoch_accuracy)
            self._log_ffattn_params(epoch)

            self.save_model(f"{type(self.model).__name__}_finetune_{epoch}.pt")

            epochs_to_val = 1  # Validate every epoch
            if epoch % epochs_to_val == 0:
                val_loss, val_accuracy, eer = self.val(val_dataloader, save_scores=True, subtitle=f"finetune_{epoch}")
                print(f"Validation loss: {val_loss}, validation accuracy: {val_accuracy*100}%")
                print(f"Validation EER: " + ("None" if eer == None else f"{eer*100}%"))
                self.statistics["val_losses"].append(val_loss)
                self.statistics["val_accuracies"].append(val_accuracy)
                self.statistics["val_eers"].append(eer)

                # Save best model (lowest validation loss)
                current_best_loss = self.statistics.get("best_val_loss", float("inf"))
                if val_loss < current_best_loss:
                    self.statistics["best_val_loss"] = val_loss
                    print(f"New best validation loss: {val_loss}. Saving best model.")
                    self.save_model(f"{type(self.model).__name__}_finetune_best.pt")

        # self._plot_eer(self.statistics["val_eers"], "Finetuning EER")
        # self._plot_loss_accuracy(
            # self.statistics["val_losses"], self.statistics["val_accuracies"], "Finetuning Loss & Accuracy"
        # )
