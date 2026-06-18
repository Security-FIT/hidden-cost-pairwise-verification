from pathlib import Path

import torch

from .utils import calculate_EER


class BaseTrainer:
    def __init__(self, model, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model
        self.device = device
        self.output_dir = Path(".")

    def save_model(self, path: str):
        """
        Save the model to the given path
        If model is a PyTorch model, it will be saved using torch.save(state_dict)
        Problem is when non-PyTorch model contains a Pytorch component (e.g. extractor). In that case,
        the trainer should implement custom saving/loading methods.

        param path: Path to save the model to
        """
        save_path = self._resolve_output_path(path)
        if isinstance(self.model, torch.nn.Module):
            torch.save(self.model.state_dict(), save_path)
        else:
            raise NotImplementedError(
                "Child classes for non-PyTorch models need to implement save_model method"
            )

    def load_model(self, path: str):
        """
        Load the model from the given path
        Try to load the model as a PyTorch model using torch.load,
        otherwise, the child class trainer should implement custom loading method.

        param path: Path to load the model from
        """
        try:
            state_dict = torch.load(path, map_location=self.device, weights_only=True)
        except TypeError:
            # Compatibility with older torch versions that do not support weights_only.
            state_dict = torch.load(path, map_location=self.device)
        try:
            self.model.load_state_dict(state_dict)
            return
        except RuntimeError as exc:
            # Handle backward compatibility (e.g. new inference buffers)
            filtered_state = dict(state_dict)
            model_state = self.model.state_dict()
            dropped = []
            for key, value in list(filtered_state.items()):
                if key.startswith("classifier.") and key in model_state and hasattr(value, "shape"):
                    if value.shape != model_state[key].shape:
                        filtered_state.pop(key)
                        dropped.append(key)
            if dropped:
                print(f"[info] Dropping mismatched classifier keys: {', '.join(dropped)}")

            result = self.model.load_state_dict(filtered_state, strict=False)
            missing = set(result.missing_keys)
            unexpected = set(result.unexpected_keys)
            
            # Allow specific missing keys for decoupled calibration or new scale/bias params
            allowed_missing = {"eval_w", "eval_b", "is_calibrated", "scale", "bias"}
            
            # Allow discarding classifier head (unexpected keys)
            allowed_unexpected_prefix = "classifier."
            unexpected_filtered = {k for k in unexpected if not k.startswith(allowed_unexpected_prefix)}

            missing_allowed = all(k in allowed_missing or k.startswith("classifier.") for k in missing)
            if not unexpected_filtered and missing_allowed:
                print(f"[info] Loaded checkpoint with allowed mismatches (missing={missing}, discarded={unexpected})")
                return

            # If not resolved, try the upgrade mechanism
            upgraded, changed = self._upgrade_state_dict(state_dict)
            if not changed:
                # If we have missing keys that aren't allowed, or unexpected keys, re-raise original error
                # to maintain strictness.
                raise exc
            
            try:
                self.model.load_state_dict(upgraded)
                print(f"[info] Loaded checkpoint with compatibility key remap: {Path(path).name}")
                return
            except Exception:
                raise exc

    def _upgrade_state_dict(self, state_dict: dict) -> tuple[dict, bool]:
        """
        Compatibility shim for older checkpoints saved before certain module refactors.

        Currently supports upgrading BatchNorm1d layers that were replaced by AdaptiveNorm1d
        (which wraps BatchNorm1d as `.bn` and GroupNorm as `.gn`).
        """
        model_keys = list(self.model.state_dict().keys())
        adaptive_prefixes = set()
        for key in model_keys:
            if ".bn." in key:
                adaptive_prefixes.add(key.split(".bn.", 1)[0])
            if ".gn." in key:
                adaptive_prefixes.add(key.split(".gn.", 1)[0])

        if not adaptive_prefixes:
            return state_dict, False

        upgraded = dict(state_dict)
        changed = False

        for prefix in adaptive_prefixes:
            # Only attempt mapping when checkpoint has BN-style keys at this prefix
            # (running stats are unique to BN and will not appear for Linear weights).
            if f"{prefix}.running_mean" not in state_dict and f"{prefix}.running_var" not in state_dict:
                continue

            for attr in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked"):
                old_key = f"{prefix}.{attr}"
                bn_key = f"{prefix}.bn.{attr}"
                if old_key in state_dict and bn_key not in upgraded:
                    upgraded[bn_key] = state_dict[old_key]
                    changed = True

            for attr in ("weight", "bias"):
                old_key = f"{prefix}.{attr}"
                gn_key = f"{prefix}.gn.{attr}"
                if old_key in state_dict and gn_key not in upgraded:
                    upgraded[gn_key] = state_dict[old_key]
                    changed = True

            # Remove old keys to avoid "unexpected key(s)" under strict loading.
            for attr in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked"):
                old_key = f"{prefix}.{attr}"
                if old_key in upgraded:
                    upgraded.pop(old_key, None)

        return upgraded, changed

    def calculate_EER(self, labels, predictions, plot_det: bool, det_subtitle: str) -> float:
        return calculate_EER(
            type(self.model).__name__,
            labels,
            predictions,
            plot_det,
            det_subtitle,
            output_dir=self.output_dir,
        )

    def set_output_dir(self, output_dir: str | Path):
        target_dir = Path(output_dir).expanduser()
        target_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = target_dir

    def _resolve_output_path(self, path: str | Path) -> Path:
        target_path = Path(path)
        if not target_path.is_absolute():
            target_path = self.output_dir / target_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        return target_path

    def train(self, train_dataloader, val_dataloader, numepochs: int = 20):
        raise NotImplementedError("Child classes should implement the train method")
    
    def val(self, val_dataloader):
        raise NotImplementedError("Child classes should implement the val method")

    def eval(self, eval_dataloader, subtitle: str = ""):
        raise NotImplementedError("Child classes should implement the eval method")
