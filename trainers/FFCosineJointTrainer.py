from collections.abc import Iterable

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import Subset
from tqdm import tqdm

from classifiers.differential.FFCosine import FFCosineJoint
from trainers.BaseFFTrainer import BaseFFTrainer
from trainers.embedding_cache import EmbeddingCache, build_pair_keys


class FFCosineJointTrainer(BaseFFTrainer):
    """
    Joint trainer for FFCosineJoint:
      L = L_ce_global + lambda * L_pair
    """

    def __init__(
        self,
        model: FFCosineJoint,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__(model, device)
        self.global_lossfn = CrossEntropyLoss()
        self.pair_loss_weight = float(getattr(model, "pair_loss_weight", 0.1))

        self._model_label_map: dict[str, int] | None = getattr(model, "model_label_map", None)
        self._dataset_path_lookup_cache: dict[int, dict[str, int]] = {}
        self._loader_lookup_cache: dict[tuple[int, ...], dict[str, int]] = {}

    @staticmethod
    def _normalize_path(path: str) -> str:
        normalized = str(path).strip().replace("\\", "/")
        while normalized.startswith("./"):
            normalized = normalized[2:]
        while "//" in normalized:
            normalized = normalized.replace("//", "/")
        return normalized

    @staticmethod
    def _split_pair_id(pair_id: str) -> tuple[str, str]:
        # MLAAD pair id format is usually "path_A|path_B"; augmented rows append suffixes.
        parts = str(pair_id).split("|")
        if len(parts) < 2:
            return str(pair_id), str(pair_id)
        return parts[0], parts[1]

    def _unwrap_dataset(self, dataset):
        visited: set[int] = set()
        current = dataset
        while True:
            obj_id = id(current)
            if obj_id in visited:
                return current
            visited.add(obj_id)

            if isinstance(current, Subset):
                current = current.dataset
                continue

            base_dataset = getattr(current, "base_dataset", None)
            if base_dataset is not None:
                current = base_dataset
                continue

            inner_dataset = getattr(current, "dataset", None)
            if inner_dataset is not None and inner_dataset is not current:
                current = inner_dataset
                continue

            return current

    def _collect_root_datasets(self, dataloader) -> list:
        candidates = []
        if hasattr(dataloader, "dataset"):
            candidates.append(dataloader.dataset)

        # Curriculum loaders expose internal stream loaders.
        for attr in (
            "pos_loader",
            "easy_neg_loader",
            "hard_neg_loader",
            "aug_neg_loader",
            "easy_loader",
            "hard_loader",
            "aug_loader",
        ):
            loader = getattr(dataloader, attr, None)
            if loader is not None and hasattr(loader, "dataset"):
                candidates.append(loader.dataset)

        roots = []
        seen: set[int] = set()
        for ds in candidates:
            root = self._unwrap_dataset(ds)
            root_id = id(root)
            if root_id in seen:
                continue
            seen.add(root_id)
            roots.append(root)
        return roots

    def _ensure_model_label_map(self, model_names: Iterable[str]) -> None:
        names = sorted({str(name) for name in model_names})
        if not names:
            raise ValueError("Could not infer generator label map for FFCosineJointTrainer.")

        if self._model_label_map is None:
            if self.model.classifier.out_features < len(names):
                raise ValueError(
                    f"Classifier head too small for {len(names)} generators "
                    f"(out_features={self.model.classifier.out_features}). "
                    "Increase --num_classes."
                )
            self._model_label_map = {name: idx for idx, name in enumerate(names)}
            self.model.model_label_map = dict(self._model_label_map)
            return

        missing = sorted({name for name in names if name not in self._model_label_map})
        if missing:
            raise ValueError(
                "Encountered generator IDs not present in the existing label map: "
                + ", ".join(missing[:10])
                + ("..." if len(missing) > 10 else "")
            )

    def _register_path_label(self, mapping: dict[str, int], path: str, label: int) -> None:
        path_raw = str(path)
        path_norm = self._normalize_path(path_raw)
        for key in (path_raw, path_norm):
            existing = mapping.get(key)
            if existing is not None and existing != label:
                raise ValueError(f"Conflicting generator labels for path '{key}': {existing} vs {label}")
            mapping[key] = label

    def _build_path_lookup_for_dataset(self, dataset) -> dict[str, int]:
        root = self._unwrap_dataset(dataset)
        cache_key = id(root)
        cached = self._dataset_path_lookup_cache.get(cache_key)
        if cached is not None:
            return cached

        protocol_df = getattr(root, "protocol_df", None)
        if protocol_df is None:
            raise ValueError(
                "FFCosineJointTrainer requires datasets exposing protocol_df with model_name_A/model_name_B."
            )
        required = {"path_A", "path_B", "model_name_A", "model_name_B"}
        if not required.issubset(set(protocol_df.columns)):
            raise ValueError(
                "FFCosineJointTrainer requires protocol columns: path_A, path_B, model_name_A, model_name_B."
            )

        names = set(protocol_df["model_name_A"].astype(str)).union(
            set(protocol_df["model_name_B"].astype(str))
        )
        self._ensure_model_label_map(names)
        assert self._model_label_map is not None

        lookup: dict[str, int] = {}
        for path_col, model_col in (("path_A", "model_name_A"), ("path_B", "model_name_B")):
            paths = protocol_df[path_col].astype(str).tolist()
            models = protocol_df[model_col].astype(str).tolist()
            for path, model_name in zip(paths, models):
                label = self._model_label_map[model_name]
                self._register_path_label(lookup, path, label)

        self._dataset_path_lookup_cache[cache_key] = lookup
        return lookup

    def _build_path_lookup_for_loader(self, dataloader) -> dict[str, int]:
        roots = self._collect_root_datasets(dataloader)
        if not roots:
            raise ValueError("Could not resolve source datasets for joint training.")

        # Build/validate label map against the union of model names across all streams
        # (important for curriculum loaders where each stream can cover different subsets).
        all_model_names: set[str] = set()
        for root in roots:
            protocol_df = getattr(root, "protocol_df", None)
            if protocol_df is None:
                continue
            required = {"model_name_A", "model_name_B"}
            if required.issubset(set(protocol_df.columns)):
                all_model_names.update(protocol_df["model_name_A"].astype(str).tolist())
                all_model_names.update(protocol_df["model_name_B"].astype(str).tolist())
        if all_model_names:
            self._ensure_model_label_map(all_model_names)

        key = tuple(sorted(id(root) for root in roots))
        cached = self._loader_lookup_cache.get(key)
        if cached is not None:
            return cached

        merged: dict[str, int] = {}
        for root in roots:
            lookup = self._build_path_lookup_for_dataset(root)
            for path, label in lookup.items():
                existing = merged.get(path)
                if existing is not None and existing != label:
                    raise ValueError(f"Conflicting labels for path '{path}': {existing} vs {label}")
                merged[path] = label

        self._loader_lookup_cache[key] = merged
        return merged

    def _batch_global_labels(self, pair_ids, path_lookup: dict[str, int]) -> tuple[torch.Tensor, torch.Tensor]:
        gt_labels: list[int] = []
        test_labels: list[int] = []
        missing_examples: list[str] = []

        for pair_id in pair_ids:
            path_a, path_b = self._split_pair_id(pair_id)
            label_a = path_lookup.get(path_a)
            if label_a is None:
                label_a = path_lookup.get(self._normalize_path(path_a))
            label_b = path_lookup.get(path_b)
            if label_b is None:
                label_b = path_lookup.get(self._normalize_path(path_b))

            if label_a is None or label_b is None:
                missing_examples.append(str(pair_id))
                continue

            gt_labels.append(int(label_a))
            test_labels.append(int(label_b))

        if missing_examples:
            examples = ", ".join(missing_examples[:5])
            suffix = "..." if len(missing_examples) > 5 else ""
            raise ValueError(
                "Could not resolve generator labels for pair ids: "
                f"{examples}{suffix}. Ensure MLAAD pair protocols include model_name columns."
            )

        return (
            torch.tensor(gt_labels, dtype=torch.long, device=self.device),
            torch.tensor(test_labels, dtype=torch.long, device=self.device),
        )

    def _joint_loss(
        self,
        pair_logits: torch.Tensor,
        pair_label: torch.Tensor,
        cls_logits_gt: torch.Tensor,
        cls_logits_test: torch.Tensor,
        gt_global_label: torch.Tensor,
        test_global_label: torch.Tensor,
    ) -> torch.Tensor:
        pair_loss = self.lossfn(pair_logits, pair_label.long())
        global_loss_gt = self.global_lossfn(cls_logits_gt, gt_global_label)
        global_loss_test = self.global_lossfn(cls_logits_test, test_global_label)
        global_loss = 0.5 * (global_loss_gt + global_loss_test)
        return global_loss + (self.pair_loss_weight * pair_loss)

    def train_epoch(self, train_dataloader) -> tuple[list[float], list[float]]:
        losses: list[float] = []
        accuracies: list[float] = []
        path_lookup = self._build_path_lookup_for_loader(train_dataloader)

        self._reset_grad_accumulation()
        iterator = self._train_batch_iterator(train_dataloader)
        total = self._train_batch_total(train_dataloader)

        for pair_ids, gt, test, pair_label in tqdm(iterator, total=total):
            gt = gt.to(self.device)
            test = test.to(self.device)
            pair_label = pair_label.to(self.device)
            self._record_batch_size(len(pair_label))
            gt_global_label, test_global_label = self._batch_global_labels(pair_ids, path_lookup)

            with self._train_autocast_context():
                pair_logits, pair_probs, cls_logits_gt, cls_logits_test = self.model(
                    gt, test, label=pair_label, return_aux=True
                )
                loss = self._joint_loss(
                    pair_logits,
                    pair_label,
                    cls_logits_gt,
                    cls_logits_test,
                    gt_global_label,
                    test_global_label,
                )

            loss_value = loss.item()
            self._step_optimizer(loss)

            predicted = torch.argmax(pair_probs, dim=1)
            correct = (predicted == pair_label.long()).sum().item()
            accuracy = correct / len(pair_label)

            losses.append(loss_value)
            accuracies.append(accuracy)

        self._finalize_optimizer_step()
        return accuracies, losses

    def val_epoch(
        self, val_dataloader, save_scores=False
    ) -> tuple[list[float], list[float], list[float], list[int], list[str]]:
        losses: list[float] = []
        labels: list[float] = []
        scores: list[float] = []
        predictions: list[int] = []
        file_names: list[str] = []

        path_lookup = self._build_path_lookup_for_loader(val_dataloader)

        use_embedding_cache = (
            hasattr(self.model, "forward_from_embeddings")
            and hasattr(self.model, "extractor")
            and hasattr(self.model, "feature_processor")
        )
        embedding_cache = None
        if use_embedding_cache:
            embedding_cache = EmbeddingCache(self.model.extractor, self.model.feature_processor, self.device)

        for pair_ids, gt, test, pair_label in tqdm(val_dataloader):
            pair_label = pair_label.to(self.device)
            gt_global_label, test_global_label = self._batch_global_labels(pair_ids, path_lookup)

            if use_embedding_cache:
                keys_gt, keys_test = build_pair_keys(pair_ids, gt, test)
                emb_gt = embedding_cache.get_embeddings(gt, keys_gt, self._eval_autocast_context())
                emb_test = embedding_cache.get_embeddings(test, keys_test, self._eval_autocast_context())
                with self._eval_autocast_context():
                    pair_logits, pair_probs, cls_logits_gt, cls_logits_test = self.model.forward_from_embeddings(
                        emb_gt, emb_test, label=pair_label, return_aux=True
                    )
            else:
                gt = gt.to(self.device)
                test = test.to(self.device)
                with self._eval_autocast_context():
                    pair_logits, pair_probs, cls_logits_gt, cls_logits_test = self.model(
                        gt, test, label=pair_label, return_aux=True
                    )

            loss = self._joint_loss(
                pair_logits.float(),
                pair_label,
                cls_logits_gt.float(),
                cls_logits_test.float(),
                gt_global_label,
                test_global_label,
            )

            predictions.extend(torch.argmax(pair_probs, dim=1).tolist())
            # Keep compatibility with existing pair trainer score convention.
            scores.extend(pair_probs[:, 0].float().tolist())
            labels.extend(pair_label.tolist())
            losses.append(loss.item())

            if save_scores:
                file_names.extend(pair_ids)

        return losses, labels, scores, predictions, file_names
