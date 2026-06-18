#!/usr/bin/env python3
"""
Utilities for Stage-3 hard-negative mining:
  1) Sample balanced negative pairs per anchor from the single-utterance train CSV.
  2) Score those pairs with a specified checkpoint/model.

The final hard-mined train protocol is built by scripts/generate_mlaad_train_pairs.py
via its `hardmined` method (consumes the scored CSV produced here).
"""

from __future__ import annotations

import argparse
import os
import random
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import build_model
from datasets.MLAAD import MLAADDataset_pair
from datasets.utils import custom_pair_batch_create
from scripts.generate_mlaad_train_pairs import load_embedding_table
from trainers.embedding_cache import EmbeddingCache, build_pair_keys


Pair = Tuple[str, str, str, str, int]
PairRow = dict[str, object]


def sample_pairs(
    train_csv: Path,
    path_column: str,
    source_column: str,
    sys_column: str | None,
    arch_column: str | None,
    seen_column: str | None,
    path_map: Path | None,
    cohort_stats_out: Path | None,
    global_canon_cap: int | None,
    pos_per_anchor: int,
    max_neg_candidates_per_anchor: int,
    seed: int,
) -> List[PairRow]:
    """
    For every utterance (anchor) in the train CSV, sample up to max_neg_candidates_per_anchor
    negative trials using a light stratification across model architecture/system metadata,
    and optionally some positives. Anchor is always path_A; partner is path_B.
    """
    if pos_per_anchor < 0:
        raise ValueError("pos_per_anchor must be non-negative.")
    if max_neg_candidates_per_anchor <= 0:
        raise ValueError("max_neg_candidates_per_anchor must be positive.")
    df = pd.read_csv(train_csv)
    missing_cols = {path_column, source_column} - set(df.columns)
    if missing_cols:
        raise ValueError(f"{train_csv} is missing required columns: {', '.join(sorted(missing_cols))}")

    sys_column = sys_column or source_column
    if sys_column not in df.columns:
        raise ValueError(f"{train_csv} is missing sys column '{sys_column}'.")

    if arch_column is None:
        for candidate in ("arch_id", "architecture", "model_architecture", "meta_architecture"):
            if candidate in df.columns:
                arch_column = candidate
                break
    if arch_column is not None and arch_column not in df.columns:
        raise ValueError(f"{train_csv} is missing arch column '{arch_column}'.")

    if seen_column is not None and seen_column not in df.columns:
        print(f"Warning: {train_csv} is missing seen column '{seen_column}'; skipping seen filtering.")
        seen_column = None

    df = df.dropna(subset=[path_column, source_column, sys_column]).copy()
    if arch_column is None:
        df["__arch_id"] = "unknown_architecture"
        arch_column = "__arch_id"
        print("Warning: no architecture column provided; using 'unknown_architecture' for all rows.")
    else:
        df[arch_column] = df[arch_column].fillna("unknown_architecture")
    if seen_column is not None:
        df[seen_column] = pd.to_numeric(df[seen_column], errors="coerce").fillna(-1).astype(int)

    records = df[[path_column, source_column, sys_column, arch_column]].copy()
    if seen_column is not None:
        records[seen_column] = df[seen_column]

    path_canon: dict[str, str] = {}
    if path_map is not None:
        if not path_map.exists():
            raise FileNotFoundError(f"Path map not found: {path_map}")
        if path_map.suffix.lower() == ".json":
            mapping = pd.read_json(path_map, typ="series").to_dict()
            path_canon = {str(k): str(v) for k, v in mapping.items()}
        else:
            map_df = pd.read_csv(path_map)
            map_cols = [c for c in map_df.columns if c.lower() in {"path", "path_a", "path_b"}]
            canon_cols = [
                c for c in map_df.columns if c.lower() in {"canonical", "canonical_path", "canonical_id"}
            ]
            if not map_cols or not canon_cols:
                raise ValueError("Path map must have 'path' and 'canonical' columns.")
            path_col = map_cols[0]
            canon_col = canon_cols[0]
            if map_df[path_col].duplicated().any():
                dupe = map_df[map_df[path_col].duplicated(keep=False)][[path_col, canon_col]]
                grouped = dupe.groupby(path_col)[canon_col].nunique()
                if (grouped > 1).any():
                    bad = grouped[grouped > 1].index.tolist()[:5]
                    preview = ", ".join(bad)
                    raise ValueError(f"Path map has one-to-many entries (e.g., {preview}).")
            path_canon = {
                str(p): str(c)
                for p, c in zip(map_df[path_col].tolist(), map_df[canon_col].tolist(), strict=False)
            }

    path_info: dict[str, dict[str, object]] = {}
    arch_to_paths: dict[str, list[str]] = {}
    sys_to_paths: dict[str, list[str]] = {}
    canon_id: dict[str, str] = {}
    for _, row in records.iterrows():
        path = str(row[path_column])
        if path in path_info:
            continue
        src = str(row[source_column])
        sys_id = str(row[sys_column])
        arch_id = str(row[arch_column])
        canon = path_canon.get(path, path)
        info = {
            "source": src,
            "sys_id": sys_id,
            "arch_id": arch_id,
        }
        if seen_column is not None:
            info["seen"] = int(row[seen_column])
        path_info[path] = info
        canon_id[path] = canon
        arch_to_paths.setdefault(arch_id, []).append(path)
        sys_to_paths.setdefault(sys_id, []).append(path)

    anchors: List[Tuple[str, dict[str, object]]] = [
        (path, info) for path, info in path_info.items()
    ]

    all_paths = list(path_info.keys())
    arch_to_sets = {arch: set(paths) for arch, paths in arch_to_paths.items()}
    sys_to_sets = {sys_id: set(paths) for sys_id, paths in sys_to_paths.items()}

    cross_arch_paths: dict[str, list[str]] = {}
    for arch, arch_paths in arch_to_paths.items():
        arch_set = arch_to_sets[arch]
        cross_arch_paths[arch] = [p for p in all_paths if p not in arch_set]

    same_arch_diff_sys: dict[str, list[str]] = {}
    for sys_id, paths in sys_to_paths.items():
        arch_id = path_info[paths[0]]["arch_id"]
        arch_paths = arch_to_paths.get(arch_id, [])
        sys_set = sys_to_sets[sys_id]
        same_arch_diff_sys[sys_id] = [p for p in arch_paths if p not in sys_set]

    neg_paths_by_sys: dict[str, list[str]] = {}
    for sys_id, paths in sys_to_paths.items():
        sys_set = sys_to_sets[sys_id]
        neg_paths_by_sys[sys_id] = [p for p in all_paths if p not in sys_set]

    rng = random.Random(seed)
    pairs: list[PairRow] = []
    cohort_stats: list[dict[str, object]] = []
    global_canon_counts: dict[str, int] = {}

    target_cross = int(round(0.4 * max_neg_candidates_per_anchor))
    target_same_arch = int(round(0.3 * max_neg_candidates_per_anchor))
    target_rest = max_neg_candidates_per_anchor - target_cross - target_same_arch

    def sample_pool(pool: list[str], target: int, exclude_canon: set[str]) -> list[str]:
        if target <= 0:
            return []
        candidates = [
            p
            for p in pool
            if canon_id.get(p, p) not in exclude_canon
            and (
                global_canon_cap is None
                or global_canon_counts.get(canon_id.get(p, p), 0) < global_canon_cap
            )
        ]
        if not candidates:
            return []
        rng.shuffle(candidates)
        selected: list[str] = []
        for p in candidates:
            canon = canon_id.get(p, p)
            if canon in exclude_canon:
                continue
            if global_canon_cap is not None and global_canon_counts.get(canon, 0) >= global_canon_cap:
                continue
            selected.append(p)
            exclude_canon.add(canon)
            global_canon_counts[canon] = global_canon_counts.get(canon, 0) + 1
            if len(selected) >= target:
                break
        return selected

    for anchor_path, info in tqdm(anchors, desc="Sampling candidates", unit="anchor"):
        anchor_src = str(info["source"])
        anchor_sys = str(info["sys_id"])
        anchor_arch = str(info["arch_id"])
        selected: list[str] = []
        exclude_canon: set[str] = {canon_id.get(anchor_path, anchor_path)}

        if pos_per_anchor > 0:
            anchor_canon = canon_id.get(anchor_path, anchor_path)
            pos_candidates = [
                p for p in sys_to_paths.get(anchor_sys, []) if canon_id.get(p, p) != anchor_canon
            ]
            rng.shuffle(pos_candidates)
            for partner_path in pos_candidates[:pos_per_anchor]:
                partner_info = path_info[partner_path]
                pairs.append(
                    {
                        "path_A": anchor_path,
                        "model_name_A": anchor_src,
                        "path_B": partner_path,
                        "model_name_B": str(partner_info["source"]),
                        "same_model": 1,
                        "sys_id_A": anchor_sys,
                        "sys_id_B": str(partner_info["sys_id"]),
                        "arch_id_A": anchor_arch,
                        "arch_id_B": str(partner_info["arch_id"]),
                        **({"seen_A": info["seen"], "seen_B": partner_info["seen"]} if seen_column else {}),
                    }
                )

        cross_pool = cross_arch_paths.get(anchor_arch, [])
        same_arch_pool = same_arch_diff_sys.get(anchor_sys, [])
        rest_pool = neg_paths_by_sys.get(anchor_sys, [])

        counts = {"cross_arch": 0, "same_arch_diff_sys": 0, "rest": 0, "same_arch_unseen": 0}
        picked = sample_pool(cross_pool, target_cross, exclude_canon)
        selected.extend(picked)
        counts["cross_arch"] = len(picked)

        if seen_column is not None:
            unseen_pool = [
                p
                for p in same_arch_pool
                if path_info.get(p, {}).get("seen") == 0
                and canon_id.get(p, p) not in exclude_canon
            ]
            unseen_target = int(round(0.3 * target_same_arch))
            unseen_target = min(len(unseen_pool), max(0, unseen_target))
            picked_unseen = sample_pool(unseen_pool, unseen_target, exclude_canon)
            selected.extend(picked_unseen)
            counts["same_arch_unseen"] = len(picked_unseen)

        remaining_same = target_same_arch - counts["same_arch_unseen"]
        picked = sample_pool(same_arch_pool, remaining_same, exclude_canon)
        selected.extend(picked)
        counts["same_arch_diff_sys"] = counts["same_arch_unseen"] + len(picked)

        picked = sample_pool(rest_pool, target_rest, exclude_canon)
        selected.extend(picked)
        counts["rest"] = len(picked)

        if len(selected) < max_neg_candidates_per_anchor:
            remaining = sample_pool(rest_pool, max_neg_candidates_per_anchor - len(selected), exclude_canon)
            selected.extend(remaining)

        if len(selected) < max_neg_candidates_per_anchor:
            print(
                f"Warning: anchor {anchor_path} has only {len(selected)} negatives "
                f"(target {max_neg_candidates_per_anchor})."
            )
        cohort_stats.append(
            {
                "path_A": anchor_path,
                "sys_id_A": anchor_sys,
                "arch_id_A": anchor_arch,
                "n_cross_arch": counts["cross_arch"],
                "n_same_arch_diff_sys": counts["same_arch_diff_sys"],
                "n_rest": counts["rest"],
                "n_total": len(selected),
                "n_same_arch_unseen": counts["same_arch_unseen"],
                "frac_same_arch_unseen": (
                    counts["same_arch_unseen"] / counts["same_arch_diff_sys"]
                    if counts["same_arch_diff_sys"] > 0
                    else 0.0
                ),
            }
        )

        for partner_path in selected:
            partner_info = path_info[partner_path]
            pairs.append(
                {
                    "path_A": anchor_path,
                    "model_name_A": anchor_src,
                    "path_B": partner_path,
                    "model_name_B": str(partner_info["source"]),
                    "same_model": 0,
                    "sys_id_A": anchor_sys,
                    "sys_id_B": str(partner_info["sys_id"]),
                    "arch_id_A": anchor_arch,
                    "arch_id_B": str(partner_info["arch_id"]),
                    **({"seen_A": info["seen"], "seen_B": partner_info["seen"]} if seen_column else {}),
                }
            )

    rng.shuffle(pairs)
    if cohort_stats:
        stats_df = pd.DataFrame(cohort_stats)
        if cohort_stats_out is not None:
            cohort_stats_out.parent.mkdir(parents=True, exist_ok=True)
            stats_df.to_csv(cohort_stats_out, index=False)
            print(f"Wrote cohort stats to {cohort_stats_out}")
        desc = stats_df[["n_cross_arch", "n_same_arch_diff_sys", "n_rest", "n_total"]].describe()
        print("Cohort counts summary:")
        print(desc.to_string())
        if "frac_same_arch_unseen" in stats_df.columns:
            unseen_desc = stats_df["frac_same_arch_unseen"].describe()
            print("Same-arch unseen fraction summary:")
            print(unseen_desc.to_string())
    if global_canon_counts:
        reuse = [c for c in global_canon_counts.values() if c > 1]
        if reuse:
            print(
                "Canonical reuse summary: "
                f"unique={len(global_canon_counts):,}, reused={len(reuse):,}, max={max(reuse)}"
            )
    return pairs


def write_pairs_csv(pairs: List[Pair] | List[PairRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not pairs:
        raise ValueError("No pairs to write.")
    if isinstance(pairs[0], dict):
        df = pd.DataFrame(pairs)
    else:
        df = pd.DataFrame(
            pairs,
            columns=["path_A", "model_name_A", "path_B", "model_name_B", "same_model"],
        )
    df.to_csv(output_path, index=False)
    print(f"Wrote {len(df):,} candidate pairs to {output_path}")


def _build_eval_namespace(args: argparse.Namespace) -> SimpleNamespace:
    """
    Build a minimal args namespace compatible with common.build_model().
    """
    return SimpleNamespace(
        extractor=args.extractor,
        processor=args.processor,
        classifier=args.classifier,
        kernel=getattr(args, "kernel", None),
        n_components=getattr(args, "n_components", None),
        covariance_type=getattr(args, "covariance_type", None),
    )


def _pool_embeddings(emb_tensor: torch.Tensor) -> torch.Tensor:
    """
    Pool extractor outputs over layers and time to get a fixed utterance embedding.
    Expects shape [layers, batch, time, feat].
    """
    if emb_tensor.dim() != 4:
        raise ValueError(f"Expected 4D extractor output, got shape {tuple(emb_tensor.shape)}")
    return emb_tensor.mean(dim=(0, 2))


def _rel_protocol_path(pairs_csv: Path, data_root: Path) -> str:
    """
    Make the protocol path relative to data_root so MLAADDataset can resolve it
    while still letting audio paths remain rooted at data_root.
    """
    return os.path.relpath(pairs_csv, data_root)


def score_pairs(
    pairs_csv: Path,
    data_root: Path,
    checkpoint: Path,
    extractor: str,
    processor: str,
    classifier: str,
    batch_size: int,
    num_workers: int,
    amp_eval: bool,
    amp_dtype: str,
    device: str | None = None,
    output_embeddings: Path | None = None,
    embeddings: Path | None = None,
    embedding_cache: bool = True,
) -> pd.DataFrame:
    """
    Load the model checkpoint and score each pair. Optionally dump per-utterance embeddings.
    """
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16}
    amp_torch_dtype = dtype_map.get(amp_dtype, torch.bfloat16)

    args_ns = SimpleNamespace(
        extractor=extractor,
        processor=processor,
        classifier=classifier,
        kernel=None,
        n_components=None,
        covariance_type=None,
    )
    model, trainer = build_model(args_ns)
    if hasattr(trainer, "set_amp_eval"):
        trainer.set_amp_eval(amp_eval, dtype=amp_torch_dtype)
    trainer.load_model(str(checkpoint))
    model.eval()
    model.to(resolved_device)

    if embeddings is not None:
        if output_embeddings is not None:
            print("Note: --output-embeddings is ignored when --embeddings is provided.")
        if not hasattr(model, "forward_from_embeddings"):
            raise RuntimeError("Model does not support forward_from_embeddings; remove --embeddings.")
        scored_df = pd.read_csv(pairs_csv)
        missing_cols = {"path_A", "path_B", "same_model"} - set(scored_df.columns)
        if missing_cols:
            raise ValueError(
                f"{pairs_csv} is missing required columns: {', '.join(sorted(missing_cols))}"
            )
        utt_ids, emb_matrix = load_embedding_table(embeddings)
        if emb_matrix.ndim != 2:
            raise ValueError(f"Expected 2D embedding matrix, got shape {emb_matrix.shape}.")
        emb_matrix = emb_matrix.astype(np.float32, copy=False)
        emb_index = {str(u): i for i, u in enumerate(utt_ids)}
        paths_a = scored_df["path_A"].astype(str).tolist()
        paths_b = scored_df["path_B"].astype(str).tolist()
        needed = set(paths_a) | set(paths_b)
        missing = [p for p in needed if p not in emb_index]
        if missing:
            preview = ", ".join(missing[:5])
            raise KeyError(
                f"{len(missing)} embeddings missing (e.g., {preview}). "
                "Ensure --embeddings covers path_A/path_B values."
            )

        idx_a = np.fromiter((emb_index[p] for p in paths_a), dtype=np.int64, count=len(paths_a))
        idx_b = np.fromiter((emb_index[p] for p in paths_b), dtype=np.int64, count=len(paths_b))
        labels_np = scored_df["same_model"].astype(int).to_numpy(dtype=np.int64)
        pair_ids = [f"{a}|{b}" for a, b in zip(paths_a, paths_b, strict=False)]

        emb_tensor = torch.from_numpy(emb_matrix)
        scores_same: List[float] = []
        scores_diff: List[float] = []
        labels: List[int] = []

        autocast_ctx = (
            torch.autocast(device_type=resolved_device.split(":")[0], dtype=amp_torch_dtype)
            if amp_eval and resolved_device.startswith("cuda")
            else nullcontext()
        )
        import inspect

        forward_params = inspect.signature(model.forward_from_embeddings).parameters
        use_label = "label" in forward_params

        with torch.no_grad():
            for start in tqdm(
                range(0, len(scored_df), batch_size),
                desc="Scoring pairs (embeddings)",
            ):
                end = start + batch_size
                batch_idx_a = torch.from_numpy(idx_a[start:end])
                batch_idx_b = torch.from_numpy(idx_b[start:end])
                emb_gt = emb_tensor.index_select(0, batch_idx_a).to(resolved_device)
                emb_test = emb_tensor.index_select(0, batch_idx_b).to(resolved_device)
                label_batch = torch.from_numpy(labels_np[start:end]).to(resolved_device)
                with autocast_ctx:
                    if use_label:
                        logits, probs = model.forward_from_embeddings(
                            emb_gt, emb_test, label=label_batch
                        )
                    else:
                        logits, probs = model.forward_from_embeddings(emb_gt, emb_test)
                probs_cpu = probs.detach().cpu()
                scores_diff.extend(probs_cpu[:, 0].tolist())
                if probs_cpu.shape[1] > 1:
                    scores_same.extend(probs_cpu[:, 1].tolist())
                else:
                    scores_same.extend([1.0 - p for p in probs_cpu[:, 0].tolist()])
                labels.extend(labels_np[start:end].tolist())

        if len(scores_same) != len(scored_df):
            raise RuntimeError(
                f"Scored {len(scores_same)} pairs but CSV has {len(scored_df)} entries; "
                "check batch sizing."
            )

        scored_df["score_diff"] = scores_diff
        scored_df["score_same"] = scores_same
        scored_df["pair_id"] = pair_ids
        return scored_df

    use_embedding_cache = (
        embedding_cache
        and hasattr(model, "forward_from_embeddings")
        and hasattr(model, "extractor")
        and hasattr(model, "feature_processor")
    )
    if output_embeddings is not None and use_embedding_cache:
        print(
            "Note: --output-embeddings is ignored when --embedding-cache is enabled. "
            "Use --no-embedding-cache to compute pooled extractor embeddings."
        )
    embedding_cache_obj = (
        EmbeddingCache(model.extractor, model.feature_processor, resolved_device)
        if use_embedding_cache
        else None
    )
    if use_embedding_cache:
        import inspect

        forward_params = inspect.signature(model.forward_from_embeddings).parameters
        forward_from_embeddings_uses_label = "label" in forward_params
    else:
        forward_from_embeddings_uses_label = False

    protocol_rel = _rel_protocol_path(pairs_csv, data_root)
    dataset = MLAADDataset_pair(
        root_dir=str(data_root),
        protocol_file_name=protocol_rel,
        variant="eval",
    )
    loader_kwargs = {
        "batch_size": batch_size,
        "collate_fn": custom_pair_batch_create,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": resolved_device.startswith("cuda"),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    dataloader = DataLoader(dataset, **loader_kwargs)

    scores_same: List[float] = []
    scores_diff: List[float] = []
    labels: List[int] = []
    pair_ids: List[str] = []
    emb_map: dict[str, np.ndarray] = {}

    autocast_ctx = (
        torch.autocast(device_type=resolved_device.split(":")[0], dtype=amp_torch_dtype)
        if amp_eval and resolved_device.startswith("cuda")
        else nullcontext()
    )

    with torch.no_grad():
        for pair_id, gt, test, label in tqdm(dataloader, desc="Scoring pairs"):
            if use_embedding_cache:
                keys_gt, keys_test = build_pair_keys(pair_id, gt, test)
                emb_gt = embedding_cache_obj.get_embeddings(gt, keys_gt, autocast_ctx)
                emb_test = embedding_cache_obj.get_embeddings(test, keys_test, autocast_ctx)
                label = label.to(resolved_device)
                with autocast_ctx:
                    if forward_from_embeddings_uses_label:
                        logits, probs = model.forward_from_embeddings(emb_gt, emb_test, label=label)
                    else:
                        logits, probs = model.forward_from_embeddings(emb_gt, emb_test)
            else:
                gt = gt.to(resolved_device)
                test = test.to(resolved_device)
                with autocast_ctx:
                    logits, probs = model(gt, test)
                    if output_embeddings is not None:
                        # Cache per-utterance pooled embeddings
                        pooled_gt = _pool_embeddings(model.extractor.extract_features(gt)).detach()
                        pooled_test = _pool_embeddings(model.extractor.extract_features(test)).detach()
            probs_cpu = probs.detach().cpu()
            scores_diff.extend(probs_cpu[:, 0].tolist())
            if probs_cpu.shape[1] > 1:
                scores_same.extend(probs_cpu[:, 1].tolist())
            else:
                scores_same.extend([1.0 - p for p in probs_cpu[:, 0].tolist()])
            labels.extend([int(x) for x in label.tolist()])
            pair_ids.extend(pair_id)

            if output_embeddings is not None and not use_embedding_cache:
                for idx, pid in enumerate(pair_id):
                    try:
                        path_a, path_b = pid.split("|", 1)
                    except ValueError:
                        continue
                    if path_a not in emb_map:
                        emb_map[path_a] = pooled_gt[idx].detach().cpu().float().numpy()
                    if path_b not in emb_map:
                        emb_map[path_b] = pooled_test[idx].detach().cpu().float().numpy()

    if len(scores_same) != len(dataset):
        raise RuntimeError(
            f"Scored {len(scores_same)} pairs but dataset has {len(dataset)} entries; "
            "check dataloader ordering."
        )

    if output_embeddings is not None and emb_map:
        utt_ids = sorted(emb_map)
        emb_matrix = np.stack([emb_map[u] for u in utt_ids], axis=0).astype(np.float32)
        output_embeddings.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_embeddings, embeddings=emb_matrix, utt_ids=np.array(utt_ids))
        print(f"Wrote utterance embeddings for {len(utt_ids):,} items to {output_embeddings}")

    scored_df = dataset.protocol_df.copy()
    scored_df["score_diff"] = scores_diff
    scored_df["score_same"] = scores_same
    scored_df["pair_id"] = pair_ids
    return scored_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hard-negative mining helpers (sampling and scoring).")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Sampling
    sample_p = subparsers.add_parser("sample", help="Sample balanced negative pairs per anchor.")
    sample_p.add_argument("--train-csv", type=Path, required=True, help="Single-utterance MLAAD train CSV.")
    sample_p.add_argument("--path-column", default="path", help="Column with relative audio paths.")
    sample_p.add_argument("--source-column", default="model_name", help="Column with source/model labels.")
    sample_p.add_argument(
        "--sys-column",
        default=None,
        help="Column with system IDs (defaults to --source-column).",
    )
    sample_p.add_argument(
        "--arch-column",
        default=None,
        help="Column with architecture IDs (defaults to arch_id/architecture/model_architecture if present).",
    )
    sample_p.add_argument(
        "--seen-column",
        default=None,
        help="Optional column with seen/unseen flags to pass through (int).",
    )
    sample_p.add_argument(
        "--path-map",
        type=Path,
        default=None,
        help="Optional CSV/JSON mapping of path -> canonical ID to de-duplicate aliases.",
    )
    sample_p.add_argument(
        "--cohort-stats-out",
        type=Path,
        default=None,
        help="Optional CSV path to write per-anchor cohort counts.",
    )
    sample_p.add_argument(
        "--global-canon-cap",
        type=int,
        default=None,
        help="Optional cap on how many times a canonical path can be reused across anchors.",
    )
    sample_p.add_argument(
        "--pos-per-anchor",
        type=int,
        default=0,
        help="Optional number of positive partners per anchor (defaults to 0 for negative-only mining).",
    )
    sample_p.add_argument(
        "--max-neg-candidates-per-anchor",
        type=int,
        default=100,
        help="Upper bound on negative candidates per anchor (balanced across other models).",
    )
    sample_p.add_argument("--seed", type=int, default=42, help="Random seed.")
    sample_p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Where to write candidate pairs CSV (pair protocol format).",
    )

    # Scoring
    score_p = subparsers.add_parser("score", help="Score candidate pairs with a trained model checkpoint.")
    score_p.add_argument("--pairs-csv", type=Path, required=True, help="Candidate pairs CSV to score.")
    score_p.add_argument("--data-root", type=Path, required=True, help="Root directory containing MLAAD audio.")
    score_p.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint to load for scoring.")
    score_p.add_argument("--extractor", required=True, help="Extractor name (same as training).")
    score_p.add_argument("--processor", required=True, help="Processor/pooling name (same as training).")
    score_p.add_argument("--classifier", required=True, help="Classifier name (same as training).")
    score_p.add_argument("--batch-size", type=int, default=8, help="Eval batch size.")
    score_p.add_argument("--num-workers", type=int, default=4, help="Dataloader workers.")
    score_p.add_argument("--amp-eval", action=argparse.BooleanOptionalAction, default=False, help="Enable AMP.")
    score_p.add_argument(
        "--amp-dtype",
        choices=["bf16", "fp16"],
        default="bf16",
        help="Autocast dtype when AMP is enabled.",
    )
    score_p.add_argument(
        "--output-embeddings",
        type=Path,
        default=None,
        help="Optional path to save per-utterance embeddings (npz with embeddings + utt_ids).",
    )
    score_p.add_argument(
        "--embeddings",
        type=Path,
        default=None,
        help="Optional embeddings npz to score pairs without re-extracting audio features.",
    )
    score_p.add_argument(
        "--embedding-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cache per-utterance embeddings during scoring to avoid recomputing them.",
    )
    score_p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Force device (e.g., cuda, cuda:1, cpu). Defaults to CUDA if available.",
    )
    score_p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output CSV path (defaults to pairs-csv with _scored suffix).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "sample":
        pairs = sample_pairs(
            train_csv=args.train_csv,
            path_column=args.path_column,
            source_column=args.source_column,
            sys_column=args.sys_column,
            arch_column=args.arch_column,
            seen_column=args.seen_column,
            path_map=args.path_map,
            cohort_stats_out=args.cohort_stats_out,
            global_canon_cap=args.global_canon_cap,
            pos_per_anchor=args.pos_per_anchor,
            max_neg_candidates_per_anchor=args.max_neg_candidates_per_anchor,
            seed=args.seed,
        )
        write_pairs_csv(pairs, args.output)
    elif args.command == "score":
        out_path = (
            args.output
            if args.output
            else args.pairs_csv.with_name(args.pairs_csv.stem + "_scored.csv")
        )
        scored_df = score_pairs(
            pairs_csv=args.pairs_csv,
            data_root=args.data_root,
            checkpoint=args.checkpoint,
            extractor=args.extractor,
            processor=args.processor,
            classifier=args.classifier,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            amp_eval=args.amp_eval,
            amp_dtype=args.amp_dtype,
            device=args.device,
            output_embeddings=args.output_embeddings,
            embeddings=args.embeddings,
            embedding_cache=args.embedding_cache,
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        scored_df.to_csv(out_path, index=False)
        print(f"Wrote scored pairs to {out_path} (n={len(scored_df):,})")
    else:
        raise ValueError(f"Unknown command {args.command}")


if __name__ == "__main__":
    main()
