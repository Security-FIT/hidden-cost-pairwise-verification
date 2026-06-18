#!/usr/bin/env python3
"""
Generate MLAAD train pair manifests.

Supported modes:
  - minimal: Stage-1 minimal regime (one random partner per anchor, ~50/50).
  - intermediate: Stage-1 intermediate regime (bounded positives/negatives per anchor).
  - curated: Stage-1 curated regime (targeted positives/negatives with per-source cap).
  - curated_balanced: Width-aware curated variant (caps per-source and per-utterance depth).
  - random: balanced random sampling (B/2 target, B/2 non-target) without materializing the full pool.
  - rival: negative-only sampling using a precomputed rival mapping (confusion-guided).
  - directional: coverage-based Stage-3 sampler using scored candidates and utterance embeddings
                 (directionally diverse hard negatives per anchor).
  - hardmined: consume a scored candidate CSV and keep the hardest negatives per anchor (negative-only).
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from bisect import bisect_right
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Hashable, Mapping, Sequence

import numpy as np
import pandas as pd

Pair = Tuple[str, str, str, str, int]
RivalPair = Tuple[str, str, str, str, int, int]
Stage1Rows = List[Tuple[str, str]]
STAGE1_REGIME_OFFSETS = {"minimal": 0, "intermediate": 101, "curated": 202, "curated_balanced": 303}
CURATED_BALANCED_DEBUG = False  # Enable to print per-anchor stats for the balanced curated regime


def load_protocol(csv_path: Path, path_column: str, source_column: str) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Protocol file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    missing_cols = {path_column, source_column} - set(df.columns)
    if missing_cols:
        raise ValueError(f"{csv_path} is missing required columns: {', '.join(sorted(missing_cols))}")
    df = df.dropna(subset=[path_column, source_column])

    rows: Stage1Rows = [(str(row[path_column]), str(row[source_column])) for _, row in df.iterrows()]
    rows.sort(key=lambda item: (item[0], item[1]))

    source_to_paths: Dict[str, List[str]] = {}
    path_to_source: Dict[str, str] = {}
    for path, source in rows:
        source_to_paths.setdefault(source, []).append(path)
        path_to_source[path] = source
    for paths in source_to_paths.values():
        paths.sort()
    return source_to_paths, path_to_source


def build_negative_index_map(rows: Stage1Rows) -> Dict[str, List[int]]:
    source_to_indices: Dict[str, List[int]] = {}
    for _, source in rows:
        source_to_indices.setdefault(source, [])
    for source in list(source_to_indices):
        indices = [idx for idx, (_, src) in enumerate(rows) if src != source]
        source_to_indices[source] = indices
    return source_to_indices


def pair_index_to_coords(num_items: int, pair_index: int) -> Tuple[int, int]:
    """
    Map a zero-based pair index to coordinates (i, j) for combinations where i < j.
    Ordering follows (0,1), (0,2), ..., (0,n-1), (1,2), ...
    """
    if num_items < 2:
        raise ValueError("pair_index_to_coords requires at least two items.")
    max_pairs = num_items * (num_items - 1) // 2
    if pair_index < 0 or pair_index >= max_pairs:
        raise IndexError(f"Pair index {pair_index} out of range for {num_items} items.")
    first = 0
    remaining = pair_index
    while remaining >= num_items - first - 1:
        remaining -= num_items - first - 1
        first += 1
    second = first + 1 + remaining
    return first, second


def build_positive_blocks(
    source_to_paths: Dict[str, List[str]]
) -> Tuple[List[Tuple[str, List[str]]], List[int], int]:
    blocks: List[Tuple[str, List[str]]] = []
    cumulative: List[int] = []
    running = 0
    for source in sorted(source_to_paths):
        paths = source_to_paths[source]
        combos = len(paths) * (len(paths) - 1) // 2
        if combos <= 0:
            continue
        blocks.append((source, paths))
        running += combos
        cumulative.append(running)
    return blocks, cumulative, running


def build_negative_blocks(
    source_to_paths: Dict[str, List[str]]
) -> Tuple[List[Tuple[str, str, List[str], List[str], int]], List[int], int]:
    blocks: List[Tuple[str, str, List[str], List[str], int]] = []
    cumulative: List[int] = []
    running = 0
    sources = sorted(source_to_paths)
    for i, src_a in enumerate(sources):
        paths_a = source_to_paths[src_a]
        len_a = len(paths_a)
        if len_a == 0:
            continue
        for src_b in sources[i + 1 :]:
            paths_b = source_to_paths[src_b]
            len_b = len(paths_b)
            if len_b == 0:
                continue
            block_size = len_a * len_b
            blocks.append((src_a, src_b, paths_a, paths_b, len_b))
            running += block_size
            cumulative.append(running)
    return blocks, cumulative, running


def decode_positive_index(
    index: int,
    blocks: List[Tuple[str, List[str]]],
    cumulative: List[int],
) -> Tuple[str, str, str, str]:
    block_idx = bisect_right(cumulative, index)
    if block_idx >= len(blocks):
        raise IndexError(f"Positive index {index} out of range.")
    start = 0 if block_idx == 0 else cumulative[block_idx - 1]
    offset = index - start
    source, paths = blocks[block_idx]
    i, j = pair_index_to_coords(len(paths), offset)
    return paths[i], source, paths[j], source


def decode_negative_index(
    index: int,
    blocks: List[Tuple[str, str, List[str], List[str], int]],
    cumulative: List[int],
) -> Tuple[str, str, str, str]:
    block_idx = bisect_right(cumulative, index)
    if block_idx >= len(blocks):
        raise IndexError(f"Negative index {index} out of range.")
    start = 0 if block_idx == 0 else cumulative[block_idx - 1]
    offset = index - start
    src_a, src_b, paths_a, paths_b, len_b = blocks[block_idx]
    a_idx = offset // len_b
    b_idx = offset % len_b
    return paths_a[a_idx], src_a, paths_b[b_idx], src_b


def build_ordered_paths(
    source_to_paths: Dict[str, List[str]], rng: random.Random
) -> Dict[str, List[str]]:
    ordered: Dict[str, List[str]] = {}
    for source in sorted(source_to_paths):
        paths = source_to_paths[source][:]
        rng.shuffle(paths)
        ordered[source] = paths
    return ordered


def sample_random_balanced_pairs(
    source_to_paths: Dict[str, List[str]],
    total_pairs: int,
    rng_seed: int,
) -> Tuple[List[Pair], int, int]:
    if total_pairs <= 0:
        return [], 0, 0
    rng = random.Random(rng_seed)
    pos_blocks, pos_cumulative, pos_total = build_positive_blocks(source_to_paths)
    neg_blocks, neg_cumulative, neg_total = build_negative_blocks(source_to_paths)

    desired_half = total_pairs // 2
    balanced_half = min(desired_half, pos_total, neg_total)
    if balanced_half == 0:
        return [], 0, 0
    if pos_total < desired_half:
        print(
            f"  Warning: requested {desired_half:,} positive pairs but only {pos_total:,} available; "
            f"using {balanced_half:,} to keep balance."
        )
    if neg_total < desired_half:
        print(
            f"  Warning: requested {desired_half:,} negative pairs but only {neg_total:,} available; "
            f"using {balanced_half:,} to keep balance."
        )

    pos_indices = rng.sample(range(pos_total), balanced_half)
    neg_indices = rng.sample(range(neg_total), balanced_half)

    pairs: List[Pair] = []
    for idx in pos_indices:
        path_a, source_a, path_b, source_b = decode_positive_index(idx, pos_blocks, pos_cumulative)
        pairs.append((path_a, source_a, path_b, source_b, 1))
    for idx in neg_indices:
        path_a, source_a, path_b, source_b = decode_negative_index(idx, neg_blocks, neg_cumulative)
        pairs.append((path_a, source_a, path_b, source_b, 0))
    rng.shuffle(pairs)
    pos_count = balanced_half
    neg_count = balanced_half
    return pairs, pos_count, neg_count


def _parse_bool(value: object) -> bool:
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "t"}


def load_rival_map(
    csv_path: Path,
    source_column: str,
    rival_column: str,
    forced_column: str | None = None,
) -> Tuple[Dict[str, List[str]], Dict[Tuple[str, str], bool]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Rivals file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    missing_cols = {source_column, rival_column} - set(df.columns)
    if missing_cols:
        raise ValueError(f"{csv_path} missing required columns: {', '.join(sorted(missing_cols))}")
    df = df.dropna(subset=[source_column, rival_column])
    mapping: Dict[str, List[str]] = {}
    forced_lookup: Dict[Tuple[str, str], bool] = {}
    forced_col = forced_column if forced_column and forced_column in df.columns else None
    if forced_column and forced_col is None:
        print(f"Warning: rivals CSV missing forced column '{forced_column}'.")

    for _, row in df.iterrows():
        source = str(row[source_column])
        rival = str(row[rival_column])
        if source == rival:
            continue
        mapping.setdefault(source, [])
        if rival not in mapping[source]:
            mapping[source].append(rival)
        if forced_col:
            forced_lookup[(source, rival)] = _parse_bool(row.get(forced_col))
    for rivals in mapping.values():
        rivals.sort()
    return mapping, forced_lookup


def sample_rival_pairs(
    rows: Stage1Rows,
    source_to_paths: Dict[str, List[str]],
    rivals: Dict[str, List[str]],
    forced_lookup: Dict[Tuple[str, str], bool],
    rng: random.Random,
    neg_per_anchor: int,
    max_pairs: int,
    max_partner_uses: int,
) -> Tuple[List[RivalPair], int, int]:
    eligible = [
        (path, source)
        for path, source in rows
        if source in rivals
        and any(
            rival in source_to_paths and source_to_paths[rival]
            for rival in rivals[source]
        )
    ]
    if max_pairs > 0 and len(eligible) * max(1, neg_per_anchor) > max_pairs:
        keep = max_pairs // max(1, neg_per_anchor)
        if keep <= 0:
            keep = min(len(eligible), max_pairs)
            eligible = rng.sample(eligible, keep)
        else:
            eligible = rng.sample(eligible, min(len(eligible), keep))

    pairs: List[Pair] = []
    partner_counts: Dict[str, int] = defaultdict(int)
    neg_per_anchor = max(0, neg_per_anchor)
    for path_a, source_a in eligible:
        rival_sources = [
            rival
            for rival in rivals.get(source_a, [])
            if rival in source_to_paths and source_to_paths[rival]
        ]
        if not rival_sources:
            continue
        for _ in range(neg_per_anchor):
            weights = [len(source_to_paths[rival]) for rival in rival_sources]
            if sum(weights) <= 0:
                rival_source = rng.choice(rival_sources)
            else:
                rival_source = rng.choices(rival_sources, weights=weights, k=1)[0]
            partner_candidates = source_to_paths.get(rival_source)
            if not partner_candidates:
                continue
            if max_partner_uses > 0:
                eligible_partners = [p for p in partner_candidates if partner_counts[p] < max_partner_uses]
                if eligible_partners:
                    partner_candidates = eligible_partners
            min_used = min(partner_counts[p] for p in partner_candidates)
            pool = [p for p in partner_candidates if partner_counts[p] == min_used]
            partner_path = rng.choice(pool)
            partner_counts[partner_path] += 1
            forced_flag = int(forced_lookup.get((source_a, rival_source), False))
            pairs.append((path_a, source_a, partner_path, rival_source, 0, forced_flag))
    rng.shuffle(pairs)
    pos_count = 0
    neg_count = len(pairs)
    if partner_counts:
        counts = sorted(partner_counts.values())
        median = counts[len(counts) // 2] if len(counts) % 2 else 0.5 * (counts[len(counts) // 2 - 1] + counts[len(counts) // 2])
        print(
            f"[rival] partner usage: min={counts[0]}, median={median:.1f}, max={counts[-1]}, "
            f"unique={len(counts)}"
        )
    return pairs, pos_count, neg_count


def generate_minimal_trials(
    rows: Stage1Rows,
    source_to_paths: Dict[str, List[str]],
    neg_index_map: Dict[str, List[int]],
    rng: random.Random,
) -> List[Pair]:
    pairs: List[Pair] = []
    for path_a, source_a in rows:
        same_candidates = [p for p in source_to_paths[source_a] if p != path_a]
        neg_indices = neg_index_map.get(source_a, [])
        label = 1 if rng.random() < 0.5 else 0
        partner_path: str | None = None
        partner_source: str | None = None
        if label == 1 and same_candidates:
            partner_path = rng.choice(same_candidates)
            partner_source = source_a
        elif label == 0 and neg_indices:
            partner_idx = rng.choice(neg_indices)
            partner_path, partner_source = rows[partner_idx]
        elif same_candidates:
            partner_path = rng.choice(same_candidates)
            partner_source = source_a
            label = 1
        elif neg_indices:
            partner_idx = rng.choice(neg_indices)
            partner_path, partner_source = rows[partner_idx]
            label = 0
        else:
            continue
        if partner_source is None or partner_path is None or partner_path == path_a:
            continue
        pairs.append((path_a, source_a, partner_path, partner_source, label))
    rng.shuffle(pairs)
    return pairs


def generate_intermediate_trials(
    rows: Stage1Rows,
    source_to_paths: Dict[str, List[str]],
    neg_index_map: Dict[str, List[int]],
    rng: random.Random,
    max_same: int,
    max_diff: int,
) -> List[Pair]:
    pairs: List[Pair] = []
    seen: set[Tuple[Tuple[str, str], Tuple[str, str]]] = set()
    max_same = max(0, max_same)
    max_diff = max(0, max_diff)
    for path_a, source_a in rows:
        same_candidates = [p for p in source_to_paths[source_a] if p != path_a]
        if max_same > 0 and same_candidates:
            keep = min(len(same_candidates), max_same)
            for partner_path in rng.sample(same_candidates, keep):
                key = tuple(sorted(((path_a, source_a), (partner_path, source_a))))
                if key in seen:
                    continue
                seen.add(key)
                pairs.append((path_a, source_a, partner_path, source_a, 1))
        neg_indices = neg_index_map.get(source_a, [])
        if max_diff > 0 and neg_indices:
            keep = min(len(neg_indices), max_diff)
            for partner_idx in rng.sample(neg_indices, keep):
                partner_path, partner_source = rows[partner_idx]
                key = tuple(sorted(((path_a, source_a), (partner_path, partner_source))))
                if key in seen:
                    continue
                seen.add(key)
                pairs.append((path_a, source_a, partner_path, partner_source, 0))
    rng.shuffle(pairs)
    return pairs


def auto_curated_pos_cap(source_to_paths: Dict[str, List[str]], target_pos: int) -> int:
    counts = [len(paths) * (len(paths) - 1) // 2 for paths in source_to_paths.values() if len(paths) >= 2]
    if not counts:
        return 0
    total_available = sum(counts)
    max_count = max(counts)
    if target_pos <= 0 or total_available <= target_pos:
        return max_count

    low = 1
    high = max_count
    best = max_count
    while low <= high:
        mid = (low + high) // 2
        capped_total = sum(min(count, mid) for count in counts)
        if capped_total >= target_pos:
            best = mid
            high = mid - 1
        else:
            low = mid + 1
    return best


def collect_capped_positive_pairs(
    selected_paths: Dict[str, List[str]],
    per_source_cap: int,
    rng_desc: str,
) -> List[Pair]:
    cap = per_source_cap if per_source_cap > 0 else None
    per_source_limits: Dict[str, int] = {}
    total = 0
    for source, paths in selected_paths.items():
        count = len(paths) * (len(paths) - 1) // 2
        if count <= 0:
            per_source_limits[source] = 0
            continue
        limit = count if cap is None else min(cap, count)
        per_source_limits[source] = limit
        total += limit
    positives: List[Pair] = []
    emitted_per_source: Dict[str, int] = {}
    for source in sorted(selected_paths):
        limit = per_source_limits.get(source, 0)
        if limit <= 0:
            continue
        emitted = 0
        paths = selected_paths[source]
        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                positives.append((paths[i], source, paths[j], source, 1))
                emitted += 1
                if emitted >= limit:
                    break
            if emitted >= limit:
                break
        emitted_per_source[source] = emitted
    return positives


def allocate_negative_quota(
    selected_paths: Dict[str, List[str]],
    target: int,
) -> Dict[Tuple[str, str], int]:
    allocations: Dict[Tuple[str, str], int] = {}
    if target <= 0:
        return allocations
    sources = sorted(selected_paths)
    pair_info: List[Tuple[Tuple[str, str], int]] = []
    total_available = 0
    for i, src_a in enumerate(sources):
        paths_a = selected_paths[src_a]
        for src_b in sources[i + 1 :]:
            paths_b = selected_paths[src_b]
            count = len(paths_a) * len(paths_b)
            if count == 0:
                continue
            pair_key = (src_a, src_b)
            pair_info.append((pair_key, count))
            total_available += count

    if total_available == 0:
        return allocations

    target = min(target, total_available)
    fractional: List[Tuple[float, Tuple[str, str]]] = []
    assigned = 0
    for key, count in pair_info:
        desired = (count / total_available) * target
        base = min(count, int(desired))
        allocations[key] = base
        fractional.append((desired - base, key))
        assigned += base

    remainder = target - assigned
    fractional.sort(key=lambda x: x[0], reverse=True)
    idx = 0
    while remainder > 0 and fractional:
        frac, key = fractional[idx % len(fractional)]
        available = dict(pair_info)[key]
        if allocations[key] < available:
            allocations[key] += 1
            remainder -= 1
        idx += 1
        if idx > len(fractional) * 5:
            break

    return allocations


def generate_negatives_from_quota(
    selected_paths: Dict[str, List[str]],
    quota: Dict[Tuple[str, str], int],
) -> List[Pair]:
    negatives: List[Pair] = []
    for (src_a, src_b), need in quota.items():
        if need <= 0:
            continue
        paths_a = selected_paths[src_a]
        paths_b = selected_paths[src_b]
        count = 0
        for path_a in paths_a:
            for path_b in paths_b:
                negatives.append((path_a, src_a, path_b, src_b, 0))
                count += 1
                if count >= need:
                    break
            if count >= need:
                break
    return negatives


def generate_curated_trials(
    source_to_paths: Dict[str, List[str]],
    rng: random.Random,
    target_pos: int,
    target_neg: int,
    per_source_cap: int,
) -> Tuple[List[Pair], int]:
    ordered_paths = build_ordered_paths(source_to_paths, rng)
    cap = per_source_cap
    if cap <= 0:
        cap = auto_curated_pos_cap(ordered_paths, target_pos)
    positives = collect_capped_positive_pairs(
        ordered_paths,
        cap,
        rng_desc=f"curated positives (cap {cap if cap > 0 else 'all'})",
    )
    if target_pos > 0 and len(positives) > target_pos:
        positives = rng.sample(positives, target_pos)

    negatives: List[Pair] = []
    if target_neg > 0:
        quota = allocate_negative_quota(ordered_paths, target_neg)
        negatives = generate_negatives_from_quota(ordered_paths, quota)
        if len(negatives) > target_neg:
            negatives = rng.sample(negatives, target_neg)

    pairs = positives + negatives
    rng.shuffle(pairs)
    return pairs, cap


def _print_curated_balanced_debug(pairs: List[Pair]) -> None:
    """Optional diagnostic to verify anchor width for curated_balanced."""
    anchor_counts = Counter(pair[0] for pair in pairs)
    if not anchor_counts:
        print("[curated_balanced] No anchors available.")
        return
    counts = sorted(anchor_counts.values())
    mean = sum(counts) / len(counts)
    median = counts[len(counts) // 2] if len(counts) % 2 else 0.5 * (counts[len(counts) // 2 - 1] + counts[len(counts) // 2])
    max_count = counts[-1]
    print(
        f"[curated_balanced] trials={len(pairs)} anchors={len(anchor_counts)} "
        f"per-anchor mean={mean:.2f} median={median:.2f} max={max_count}"
    )


def _pick_least_used(candidates: List[str], trial_counts: Mapping[str, int], rng: random.Random) -> str | None:
    if not candidates:
        return None
    min_count = min(trial_counts[c] for c in candidates)
    pool = [c for c in candidates if trial_counts[c] == min_count]
    return rng.choice(pool)


def generate_curated_balanced_trials(
    source_to_paths: Dict[str, List[str]],
    rng: random.Random,
    target_pos: int,
    target_neg: int,
    per_source_cap: int,
    max_trials_per_utt: int | None = 0,
) -> Tuple[List[Pair], int]:
    """
    New curated regime that is class balanced at the source level and also
    preserves anchor diversity. This does not replace generate_curated_trials.
    It is used when the regime name is 'curated_balanced'.
    Pass max_trials_per_utt=0 to disable the per-utterance cap.
    """
    cap = max_trials_per_utt if max_trials_per_utt and max_trials_per_utt > 0 else None
    trial_count_per_utt: Dict[str, int] = defaultdict(int)
    pos_count_per_source: Dict[str, int] = defaultdict(int)
    seen_pairs: set[Tuple[str, str]] = set()

    sources = list(source_to_paths)
    rng.shuffle(sources)
    positives: List[Pair] = []
    total_pos = 0
    pos_cap = per_source_cap if per_source_cap > 0 else target_pos

    for source in sources:
        utts = source_to_paths[source]
        if len(utts) < 2 or total_pos >= target_pos:
            continue
        shuffled_utts = utts[:]
        rng.shuffle(shuffled_utts)
        while pos_count_per_source[source] < pos_cap and total_pos < target_pos:
            progress = False
            for anchor in shuffled_utts:
                if pos_count_per_source[source] >= pos_cap or total_pos >= target_pos:
                    break
                if cap is not None and trial_count_per_utt[anchor] >= cap:
                    continue
                candidates = [u for u in shuffled_utts if u != anchor and (cap is None or trial_count_per_utt[u] < cap)]
                if not candidates:
                    continue
                partner = _pick_least_used(candidates, trial_count_per_utt, rng)
                if partner is None:
                    continue
                key = tuple(sorted((anchor, partner)))
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                positives.append((anchor, source, partner, source, 1))
                trial_count_per_utt[anchor] += 1
                trial_count_per_utt[partner] += 1
                pos_count_per_source[source] += 1
                total_pos += 1
                progress = True
                if pos_count_per_source[source] >= pos_cap or total_pos >= target_pos:
                    break
            if not progress:
                break

    total_utts = sum(len(v) for v in source_to_paths.values())
    neg_quota: Dict[str, int] = {}
    if total_utts > 0 and target_neg > 0:
        neg_cap = per_source_cap if per_source_cap > 0 else target_neg
        assigned: Dict[str, int] = {}
        for source, utts in source_to_paths.items():
            base = target_neg * len(utts) / total_utts
            quota = min(neg_cap, int(round(base)))
            if quota > 0:
                assigned[source] = quota
        total_quota = sum(assigned.values())
        if total_quota > target_neg:
            scale = target_neg / total_quota if total_quota else 0.0
            adjusted: Dict[str, int] = {}
            fractional_scaled: List[Tuple[float, str]] = []
            for source, quota in assigned.items():
                scaled = quota * scale
                base = int(math.floor(scaled))
                adjusted[source] = base
                fractional_scaled.append((scaled - base, source))
            remaining = target_neg - sum(adjusted.values())
            fractional_scaled.sort(key=lambda x: x[0], reverse=True)
            for i in range(remaining):
                if not fractional_scaled:
                    break
                adjusted[fractional_scaled[i % len(fractional_scaled)][1]] += 1
            neg_quota = {src: q for src, q in adjusted.items() if q > 0}
        else:
            neg_quota = {src: q for src, q in assigned.items() if q > 0}

    negatives: List[Pair] = []
    total_neg = 0
    if neg_quota:
        for src in neg_quota:
            if neg_quota[src] < 0:
                neg_quota[src] = 0
    while total_neg < target_neg and any(q > 0 for q in neg_quota.values()):
        progress = False
        sources_shuffled = [s for s, q in neg_quota.items() if q > 0]
        rng.shuffle(sources_shuffled)
        for source_a in sources_shuffled:
            if total_neg >= target_neg or neg_quota.get(source_a, 0) <= 0:
                continue
            anchor_candidates = [u for u in source_to_paths[source_a] if cap is None or trial_count_per_utt[u] < cap]
            if not anchor_candidates:
                continue
            anchor_candidates.sort(key=lambda x: (trial_count_per_utt[x], rng.random()))
            partner_sources = [
                s for s in source_to_paths
                if s != source_a and any((cap is None or trial_count_per_utt[u] < cap) for u in source_to_paths[s])
            ]
            if not partner_sources:
                continue
            rng.shuffle(partner_sources)

            for anchor in anchor_candidates:
                if neg_quota.get(source_a, 0) <= 0 or total_neg >= target_neg:
                    break
                if cap is not None and trial_count_per_utt[anchor] >= cap:
                    continue
                candidate_partner_sources = [
                    s for s in partner_sources if any((cap is None or trial_count_per_utt[u] < cap) for u in source_to_paths[s])
                ]
                rng.shuffle(candidate_partner_sources)
                added = False
                for partner_source in candidate_partner_sources:
                    partner_candidates = [u for u in source_to_paths[partner_source] if cap is None or trial_count_per_utt[u] < cap]
                    if not partner_candidates:
                        continue
                    partner_candidates.sort(key=lambda x: (trial_count_per_utt[x], rng.random()))
                    for partner in partner_candidates:
                        if partner == anchor:
                            continue
                        key = tuple(sorted((anchor, partner)))
                        if key in seen_pairs:
                            continue
                        seen_pairs.add(key)
                        negatives.append((anchor, source_a, partner, partner_source, 0))
                        trial_count_per_utt[anchor] += 1
                        trial_count_per_utt[partner] += 1
                        neg_quota[source_a] = max(0, neg_quota.get(source_a, 0) - 1)
                        total_neg += 1
                        progress = True
                        added = True
                        break
                    if added or neg_quota.get(source_a, 0) <= 0 or total_neg >= target_neg:
                        break
            # end anchor loop
        if not progress:
            break

    pairs = positives + negatives
    rng.shuffle(pairs)
    if CURATED_BALANCED_DEBUG:
        _print_curated_balanced_debug(pairs)
    if len(positives) < target_pos:
        print(f"[curated_balanced] Warning: generated {len(positives):,} < target_pos {target_pos:,}")
    if len(negatives) < target_neg:
        print(f"[curated_balanced] Warning: generated {len(negatives):,} < target_neg {target_neg:,}")
    effective_cap = pos_cap if pos_cap > 0 else max(pos_count_per_source.values(), default=0)
    return pairs, effective_cap


def load_embeddings(emb_path: Path) -> Dict[str, np.ndarray]:
    data = np.load(emb_path, allow_pickle=True)
    if isinstance(data, np.lib.npyio.NpzFile):
        return {k: data[k] for k in data.files}
    if isinstance(data, np.ndarray) and data.dtype == object:
        maybe_dict = data.item()
        if isinstance(maybe_dict, dict):
            return {str(k): np.asarray(v) for k, v in maybe_dict.items()}
    if isinstance(data, dict):
        return {str(k): np.asarray(v) for k, v in data.items()}
    raise ValueError(
        f"Unsupported embedding format in {emb_path}. Expect npz with keyed arrays or npy containing a dict."
    )


def load_embedding_table(emb_path: Path) -> tuple[list[str], np.ndarray]:
    """
    Load an utterance embedding table. Expected formats:
      - npz with arrays: embeddings [N, D], utt_ids [N]
      - npy containing a dict {utt_id: vector}
      - npz with keyed arrays (dict-like)
    Returns (utt_ids, embeddings_matrix).
    """
    data = np.load(emb_path, allow_pickle=True)
    if isinstance(data, np.lib.npyio.NpzFile) and "embeddings" in data and "utt_ids" in data:
        emb = np.asarray(data["embeddings"])
        utt_ids_arr = np.asarray(data["utt_ids"])
        utt_ids = [str(u) for u in utt_ids_arr.tolist()]
        if emb.shape[0] != len(utt_ids):
            raise ValueError(
                f"Embedding/table shape mismatch in {emb_path}: {emb.shape[0]} rows vs {len(utt_ids)} ids."
            )
        return utt_ids, emb

    if isinstance(data, np.lib.npyio.NpzFile):
        # Fallback to dict-like behaviour
        mapping = {k: np.asarray(data[k]) for k in data.files}
        utt_ids = sorted(mapping)
        emb = np.stack([mapping[u] for u in utt_ids], axis=0)
        return utt_ids, emb

    if isinstance(data, np.ndarray) and data.dtype == object:
        maybe_dict = data.item()
        if isinstance(maybe_dict, dict):
            utt_ids = sorted(maybe_dict)
            emb = np.stack([np.asarray(maybe_dict[u]) for u in utt_ids], axis=0)
            return utt_ids, emb

    if isinstance(data, dict):
        utt_ids = sorted(data)
        emb = np.stack([np.asarray(data[u]) for u in utt_ids], axis=0)
        return utt_ids, emb

    raise ValueError(
        f"Unsupported embedding table format in {emb_path}. "
        "Expect npz with embeddings/utt_ids or a dict of id -> vector."
    )


def sample_diverse_balanced_pairs(
    source_to_paths: Dict[str, List[str]],
    path_to_source: Dict[str, str],
    total_pairs: int,
    rng_seed: int,
    emb_path: Path,
    candidate_multiplier: int,
    pca_dim: int | None,
) -> Tuple[List[Pair], int, int]:
    if total_pairs <= 0:
        return [], 0, 0
    if candidate_multiplier <= 0:
        raise ValueError("candidate_multiplier must be positive.")
    rng = random.Random(rng_seed)
    pos_blocks, pos_cumulative, pos_total = build_positive_blocks(source_to_paths)
    neg_blocks, neg_cumulative, neg_total = build_negative_blocks(source_to_paths)
    desired_half = total_pairs // 2
    balanced_half = min(desired_half, pos_total, neg_total)
    if balanced_half == 0:
        return [], 0, 0
    if pos_total < desired_half:
        print(
            f"  Warning: requested {desired_half:,} positive pairs but only {pos_total:,} available; "
            f"using {balanced_half:,} to keep balance."
        )
    if neg_total < desired_half:
        print(
            f"  Warning: requested {desired_half:,} negative pairs but only {neg_total:,} available; "
            f"using {balanced_half:,} to keep balance."
        )

    cand_per_class = min(balanced_half * candidate_multiplier, pos_total, neg_total)
    pos_indices = rng.sample(range(pos_total), cand_per_class)
    neg_indices = rng.sample(range(neg_total), cand_per_class)

    candidate_trials: List[Tuple[str, str, int]] = []
    needed_paths: set[str] = set()
    for idx in pos_indices:
        path_a, _, path_b, _ = decode_positive_index(idx, pos_blocks, pos_cumulative)
        candidate_trials.append((path_a, path_b, 1))
        needed_paths.update([path_a, path_b])
    for idx in neg_indices:
        path_a, _, path_b, _ = decode_negative_index(idx, neg_blocks, neg_cumulative)
        candidate_trials.append((path_a, path_b, 0))
        needed_paths.update([path_a, path_b])

    embeddings = load_embeddings(emb_path)
    missing = [p for p in needed_paths if p not in embeddings]
    if missing:
        preview = ", ".join(missing[:5])
        raise KeyError(
            f"{len(missing)} embeddings missing (e.g., {preview}). Ensure keys match protocol paths."
        )

    selected_trials = select_balanced_diverse_trials(
        candidate_trials,
        embeddings,
        n_pairs=balanced_half * 2,
        pca_dim=pca_dim,
        random_state=rng_seed,
    )

    pairs: List[Pair] = []
    for u_id, v_id, label in selected_trials:
        src_u = path_to_source.get(u_id)
        src_v = path_to_source.get(v_id)
        if src_u is None or src_v is None:
            raise KeyError(f"Missing source mapping for paths {u_id} or {v_id}.")
        pairs.append((u_id, src_u, v_id, src_v, int(label)))
    rng.shuffle(pairs)
    pos_count = balanced_half
    neg_count = balanced_half
    return pairs, pos_count, neg_count


def _sample_indices(indices: np.ndarray, count: int, rng: np.random.Generator) -> list[int]:
    if count <= 0 or indices.size == 0:
        return []
    count = min(count, indices.size)
    return rng.choice(indices, size=count, replace=False).tolist()


def _sample_banded_indices(indices: np.ndarray, count: int, rng: np.random.Generator) -> list[int]:
    if count <= 0 or indices.size == 0:
        return []
    band1 = indices[: min(5, indices.size)]
    band2 = indices[5: min(20, indices.size)]
    n1 = min(len(band1), max(1, count // 2))
    n2 = min(len(band2), count - n1)
    selected: list[int] = []
    if n1 > 0:
        selected.extend(_sample_indices(band1, n1, rng))
    if n2 > 0:
        selected.extend(_sample_indices(band2, n2, rng))
    if len(selected) < count:
        selected_set = set(selected)
        remaining = np.array([idx for idx in indices if idx not in selected_set])
        selected.extend(_sample_indices(remaining, count - len(selected), rng))
    return selected


def _select_hard_negatives(
    anchor_neg: pd.DataFrame,
    score_column: str,
    num_to_select: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if num_to_select <= 0 or anchor_neg.empty:
        return anchor_neg.iloc[0:0]

    anchor_neg = anchor_neg.copy()
    anchor_neg[score_column] = pd.to_numeric(anchor_neg[score_column], errors="coerce")

    anchor_sorted = anchor_neg.sort_values(by=[score_column, "_rand"], ascending=[False, True])
    q95 = float(anchor_sorted[score_column].quantile(0.95))
    if not np.isfinite(q95):
        anchor_id = (
            str(anchor_sorted["path_A"].iloc[0])
            if "path_A" in anchor_sorted.columns and not anchor_sorted.empty
            else "<unknown>"
        )
        raise ValueError(f"Cannot compute q95 for anchor {anchor_id}: no finite '{score_column}' scores.")
    filtered = anchor_sorted[anchor_sorted[score_column] <= q95]

    if len(filtered) < 2:
        pool = anchor_sorted.iloc[1:10]
        if pool.empty:
            pool = anchor_sorted
        remainder = anchor_sorted.drop(pool.index, errors="ignore")
        selected_indices = _sample_indices(pool.index.to_numpy(), num_to_select, rng)
        if len(selected_indices) < num_to_select and not remainder.empty:
            selected_indices.extend(
                _sample_indices(remainder.index.to_numpy(), num_to_select - len(selected_indices), rng)
            )
        if len(selected_indices) < num_to_select:
            selected_set = set(selected_indices)
            remaining = np.array([idx for idx in anchor_sorted.index if idx not in selected_set])
            selected_indices.extend(_sample_indices(remaining, num_to_select - len(selected_indices), rng))
        return anchor_sorted.loc[selected_indices]

    top_k = min(20, len(filtered))
    pool = filtered.iloc[:top_k]
    remainder = filtered.iloc[top_k:]

    selected_indices = _sample_banded_indices(pool.index.to_numpy(), num_to_select, rng)
    if len(selected_indices) < num_to_select and not remainder.empty:
        selected_indices.extend(
            _sample_indices(remainder.index.to_numpy(), num_to_select - len(selected_indices), rng)
        )
    if len(selected_indices) < num_to_select:
        selected_set = set(selected_indices)
        remaining = np.array([idx for idx in anchor_sorted.index if idx not in selected_set])
        selected_indices.extend(_sample_indices(remaining, num_to_select - len(selected_indices), rng))

    return anchor_sorted.loc[selected_indices]


def _l2_normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < eps:
        raise ValueError("Zero-norm embedding encountered during normalization.")
    return vec / norm


def _direction_vectors(
    anchor_emb: np.ndarray,
    partner_embs: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    u_anchor = _l2_normalize(anchor_emb, eps=eps)
    norms = np.linalg.norm(partner_embs, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    u_partners = partner_embs / norms
    proj = (u_partners @ u_anchor)[:, None] * u_anchor[None, :]
    tangent = u_partners - proj
    tan_norms = np.linalg.norm(tangent, axis=1, keepdims=True)

    fallback = u_partners - u_anchor[None, :]
    fb_norms = np.linalg.norm(fallback, axis=1, keepdims=True)
    use_fallback = tan_norms.squeeze() < eps
    if np.any(use_fallback):
        tangent[use_fallback] = fallback[use_fallback]
        tan_norms[use_fallback] = fb_norms[use_fallback]

    tan_norms = np.maximum(tan_norms, eps)
    return tangent / tan_norms


def _select_directional_indices(
    directions: np.ndarray,
    scores: np.ndarray,
    rand: np.ndarray,
    num_to_select: int,
    rng: np.random.Generator,
    prefer_mask: np.ndarray | None = None,
    seed_pos: int | None = None,
    global_sim: np.ndarray | None = None,
    global_weight: float = 0.0,
    coverage: np.ndarray | None = None,
    rank_bucket: np.ndarray | None = None,
) -> list[int]:
    if num_to_select <= 0:
        return []
    n = directions.shape[0]
    if n == 0:
        return []
    if global_sim is None:
        global_sim = np.zeros(n, dtype=float)
    if global_sim.shape[0] != n:
        raise ValueError("global_sim size mismatch.")
    if coverage is None:
        coverage = np.zeros(n, dtype=float)
    if coverage.shape[0] != n:
        raise ValueError("coverage size mismatch.")
    if rank_bucket is None:
        rank_bucket = np.zeros(n, dtype=float)
    if rank_bucket.shape[0] != n:
        raise ValueError("rank_bucket size mismatch.")
    if seed_pos is None:
        seed_metric = (-scores) + (global_weight * global_sim)
        order = np.lexsort((rand, -scores, coverage, rank_bucket, seed_metric))
        seed_pos = int(order[0])
    selected = [seed_pos]
    while len(selected) < num_to_select:
        candidates = [i for i in range(n) if i not in selected]
        if not candidates:
            break
        if prefer_mask is not None:
            preferred = [i for i in candidates if prefer_mask[i]]
            if preferred:
                candidates = preferred
        cand_dirs = directions[candidates]
        sims = cand_dirs @ directions[selected].T
        max_sim = sims.max(axis=1)
        effective_sim = max_sim + (global_weight * global_sim[candidates])
        order = np.lexsort(
            (
                rand[candidates],
                -scores[candidates],
                coverage[candidates],
                rank_bucket[candidates],
                effective_sim,
            )
        )
        selected.append(candidates[int(order[0])])
    return selected


def _pool_by_bounds(
    anchor_df: pd.DataFrame,
    score_column: str,
    bounds: list[tuple[float, float]],
    fallback_top: int = 10,
) -> pd.DataFrame:
    for low, high in bounds:
        if not np.isfinite(low) or not np.isfinite(high):
            continue
        if low > high:
            continue
        pool = anchor_df[(anchor_df[score_column] >= low) & (anchor_df[score_column] <= high)]
        if not pool.empty:
            return pool
    return anchor_df.sort_values(by=[score_column, "_rand"], ascending=[False, True]).head(
        min(fallback_top, len(anchor_df))
    )


def _resolve_first_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for name in candidates:
        if name in df.columns:
            return name
    return None


def _build_anchor_stats(neg_df: pd.DataFrame, score_column: str) -> pd.DataFrame:
    quantiles = neg_df.groupby("path_A")[score_column].quantile(
        [0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99, 0.995]
    )
    quantiles = quantiles.unstack(level=1).rename(
        columns={
            0.60: "q60",
            0.70: "q70",
            0.80: "q80",
            0.85: "q85",
            0.90: "q90",
            0.95: "q95",
            0.99: "q99",
            0.995: "q995",
        }
    )
    max_vals = neg_df.groupby("path_A")[score_column].max().rename("max")
    stats = quantiles.join(max_vals)
    stats["risk"] = stats["max"] - stats["q95"]
    return stats


def select_hard_mined_pairs(
    scored_pairs_csv: Path,
    score_column: str,
    hard_neg_per_anchor: int,
    rng_seed: int,
) -> Tuple[List[Pair], int, int]:
    """
    From a scored candidate pool, pick the hardest negatives (highest score) per anchor utterance.
    The score_column is expected to be the probability of "same model".
    """
    if hard_neg_per_anchor <= 0:
        raise ValueError("hard_neg_per_anchor must be positive.")
    df = pd.read_csv(scored_pairs_csv)
    required_cols = {
        "path_A",
        "model_name_A",
        "path_B",
        "model_name_B",
        "same_model",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Scored pairs file is missing required columns: {', '.join(sorted(missing))}")
    if score_column not in df.columns:
        raise ValueError(f"Score column '{score_column}' not found in {scored_pairs_csv}.")

    rng = np.random.default_rng(rng_seed)
    df = df.copy()
    df["same_model"] = df["same_model"].astype(int)
    df["_rand"] = rng.random(len(df))  # tie-breaker for stable shuffling

    neg_df = df[df["same_model"] == 0]
    if neg_df.empty:
        raise ValueError("No negative candidates found in scored pairs.")

    anchors = sorted(set(df["path_A"]))
    neg_counts = neg_df.groupby("path_A").size()
    lacking = [a for a in anchors if neg_counts.get(a, 0) < hard_neg_per_anchor]
    if lacking:
        preview = ", ".join(lacking[:5])
        raise ValueError(
            f"Hardmined requires at least {hard_neg_per_anchor} negatives per anchor; "
            f"{len(lacking)} anchors are below that threshold (e.g., {preview})."
        )

    grouped_neg = {anchor: group for anchor, group in neg_df.groupby("path_A", sort=False)}
    neg_rows: list[pd.DataFrame] = []
    for anchor in anchors:
        anchor_neg = grouped_neg.get(anchor)
        if anchor_neg is None or anchor_neg.empty:
            continue
        neg_rows.append(_select_hard_negatives(anchor_neg, score_column, hard_neg_per_anchor, rng))
    neg_selected = pd.concat(neg_rows, ignore_index=True) if neg_rows else pd.DataFrame(columns=df.columns)

    selected = neg_selected.sample(frac=1.0, random_state=rng_seed).reset_index(drop=True)

    pairs: List[Pair] = [
        (row["path_A"], row["model_name_A"], row["path_B"], row["model_name_B"], int(row["same_model"]))
        for _, row in selected.iterrows()
    ]
    pos_count = 0
    neg_count = len(neg_selected)
    return pairs, pos_count, neg_count


def _apply_pca_generic(matrix: np.ndarray, pca_dim: int | None, random_state: int | None) -> np.ndarray:
    if pca_dim is None:
        return matrix
    if pca_dim <= 0:
        raise ValueError("pca_dim must be positive.")
    if pca_dim >= matrix.shape[1]:
        return matrix
    try:
        from sklearn.decomposition import PCA

        centered = matrix - np.mean(matrix, axis=0, keepdims=True)
        return PCA(n_components=pca_dim, random_state=random_state).fit_transform(centered)
    except Exception:
        # Lightweight fallback using SVD
        centered = matrix - np.mean(matrix, axis=0, keepdims=True)
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        components = Vt[:pca_dim]
        return centered @ components.T


def _build_pair_features(
    trials: Sequence[Tuple[Hashable, Hashable, int]],
    embeddings: Mapping[Hashable, np.ndarray],
) -> np.ndarray:
    """
    Construct concatenated pair features for every trial: [z(u), z(v), |z(u) - z(v)|].
    """
    features: List[np.ndarray] = []
    embed_dim: int | None = None
    for idx, (u_id, v_id, y) in enumerate(trials):
        if y not in (0, 1):
            raise ValueError(f"Invalid label {y!r} at trial index {idx}; expected 0 or 1.")
        if u_id not in embeddings:
            raise KeyError(f"Embedding for utterance {u_id!r} missing.")
        if v_id not in embeddings:
            raise KeyError(f"Embedding for utterance {v_id!r} missing.")
        z_u = np.asarray(embeddings[u_id], dtype=float).ravel()
        z_v = np.asarray(embeddings[v_id], dtype=float).ravel()
        if embed_dim is None:
            embed_dim = z_u.shape[0]
        if z_u.shape[0] != embed_dim or z_v.shape[0] != embed_dim:
            raise ValueError(
                f"Embedding dimensionality mismatch at trial {idx}: "
                f"expected {embed_dim}, got {z_u.shape[0]} and {z_v.shape[0]}."
            )
        features.append(np.concatenate([z_u, z_v, np.abs(z_u - z_v)], axis=0))
    if not features:
        raise ValueError("No trials provided to build features from.")
    return np.stack(features, axis=0)


def _farthest_first_subset(
    features: np.ndarray,
    indices: Sequence[int],
    target_size: int,
    rng: np.random.Generator,
) -> List[int]:
    """Run farthest-first traversal over `indices` using precomputed features."""
    if target_size == 0:
        return []
    if len(indices) < target_size:
        raise ValueError(f"Not enough candidates: need {target_size}, have {len(indices)}.")

    subset = features[np.array(indices, dtype=int)]
    seed_local = int(rng.integers(0, len(indices)))
    selected_local: List[int] = [seed_local]
    selected_global: List[int] = [indices[seed_local]]

    min_sq_dists = np.sum((subset - subset[seed_local]) ** 2, axis=1)
    min_sq_dists[seed_local] = -np.inf

    while len(selected_global) < target_size:
        next_local = int(np.argmax(min_sq_dists))
        selected_local.append(next_local)
        selected_global.append(indices[next_local])

        candidate = subset[next_local]
        candidate_sq_dist = np.sum((subset - candidate) ** 2, axis=1)
        min_sq_dists = np.minimum(min_sq_dists, candidate_sq_dist)
        min_sq_dists[selected_local] = -np.inf

    return selected_global


def select_balanced_diverse_trials(
    trials: Sequence[Tuple[Hashable, Hashable, int]],
    embeddings: Mapping[Hashable, np.ndarray],
    n_pairs: int,
    pca_dim: int | None = None,
    random_state: int | None = None,
) -> List[Tuple[Hashable, Hashable, int]]:
    """
    Select n_pairs trials with balanced labels and maximal diversity in feature space.

    Steps:
      1) Split trials into positive/negative pools.
      2) Build pair features phi = [z(u), z(v), |z(u)-z(v)|].
      3) Optionally reduce dimensionality with PCA.
      4) Run farthest-first selection separately per class to pick n_pairs/2 from each.
      5) Return the concatenated selected trials.
    """
    if n_pairs <= 0 or n_pairs % 2 != 0:
        raise ValueError("n_pairs must be a positive, even integer.")

    pos_indices = [idx for idx, (_, _, y) in enumerate(trials) if y == 1]
    neg_indices = [idx for idx, (_, _, y) in enumerate(trials) if y == 0]
    needed_per_class = n_pairs // 2
    if len(pos_indices) < needed_per_class or len(neg_indices) < needed_per_class:
        raise ValueError(
            f"Insufficient class balance: need at least {needed_per_class} positives and "
            f"{needed_per_class} negatives (have {len(pos_indices)} / {len(neg_indices)})."
        )

    features = _build_pair_features(trials, embeddings)
    features = _apply_pca_generic(features, pca_dim, random_state)
    rng = np.random.default_rng(random_state)

    selected_pos = _farthest_first_subset(features, pos_indices, needed_per_class, rng)
    selected_neg = _farthest_first_subset(features, neg_indices, needed_per_class, rng)
    selected_indices = selected_pos + selected_neg
    return [trials[i] for i in selected_indices]


def _farthest_first_points(features: np.ndarray, target_size: int, rng: np.random.Generator) -> list[int]:
    if target_size <= 0:
        return []
    n = features.shape[0]
    if target_size >= n:
        return list(range(n))
    seed_idx = int(rng.integers(0, n))
    selected = [seed_idx]
    min_sq = np.sum((features - features[seed_idx]) ** 2, axis=1)
    min_sq[seed_idx] = -np.inf
    while len(selected) < target_size:
        next_idx = int(np.argmax(min_sq))
        selected.append(next_idx)
        dist = np.sum((features - features[next_idx]) ** 2, axis=1)
        min_sq = np.minimum(min_sq, dist)
        min_sq[selected] = -np.inf
    return selected


def sample_directional_hardmined(
    scored_pairs_csv: Path,
    embeddings_path: Path,
    score_column: str,
    hard_neg_per_anchor: int,
    seed: int,
) -> Tuple[List[Pair], int, int]:
    """
    Coverage-aware Stage-3 sampler with uneven per-anchor budgets:
      1) Compute tail spikiness risk per anchor.
      2) Allocate k in {1,2,3} negatives per anchor while keeping total fixed.
      3) Select tail sculptors with directional novelty and stabilizers with global coverage.
    """
    if hard_neg_per_anchor < 2:
        raise ValueError("Directional hardmined requires hard_neg_per_anchor >= 2.")

    df = pd.read_csv(scored_pairs_csv)
    required_cols = {"path_A", "model_name_A", "path_B", "model_name_B", "same_model", score_column}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Scored pairs file {scored_pairs_csv} missing columns: {', '.join(sorted(missing_cols))}")

    rng = np.random.default_rng(seed)
    df = df.copy()
    df["_rand"] = rng.random(len(df))
    df["same_model"] = df["same_model"].astype(int)
    df[score_column] = pd.to_numeric(df[score_column], errors="coerce")
    neg_df = df[df["same_model"] == 0].dropna(subset=[score_column]).copy()
    if neg_df.empty:
        raise ValueError("No negative candidates found in scored pairs.")

    sys_col_a = _resolve_first_column(df, ["sys_id_A", "sysA", "sys_A", "model_name_A"])
    sys_col_b = _resolve_first_column(df, ["sys_id_B", "sysB", "sys_B", "model_name_B"])
    if sys_col_a is None or sys_col_b is None:
        raise ValueError("Missing sys_id columns (expected sys_id_A/sys_id_B or model_name_A/model_name_B).")

    arch_col_a = _resolve_first_column(
        df,
        ["arch_id_A", "archA", "architecture_A", "model_architecture_A"],
    )
    arch_col_b = _resolve_first_column(
        df,
        ["arch_id_B", "archB", "architecture_B", "model_architecture_B"],
    )
    if arch_col_a is None:
        neg_df["arch_id_A"] = "unknown_architecture"
        arch_col_a = "arch_id_A"
    if arch_col_b is None:
        neg_df["arch_id_B"] = "unknown_architecture"
        arch_col_b = "arch_id_B"

    neg_df[sys_col_a] = neg_df[sys_col_a].fillna("unknown_sys")
    neg_df[sys_col_b] = neg_df[sys_col_b].fillna("unknown_sys")
    neg_df[arch_col_a] = neg_df[arch_col_a].fillna("unknown_architecture")
    neg_df[arch_col_b] = neg_df[arch_col_b].fillna("unknown_architecture")

    utt_ids, emb_matrix = load_embedding_table(embeddings_path)
    id_to_row = {u: i for i, u in enumerate(utt_ids)}
    emb_map = {u: emb_matrix[i] for i, u in enumerate(utt_ids)}

    anchors = sorted(set(neg_df["path_A"]))
    missing_embeds = [u for u in anchors if u not in id_to_row]
    if missing_embeds:
        preview = ", ".join(missing_embeds[:5])
        raise KeyError(
            f"{len(missing_embeds)} anchors missing embeddings (e.g., {preview}). "
            f"Ensure embeddings from the same checkpoint: {embeddings_path}"
        )

    missing_partners = [p for p in df["path_B"].unique() if p not in emb_map]
    if missing_partners:
        preview = ", ".join(missing_partners[:5])
        raise KeyError(
            f"{len(missing_partners)} partner embeddings missing (e.g., {preview}). "
            f"Ensure embeddings for path_B are included: {embeddings_path}"
        )

    neg_counts = neg_df.groupby("path_A").size()

    stats = _build_anchor_stats(neg_df, score_column)
    if stats[["q95", "q99"]].isna().any().any():
        bad = stats[stats[["q95", "q99"]].isna().any(axis=1)].index.tolist()[:5]
        preview = ", ".join(bad)
        raise ValueError(f"Missing score quantiles for anchors (e.g., {preview}).")

    budget_fraction = 0.15
    baseline = hard_neg_per_anchor
    low_budget = baseline - 1
    high_budget = baseline + 1
    if low_budget < 1:
        raise ValueError("hard_neg_per_anchor too small for uneven budgets.")

    n_anchors = len(anchors)
    forced_low = {a for a in anchors if neg_counts.get(a, 0) < baseline}
    forced_missing = {a for a in anchors if neg_counts.get(a, 0) == 0}
    if forced_missing:
        preview = ", ".join(list(forced_missing)[:5])
        raise ValueError(f"{len(forced_missing)} anchors have zero negatives (e.g., {preview}).")
    if forced_low:
        print(
            f"Warning: {len(forced_low)} anchors have fewer than {baseline} negatives; "
            "forcing budget=1 and rebalancing."
        )

    m = int(round(budget_fraction * n_anchors))
    if m * 2 > n_anchors:
        m = n_anchors // 2

    low_size = max(m, len(forced_low))
    if low_size * 2 > n_anchors:
        print(
            "Warning: too many low-budget anchors to fully rebalance; "
            "total negatives may drop below target."
        )
        low_size = min(low_size, n_anchors)
    high_size = min(low_size, n_anchors - low_size)

    risk_sorted = stats.sort_values("risk", ascending=False)
    risk_sorted_asc = stats.sort_values("risk", ascending=True)
    low_risk = set(forced_low)
    if len(low_risk) < low_size:
        for anchor in risk_sorted_asc.index:
            if anchor in low_risk:
                continue
            low_risk.add(anchor)
            if len(low_risk) >= low_size:
                break

    high_risk = set()
    for anchor in risk_sorted.index:
        if anchor in low_risk:
            continue
        high_risk.add(anchor)
        if len(high_risk) >= high_size:
            break

    budget = {anchor: baseline for anchor in anchors}
    for anchor in high_risk:
        budget[anchor] = high_budget
    for anchor in low_risk:
        budget[anchor] = low_budget

    budget_counts = Counter(budget.values())
    print(
        "Budget counts: "
        f"k=1:{budget_counts.get(low_budget, 0)}, "
        f"k=2:{budget_counts.get(baseline, 0)}, "
        f"k=3:{budget_counts.get(high_budget, 0)}"
    )
    if high_risk:
        high_k3 = sum(1 for a in high_risk if budget.get(a) == high_budget)
        print(f"High-risk anchors with k=3: {high_k3:,} / {len(high_risk):,}")

    risk_extreme = float(stats["risk"].quantile(0.95))
    spiky_margin = 0.01
    spiky_by_gap = (stats["max"] - stats["q99"]) > spiky_margin
    very_spiky = set(stats.index[spiky_by_gap | (stats["risk"] >= risk_extreme)])

    grouped_neg = {anchor: group for anchor, group in neg_df.groupby("path_A", sort=False)}
    selected_by_anchor: dict[str, list[dict[str, object]]] = {anchor: [] for anchor in anchors}
    selected_sets: dict[str, set[int]] = {anchor: set() for anchor in anchors}
    count_sys_b: Counter[str] = Counter()
    count_pair: Counter[tuple[str, str]] = Counter()
    dirs_used: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    dir_seen: Counter[tuple[str, str]] = Counter()
    dir_memory_cap = 100
    tail_global_weight = 0.35
    stabilizer_global_weight = 0.12
    tail_pool_sizes: list[int] = []

    anchor_order = anchors[:]
    rng.shuffle(anchor_order)

    def build_tail_pool(anchor_df: pd.DataFrame, stats_row: pd.Series, tail_needed: int) -> pd.DataFrame:
        bounds = [
            (stats_row["q95"], stats_row["q99"]),
            (stats_row["q90"], stats_row["q99"]),
        ]
        pool = _pool_by_bounds(anchor_df, score_column, bounds, fallback_top=10)
        if not pool.empty:
            pool = pool.sort_values(by=[score_column, "_rand"], ascending=[False, True]).head(
                min(20, len(pool))
            )

        q995 = stats_row.get("q995")
        if np.isfinite(q995):
            trimmed = pool[pool[score_column] <= q995]
            if not trimmed.empty:
                pool = trimmed
        if not np.isfinite(q995):
            if len(pool) > max(2, tail_needed):
                pool = pool.sort_values(by=[score_column, "_rand"], ascending=[False, True]).iloc[1:]

        tail_cap_rank = 2
        pool = pool.sort_values(by=[score_column, "_rand"], ascending=[False, True])
        if len(pool) >= tail_cap_rank:
            cap_score = float(pool.iloc[tail_cap_rank - 1][score_column])
            pool = pool[pool[score_column] <= cap_score]

        if pool.empty:
            anchor_sorted = anchor_df.sort_values(by=[score_column, "_rand"], ascending=[False, True])
            if np.isfinite(q995):
                trimmed = anchor_sorted[anchor_sorted[score_column] <= q995]
                if not trimmed.empty:
                    anchor_sorted = trimmed
            if not np.isfinite(q995) and len(anchor_sorted) > max(2, tail_needed):
                anchor_sorted = anchor_sorted.iloc[1:]
            pool = anchor_sorted.head(min(20, len(anchor_sorted)))
            if len(pool) >= tail_cap_rank:
                cap_score = float(pool.iloc[tail_cap_rank - 1][score_column])
                pool = pool[pool[score_column] <= cap_score]
        return pool

    def build_shoulder_pool(anchor_df: pd.DataFrame, stats_row: pd.Series) -> pd.DataFrame:
        return _pool_by_bounds(
            anchor_df,
            score_column,
            [
                (stats_row["q85"], stats_row["q95"]),
                (stats_row["q80"], stats_row["q95"]),
                (stats_row["q70"], stats_row["q95"]),
            ],
            fallback_top=15,
        )

    def build_easy_pool(anchor_df: pd.DataFrame, stats_row: pd.Series) -> pd.DataFrame:
        return _pool_by_bounds(
            anchor_df,
            score_column,
            [
                (stats_row["q70"], stats_row["q85"]),
                (stats_row["q60"], stats_row["q85"]),
            ],
            fallback_top=20,
        )

    def select_stabilizer(
        anchor_df: pd.DataFrame,
        stats_row: pd.Series,
        anchor_sys: str,
        anchor_arch: str,
        anchor_emb: np.ndarray,
        anchor_budget: int,
        is_spiky: bool,
        prefer_sys: set[str] | None = None,
    ) -> tuple[int | None, np.ndarray | None]:
        pool = build_shoulder_pool(anchor_df, stats_row)
        if is_spiky and anchor_budget == high_budget:
            pool = build_easy_pool(anchor_df, stats_row)
        pool = pool[~pool.index.isin(selected_sets[anchor])]
        if pool.empty:
            pool = anchor_df[~anchor_df.index.isin(selected_sets[anchor])]
        if pool.empty:
            return None, None
        if prefer_sys:
            preferred = pool[pool[sys_col_b].isin(prefer_sys)]
            if not preferred.empty:
                pool = preferred

        partner_paths = pool["path_B"].tolist()
        partner_embs = np.stack([np.asarray(emb_map[p], dtype=float).ravel() for p in partner_paths], axis=0)
        directions = _direction_vectors(anchor_emb, partner_embs)
        scores = pool[score_column].to_numpy()
        rand_vals = pool["_rand"].to_numpy()
        sys_b = pool[sys_col_b].astype(str).to_numpy()
        arch_b = pool[arch_col_b].astype(str).to_numpy()
        arch_diff = (arch_b != anchor_arch).astype(int)

        count_sys = np.array([count_sys_b[s] for s in sys_b])
        count_pairs = np.array([count_pair[(anchor_sys, s)] for s in sys_b])
        global_sim = np.empty(len(pool), dtype=float)
        for i, s in enumerate(sys_b):
            used = dirs_used.get((anchor_sys, s))
            if used:
                used_arr = np.stack(used, axis=0)
                global_sim[i] = float(np.max(directions[i] @ used_arr.T))
            else:
                global_sim[i] = -1.0

        global_pen = stabilizer_global_weight * global_sim
        order = np.lexsort((rand_vals, global_pen, -scores, -arch_diff, count_pairs, count_sys))
        pos = int(order[0])
        return int(pool.index[pos]), directions[pos]

    def update_dir_memory(key: tuple[str, str], direction: np.ndarray) -> None:
        dir_seen[key] += 1
        store = dirs_used[key]
        if len(store) < dir_memory_cap:
            store.append(direction)
            return
        replace_idx = int(rng.integers(0, dir_seen[key]))
        if replace_idx < dir_memory_cap:
            store[replace_idx] = direction

    # Phase 1: tail sculptors
    for anchor in anchor_order:
        k = budget[anchor]
        tail_count = 0 if k <= low_budget else min(2, k - 1)
        if tail_count <= 0:
            continue
        anchor_df = grouped_neg[anchor]
        stats_row = stats.loc[anchor]
        anchor_emb = np.asarray(emb_map[anchor], dtype=float).ravel()
        anchor_sys = str(anchor_df[sys_col_a].iloc[0])
        anchor_arch = str(anchor_df[arch_col_a].iloc[0])

        pool = build_tail_pool(anchor_df, stats_row, tail_count)
        tail_pool_sizes.append(len(pool))
        pool = pool[~pool.index.isin(selected_sets[anchor])]
        if pool.empty:
            pool = anchor_df[~anchor_df.index.isin(selected_sets[anchor])]
        if pool.empty:
            continue

        partner_paths = pool["path_B"].tolist()
        partner_embs = np.stack([np.asarray(emb_map[p], dtype=float).ravel() for p in partner_paths], axis=0)
        directions = _direction_vectors(anchor_emb, partner_embs)
        scores = pool[score_column].to_numpy()
        rand_vals = pool["_rand"].to_numpy()
        sys_b = pool[sys_col_b].astype(str).to_numpy()
        coverage_pairs = np.array([count_pair[(anchor_sys, s)] for s in sys_b], dtype=float)
        order_scores = np.argsort(-scores)
        rank_bucket = np.ones(len(pool), dtype=float)
        rank_bucket[order_scores[: min(5, len(pool))]] = 0.0
        global_sim = np.empty(len(pool), dtype=float)
        for i, s in enumerate(sys_b):
            used = dirs_used.get((anchor_sys, s))
            if used:
                used_arr = np.stack(used, axis=0)
                global_sim[i] = float(np.max(directions[i] @ used_arr.T))
            else:
                global_sim[i] = -1.0

        if tail_count == 2:
            seed_pos = _select_directional_indices(
                directions,
                scores,
                rand_vals,
                1,
                rng,
                global_sim=global_sim,
                global_weight=tail_global_weight,
                coverage=coverage_pairs,
                rank_bucket=rank_bucket,
            )[0]
            sys_seed = str(pool.iloc[seed_pos][sys_col_b])
            arch_seed = str(pool.iloc[seed_pos][arch_col_b])
            prefer_mask = (pool[sys_col_b].astype(str).to_numpy() != sys_seed) | (
                pool[arch_col_b].astype(str).to_numpy() != arch_seed
            )
            selected_pos = _select_directional_indices(
                directions,
                scores,
                rand_vals,
                2,
                rng,
                prefer_mask=prefer_mask,
                seed_pos=seed_pos,
                global_sim=global_sim,
                global_weight=tail_global_weight,
                coverage=coverage_pairs,
                rank_bucket=rank_bucket,
            )
        else:
            selected_pos = _select_directional_indices(
                directions,
                scores,
                rand_vals,
                1,
                rng,
                global_sim=global_sim,
                global_weight=tail_global_weight,
                coverage=coverage_pairs,
                rank_bucket=rank_bucket,
            )

        tail_picks: list[tuple[int, np.ndarray, str]] = []
        for pos in selected_pos:
            row_idx = int(pool.index[pos])
            tail_picks.append((row_idx, directions[pos], str(pool.iloc[pos][sys_col_b])))

        if tail_count == 2 and len(pool) < 2 and tail_picks:
            first_dir = tail_picks[0][1]
            shoulder = build_shoulder_pool(anchor_df, stats_row)
            shoulder = shoulder[~shoulder.index.isin(selected_sets[anchor])]
            if not shoulder.empty:
                shoulder = shoulder.sort_values(by=[score_column, "_rand"], ascending=[False, True])
                shoulder = shoulder.head(min(10, len(shoulder)))
                shoulder_paths = shoulder["path_B"].tolist()
                shoulder_embs = np.stack(
                    [np.asarray(emb_map[p], dtype=float).ravel() for p in shoulder_paths], axis=0
                )
                shoulder_dirs = _direction_vectors(anchor_emb, shoulder_embs)
                shoulder_scores = shoulder[score_column].to_numpy()
                shoulder_rand = shoulder["_rand"].to_numpy()
                shoulder_sys = shoulder[sys_col_b].astype(str).to_numpy()
                shoulder_cov = np.array([count_pair[(anchor_sys, s)] for s in shoulder_sys], dtype=float)
                shoulder_global = np.empty(len(shoulder), dtype=float)
                for i, s in enumerate(shoulder_sys):
                    used = dirs_used.get((anchor_sys, s))
                    if used:
                        used_arr = np.stack(used, axis=0)
                        shoulder_global[i] = float(np.max(shoulder_dirs[i] @ used_arr.T))
                    else:
                        shoulder_global[i] = -1.0
                max_sim = shoulder_dirs @ first_dir.reshape(-1, 1)
                effective_sim = max_sim.squeeze() + tail_global_weight * shoulder_global
                order = np.lexsort((shoulder_rand, shoulder_cov, -shoulder_scores, effective_sim))
                second_idx = int(order[0])
                row_idx = int(shoulder.index[second_idx])
                tail_picks.append((row_idx, shoulder_dirs[second_idx], str(shoulder_sys[second_idx])))

        for row_idx, direction, sys_b_val in tail_picks:
            if row_idx in selected_sets[anchor]:
                continue
            selected_sets[anchor].add(row_idx)
            selected_by_anchor[anchor].append({"idx": row_idx, "role": "tail"})
            count_sys_b[sys_b_val] += 1
            count_pair[(anchor_sys, sys_b_val)] += 1
            update_dir_memory((anchor_sys, sys_b_val), direction)

    # Phase 2: stabilizers
    for anchor in anchor_order:
        k = budget[anchor]
        if len(selected_by_anchor[anchor]) >= k:
            continue
        anchor_df = grouped_neg[anchor]
        stats_row = stats.loc[anchor]
        anchor_emb = np.asarray(emb_map[anchor], dtype=float).ravel()
        anchor_sys = str(anchor_df[sys_col_a].iloc[0])
        anchor_arch = str(anchor_df[arch_col_a].iloc[0])

        row_idx, direction = select_stabilizer(
            anchor_df,
            stats_row,
            anchor_sys,
            anchor_arch,
            anchor_emb,
            k,
            anchor in very_spiky,
        )
        if row_idx is None:
            continue
        selected_sets[anchor].add(row_idx)
        selected_by_anchor[anchor].append({"idx": row_idx, "role": "stabilizer"})
        sys_b_val = str(neg_df.loc[row_idx, sys_col_b])
        count_sys_b[sys_b_val] += 1
        count_pair[(anchor_sys, sys_b_val)] += 1
        if direction is not None:
            update_dir_memory((anchor_sys, sys_b_val), direction)

    # Fill any remaining gaps
    for anchor in anchors:
        k = budget[anchor]
        if len(selected_by_anchor[anchor]) >= k:
            continue
        anchor_df = grouped_neg[anchor]
        remaining = anchor_df[~anchor_df.index.isin(selected_sets[anchor])]
        if remaining.empty:
            continue
        remaining = remaining.sort_values(by=[score_column, "_rand"], ascending=[False, True])
        needed = k - len(selected_by_anchor[anchor])
        for row_idx in remaining.head(needed).index.tolist():
            selected_sets[anchor].add(int(row_idx))
            selected_by_anchor[anchor].append({"idx": int(row_idx), "role": "fill"})
            sys_b = str(anchor_df.loc[row_idx, sys_col_b])
            anchor_sys = str(anchor_df[sys_col_a].iloc[0])
            count_sys_b[sys_b] += 1
            count_pair[(anchor_sys, sys_b)] += 1

    # Global repair pass for underrepresented systems
    if count_sys_b:
        counts = np.array(list(count_sys_b.values()), dtype=float)
        target = float(np.median(counts) * 0.8)
        under_sys = {s for s, c in count_sys_b.items() if c < target}
        if under_sys:
            anchors_shuffled = anchors[:]
            rng.shuffle(anchors_shuffled)
            for sys_target in list(under_sys):
                for anchor in anchors_shuffled:
                    entries = [e for e in selected_by_anchor[anchor] if e["role"] == "stabilizer"]
                    if not entries:
                        continue
                    anchor_df = grouped_neg[anchor]
                    stats_row = stats.loc[anchor]
                    anchor_emb = np.asarray(emb_map[anchor], dtype=float).ravel()
                    anchor_sys = str(anchor_df[sys_col_a].iloc[0])
                    anchor_arch = str(anchor_df[arch_col_a].iloc[0])

                    old_idx = int(entries[0]["idx"])
                    selected_sets[anchor].remove(old_idx)
                    new_idx, _ = select_stabilizer(
                        anchor_df,
                        stats_row,
                        anchor_sys,
                        anchor_arch,
                        anchor_emb,
                        budget[anchor],
                        anchor in very_spiky,
                        prefer_sys={sys_target},
                    )
                    if new_idx is None:
                        selected_sets[anchor].add(old_idx)
                        continue
                    new_idx = int(new_idx)
                    if new_idx == old_idx:
                        selected_sets[anchor].add(old_idx)
                        continue

                    old_sys = str(anchor_df.loc[old_idx, sys_col_b])
                    count_sys_b[old_sys] -= 1
                    count_pair[(anchor_sys, old_sys)] -= 1
                    if count_sys_b[old_sys] <= 0:
                        del count_sys_b[old_sys]
                    entries[0]["idx"] = new_idx
                    selected_sets[anchor].add(new_idx)
                    new_sys = str(anchor_df.loc[new_idx, sys_col_b])
                    count_sys_b[new_sys] += 1
                    count_pair[(anchor_sys, new_sys)] += 1
                    if count_sys_b.get(sys_target, 0) >= target:
                        under_sys.discard(sys_target)
                    break

    selected_indices = [entry["idx"] for entries in selected_by_anchor.values() for entry in entries]
    expected_total = baseline * len(anchors)
    if len(selected_indices) != expected_total:
        raise ValueError(
            f"Directional hardmined selected {len(selected_indices)} negatives, "
            f"expected {expected_total} (anchors={len(anchors)}, baseline={baseline})."
        )
    if tail_pool_sizes:
        tail_sizes = np.asarray(tail_pool_sizes, dtype=float)
        small_count = int((tail_sizes < 2).sum())
        stats_tail = np.quantile(tail_sizes, [0.0, 0.25, 0.5, 0.75, 1.0])
        print(
            "Tail pool sizes (after cap): "
            f"min={stats_tail[0]:.0f}, p25={stats_tail[1]:.0f}, "
            f"median={stats_tail[2]:.0f}, p75={stats_tail[3]:.0f}, "
            f"max={stats_tail[4]:.0f}, <2={small_count}"
        )
    neg_selected = neg_df.loc[selected_indices].copy()
    selected = neg_selected.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    pairs: List[Pair] = [
        (row["path_A"], row["model_name_A"], row["path_B"], row["model_name_B"], int(row["same_model"]))
        for _, row in selected.iterrows()
    ]
    pos_count = 0
    neg_count = len(neg_selected)
    return pairs, pos_count, neg_count


def write_pairs(output_path: Path, pairs: List[Tuple]) -> None:
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        if pairs and len(pairs[0]) == 6:
            writer.writerow(["path_A", "model_name_A", "path_B", "model_name_B", "same_model", "is_forced_meta"])
            for path_a, source_a, path_b, source_b, label, forced in pairs:
                writer.writerow([path_a, source_a, path_b, source_b, label, forced])
        else:
            writer.writerow(["path_A", "model_name_A", "path_B", "model_name_B", "same_model"])
            for path_a, source_a, path_b, source_b, label in pairs:
                writer.writerow([path_a, source_a, path_b, source_b, label])


def format_output_name(base_name: str, seed_suffix: str) -> str:
    stem, ext = base_name.rsplit(".", 1) if "." in base_name else (base_name, "")
    return f"{stem}{seed_suffix}.{ext}" if ext else f"{stem}{seed_suffix}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate balanced MLAAD train pairs (B/2 target + B/2 non-target)."
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        required=True,
        help="Single-utterance MLAAD train protocol (columns: path, model_name).",
    )
    parser.add_argument(
        "--path-column",
        default="path",
        help="CSV column containing relative audio paths.",
    )
    parser.add_argument(
        "--source-column",
        default="model_name",
        help="CSV column identifying the source/model label.",
    )
    parser.add_argument(
        "--total-pairs",
        type=int,
        default=44000,
        help="Total number of pairs B to sample (balanced 50/50 targets).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling.",
    )
    parser.add_argument(
        "--method",
        choices=[
            "minimal",
            "intermediate",
            "curated",
            "curated_balanced",
            "random",
            "rival",
            "directional",
            "directional_hardmined",
            "hardmined",
        ],
        default="random",
        help="Sampling strategy: Stage-1 regimes (minimal/intermediate/curated/curated_balanced), or Stage-3 style (random/rival/directional/directional_hardmined/hardmined).",
    )
    parser.add_argument(
        "--rivals-csv",
        type=Path,
        help="Rival mapping CSV (columns: model_name, rival_model_name) for rival method.",
    )
    parser.add_argument(
        "--rival-source-column",
        default="model_name",
        help="Column in rivals CSV containing the source/model label.",
    )
    parser.add_argument(
        "--rival-target-column",
        default="rival_model_name",
        help="Column in rivals CSV containing the rival label.",
    )
    parser.add_argument(
        "--rival-forced-column",
        default="is_forced_meta",
        help="Column in rivals CSV marking metadata-forced rivals (default: is_forced_meta).",
    )
    parser.add_argument(
        "--rival-neg-per-anchor",
        type=int,
        default=2,
        help="Rival method: negatives per anchor (default: 2).",
    )
    parser.add_argument(
        "--rival-max-partner-uses",
        type=int,
        default=0,
        help="Rival method: max times any partner utterance can appear (0 disables the cap).",
    )
    parser.add_argument(
        "--intermediate-max-same",
        type=int,
        default=2,
        help="Intermediate regime: max positive partners per anchor.",
    )
    parser.add_argument(
        "--intermediate-max-diff",
        type=int,
        default=2,
        help="Intermediate regime: max negative partners per anchor.",
    )
    parser.add_argument(
        "--curated-target-pos",
        type=int,
        default=40000,
        help="Curated regime: target number of positive pairs.",
    )
    parser.add_argument(
        "--curated-target-neg",
        type=int,
        default=40000,
        help="Curated regime: target number of negative pairs.",
    )
    parser.add_argument(
        "--curated-pos-cap",
        type=int,
        default=0,
        help="Curated regime: per-source positive cap (0 = auto).",
    )
    parser.add_argument(
        "--curated-max-trials-per-utt",
        type=int,
        default=0,
        help="Curated_balanced regime: max times any utterance can appear (anchor or partner). 0 disables the cap.",
    )
    parser.add_argument(
        "--candidate-multiplier",
        type=int,
        default=5,
        help="For directional method: sample this multiple of B/2 per class before diversity pruning.",
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        help="Embeddings file (npz with keyed arrays or npy containing a dict) required for directional method.",
    )
    parser.add_argument(
        "--pca-dim",
        type=int,
        default=None,
        help="Optional PCA dimension for directional method (applied to pair features).",
    )
    parser.add_argument(
        "--scored-pairs",
        type=Path,
        help="Scored candidate pairs CSV for hardmined/directional methods (must include score column).",
    )
    parser.add_argument(
        "--hard-neg-per-anchor",
        type=int,
        default=2,
        help="Hardmined/directional (negative-only): number of negatives to keep per anchor.",
    )
    parser.add_argument(
        "--hard-score-column",
        default="score_same",
        help="Hardmined/directional: score column to use (probability of 'same').",
    )
    parser.add_argument(
        "--dir-pos-per-anchor",
        type=int,
        default=0,
        help="Directional hardmined: ignored (directional is negative-only).",
    )
    parser.add_argument(
        "--dir-neg-per-anchor",
        dest="hard_neg_per_anchor",
        type=int,
        default=argparse.SUPPRESS,
        help="Alias for --hard-neg-per-anchor (directional/hardmined).",
    )
    parser.add_argument(
        "--dir-target-budget",
        type=int,
        default=44000,
        help="Directional hardmined: ignored (use --hard-neg-per-anchor).",
    )
    parser.add_argument(
        "--dir-pca-dim",
        type=int,
        default=32,
        help="Directional hardmined: ignored (selection uses original embedding space).",
    )
    parser.add_argument(
        "--dir-score-column",
        dest="hard_score_column",
        default=argparse.SUPPRESS,
        help="Alias for --hard-score-column (directional/hardmined).",
    )
    parser.add_argument(
        "--dir-skip-easiest-pos",
        action="store_true",
        help="Directional hardmined: ignored (directional is negative-only).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to store the sampled train manifest (default: pair-protocols-stage3 next to the train CSV).",
    )
    parser.add_argument(
        "--output-name",
        default="",
        help="Optional output filename (default: train_pairs_stage3_random_B{N}.csv, with seed suffix if requested).",
    )
    parser.add_argument(
        "--seed-in-filename",
        action="store_true",
        help="Append '_seed{seed}' to the output filename for reproducibility.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_to_paths, path_to_source = load_protocol(args.train_csv, args.path_column, args.source_column)

    method = args.method
    seed_suffix = f"_seed{args.seed}" if args.seed_in_filename else ""
    rows_list: Stage1Rows = [(p, s) for s, paths in source_to_paths.items() for p in paths]
    default_output_dir = (
        args.train_csv.parent
        / ("pair-protocols-stage1" if method in {"minimal", "intermediate", "curated", "curated_balanced"} else "pair-protocols-stage3")
    )

    if method == "minimal":
        neg_index_map = build_negative_index_map(rows_list)
        rng = random.Random(args.seed + STAGE1_REGIME_OFFSETS["minimal"])
        pairs = generate_minimal_trials(rows_list, source_to_paths, neg_index_map, rng)
        pos_count = sum(1 for _, _, _, _, lbl in pairs if lbl == 1)
        neg_count = len(pairs) - pos_count
        base_name = args.output_name.strip() or "train_pairs_stage1_minimal.csv"
    elif method == "intermediate":
        neg_index_map = build_negative_index_map(rows_list)
        rng = random.Random(args.seed + STAGE1_REGIME_OFFSETS["intermediate"])
        pairs = generate_intermediate_trials(
            rows_list,
            source_to_paths,
            neg_index_map,
            rng,
            max_same=args.intermediate_max_same,
            max_diff=args.intermediate_max_diff,
        )
        pos_count = sum(1 for _, _, _, _, lbl in pairs if lbl == 1)
        neg_count = len(pairs) - pos_count
        base_name = args.output_name.strip() or "train_pairs_stage1_intermediate.csv"
    elif method == "curated":
        rng = random.Random(args.seed + STAGE1_REGIME_OFFSETS["curated"])
        pairs, cap = generate_curated_trials(
            source_to_paths,
            rng,
            target_pos=args.curated_target_pos,
            target_neg=args.curated_target_neg,
            per_source_cap=args.curated_pos_cap,
        )
        pos_count = sum(1 for _, _, _, _, lbl in pairs if lbl == 1)
        neg_count = len(pairs) - pos_count
        base_name = args.output_name.strip() or "train_pairs_stage1_curated.csv"
        print(f"  Curated per-source cap used: {cap}")
    elif method == "curated_balanced":
        rng = random.Random(args.seed + STAGE1_REGIME_OFFSETS["curated_balanced"])
        pairs, cap = generate_curated_balanced_trials(
            source_to_paths,
            rng,
            target_pos=args.curated_target_pos,
            target_neg=args.curated_target_neg,
            per_source_cap=args.curated_pos_cap,
            max_trials_per_utt=args.curated_max_trials_per_utt,
        )
        pos_count = sum(1 for _, _, _, _, lbl in pairs if lbl == 1)
        neg_count = len(pairs) - pos_count
        base_name = args.output_name.strip() or "train_pairs_stage1_curated_balanced.csv"
        print(f"  Curated_balanced per-source cap used: {cap}")
    elif method == "hardmined":
        if not args.scored_pairs:
            raise ValueError("--scored-pairs is required when --method hardmined.")
        pairs, pos_count, neg_count = select_hard_mined_pairs(
            args.scored_pairs,
            score_column=args.hard_score_column,
            hard_neg_per_anchor=args.hard_neg_per_anchor,
            rng_seed=args.seed,
        )
        base_name = args.output_name.strip() or f"train_pairs_stage3_hardmined_B{len(pairs)}.csv"
    elif method == "rival":
        if not args.rivals_csv:
            raise ValueError("--rivals-csv is required when --method rival.")
        rivals, forced_lookup = load_rival_map(
            args.rivals_csv,
            args.rival_source_column,
            args.rival_target_column,
            args.rival_forced_column,
        )
        rng = random.Random(args.seed)
        pairs, pos_count, neg_count = sample_rival_pairs(
            rows_list,
            source_to_paths,
            rivals,
            forced_lookup,
            rng,
            neg_per_anchor=args.rival_neg_per_anchor,
            max_pairs=args.total_pairs,
            max_partner_uses=args.rival_max_partner_uses,
        )
        base_name = args.output_name.strip() or f"train_pairs_stage3_rival_B{len(pairs)}.csv"
    elif method in {"directional", "directional_hardmined"}:
        if not args.scored_pairs or not args.embeddings:
            raise ValueError("--scored-pairs and --embeddings are required when --method directional.")
        if args.dir_pos_per_anchor != 0:
            print("Warning: --dir-pos-per-anchor is ignored (directional is negative-only).")
        if args.dir_target_budget != 44000:
            print("Warning: --dir-target-budget is ignored (use --hard-neg-per-anchor).")
        if args.dir_pca_dim != 32:
            print("Warning: --dir-pca-dim is ignored (selection uses original embedding space).")
        if args.dir_skip_easiest_pos:
            print("Warning: --dir-skip-easiest-pos is ignored (directional is negative-only).")
        pairs, pos_count, neg_count = sample_directional_hardmined(
            scored_pairs_csv=args.scored_pairs,
            embeddings_path=args.embeddings,
            score_column=args.hard_score_column,
            hard_neg_per_anchor=args.hard_neg_per_anchor,
            seed=args.seed,
        )
        base_name = args.output_name.strip() or f"train_pairs_stage3_directional_B{len(pairs)}.csv"
    else:
        pairs, pos_count, neg_count = sample_random_balanced_pairs(
            source_to_paths,
            args.total_pairs,
            args.seed,
        )
        base_name = args.output_name.strip() or f"train_pairs_stage3_random_B{len(pairs)}.csv"

    if not pairs:
        print("No pairs generated (insufficient data or non-positive total-pairs).")
        return

    actual_total = len(pairs)
    output_dir = args.output_dir if args.output_dir else default_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    out_name = format_output_name(base_name, seed_suffix)
    out_path = output_dir / out_name

    write_pairs(out_path, pairs)
    print(f"Wrote {out_path}")
    print(f"  positives={pos_count:,}, negatives={neg_count:,}, total={actual_total:,}")


if __name__ == "__main__":
    main()
