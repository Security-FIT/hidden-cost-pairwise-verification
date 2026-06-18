from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from sklearn.metrics import DetCurveDisplay, det_curve

try:
    # Interspeech CI framework
    from confidence_intervals import evaluate_with_conf_int
except ImportError:
    evaluate_with_conf_int = None


def calculate_EER(
    name,
    labels,
    predictions,
    plot_det: bool,
    det_subtitle: str,
    output_dir: Path | str | None = None,
) -> float:
    """
    Calculate the Equal Error Rate (EER) from the labels and predictions
    """
    fpr, fnr, _ = det_curve(labels, predictions, pos_label=0)

    # eer from fpr and fnr can differ a bit (its an approximation), so we compute both and take the average
    eer_fpr = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    eer_fnr = fnr[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = (eer_fpr + eer_fnr) / 2

    # Display the DET curve
    if plot_det:
        eer_probit = norm.ppf(eer)

        DetCurveDisplay(fpr=fpr, fnr=fnr, pos_label=0).plot()
        plt.plot(eer_probit, eer_probit, marker="o", markersize=4, label=f"EER: {eer:.2f}", color="red")
        plt.legend()
        plt.title(f"DET Curve {name} {det_subtitle}")
        save_path = Path(f"{name}_{det_subtitle}_DET.png")
        if output_dir is not None:
            save_path = Path(output_dir) / save_path
            save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)

    return eer


def bootstrap_eer_confidence_interval(
    labels: Sequence[float] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    n_bootstrap: int = 500,
    alpha: float = 0.05,
    conditions: Sequence[int] | np.ndarray | None = None,
    random_state: int | None = 0,
) -> tuple[float, float] | None:
    """
    Estimate a two-sided confidence interval for EER via bootstrapping.

    params:
        labels: Ground-truth labels for the development set.
        scores: Model scores for the same examples (probability of bonafide).
        n_bootstrap: Number of bootstrap samples.
        alpha: Significance level (0.05 -> 95% CI).
        conditions: Optional grouping variable for condition-aware sampling (speaker/session).
        random_state: Seed for reproducible bootstrapping.
    returns:
        (lower, upper) EER bounds or None if the bootstrap estimate is not available.
    """
    labels_arr = np.asarray(labels)
    scores_arr = np.asarray(scores)
    if labels_arr.size == 0:
        return None
    conditions_arr = None if conditions is None else np.asarray(conditions)

    rng = np.random.default_rng(random_state)
    indices = np.arange(labels_arr.shape[0])
    eers = []
    for _ in range(n_bootstrap):
        if conditions_arr is None:
            sample_idx = rng.choice(indices, size=indices.size, replace=True)
        else:
            unique_conditions = np.unique(conditions_arr)
            sampled_conditions = rng.choice(unique_conditions, size=len(unique_conditions), replace=True)
            sampled_lists = []
            for cond in unique_conditions:
                count = np.sum(sampled_conditions == cond)
                if count == 0:
                    continue
                cond_indices = indices[conditions_arr == cond]
                cond_sample = rng.choice(cond_indices, size=len(cond_indices), replace=True)
                sampled_lists.append(np.repeat(cond_sample, count))
            sample_idx = np.concatenate(sampled_lists) if sampled_lists else np.array([], dtype=int)
            if sample_idx.size == 0:
                continue
        sample_labels = labels_arr[sample_idx]
        if len(np.unique(sample_labels)) < 2:
            # Skip degenerate resamples containing a single class
            continue
        sample_scores = scores_arr[sample_idx]
        eers.append(calculate_EER("bootstrap", sample_labels, sample_scores, False, det_subtitle="bootstrap"))

    if not eers:
        return None

    lower = np.percentile(eers, 100 * (alpha / 2))
    upper = np.percentile(eers, 100 * (1 - alpha / 2))
    return float(lower), float(upper)


def interspeech_eer_confidence_interval(
    labels: Sequence[float] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    conditions: Sequence[int] | np.ndarray | None = None,
) -> tuple[float | None, tuple[float, float] | None]:
    """
    Compute EER and its confidence interval using the Interspeech CI framework.

    Returns (eer, (low, high)) or (None, None) when the dependency is missing.
    """
    if evaluate_with_conf_int is None:
        return None, None

    labels_arr = np.asarray(labels)
    scores_arr = np.asarray(scores)
    cond_arr = None if conditions is None else np.asarray(conditions)

    def _eer_metric(metric_labels, metric_scores):
        return calculate_EER("CI", metric_labels, metric_scores, False, det_subtitle="CI")

    alpha_pct = alpha * 100 if alpha < 1 else alpha
    center, ci_bounds = evaluate_with_conf_int(
        samples=scores_arr,
        metric=_eer_metric,
        labels=labels_arr,
        conditions=cond_arr,
        num_bootstraps=n_bootstrap,
        alpha=alpha_pct,
    )
    ci_tuple = None if ci_bounds is None else (float(ci_bounds[0]), float(ci_bounds[1]))
    return float(center), ci_tuple


def confidence_intervals_overlap(
    ci_a: tuple[float, float] | None, ci_b: tuple[float, float] | None
) -> bool:
    """Return True if the provided confidence intervals overlap."""
    if ci_a is None or ci_b is None:
        return False
    return max(ci_a[0], ci_b[0]) <= min(ci_a[1], ci_b[1])


def _order_level_results(
    level_results: Sequence[Mapping[str, Any]], level_order: Sequence[Any] | None
) -> list[Mapping[str, Any]]:
    if level_order:
        order_map = {lvl: idx for idx, lvl in enumerate(level_order)}
        return sorted(
            level_results, key=lambda res: order_map.get(res.get("level"), len(order_map))
        )
    try:
        return sorted(level_results, key=lambda res: res.get("level"))
    except TypeError:
        # Non-comparable level identifiers, keep input order
        return list(level_results)


def select_stage_two_budget(
    level_results: Sequence[Mapping[str, Any]],
    max_delta_pp: float = 0.3,
    level_order: Sequence[Any] | None = None,
) -> dict[str, Any] | None:
    """
    Choose the smallest data level whose EER improvement over the next level
    is within the allowed threshold and whose 95% CIs overlap.

    Each element of `level_results` must expose: {"level": <sortable>, "eer": float, "eer_ci": (low, high)}.
    """
    if not level_results:
        return None

    ordered = _order_level_results(level_results, level_order)
    selected = ordered[-1]
    improvement_pp: float | None = None
    compared_to: Any = None

    for lower, higher in zip(ordered, ordered[1:]):
        if lower.get("eer") is None or higher.get("eer") is None:
            continue
        ci_lower, ci_higher = lower.get("eer_ci"), higher.get("eer_ci")
        if not confidence_intervals_overlap(ci_lower, ci_higher):
            continue

        improvement_pp = (lower["eer"] - higher["eer"]) * 100
        compared_to = higher.get("level")
        if improvement_pp <= max_delta_pp:
            selected = lower
            break
        improvement_pp = None
        compared_to = None

    return {
        "selected": selected,
        "compared_to": compared_to,
        "improvement_pp": improvement_pp,
        "ordered_results": ordered,
    }
