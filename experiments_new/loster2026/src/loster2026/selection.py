"""Audit 4's prospective numeric selection gates; no automatic scientific freeze."""
import math
import statistics
from .config import VARIANTS, PILOT_DATASETS, PILOT_SEEDS, planned_runs

def summarize(records):
    expected = {(r["dataset"], r["variant"], r["seed"]) for r in planned_runs()}
    by_key = {}
    for row in records:
        key = (row["dataset"], row["variant"], row["seed"])
        if key not in expected or key in by_key:
            raise ValueError("Unknown or duplicate pilot result")
        by_key[key] = row
    missing = sorted(expected - set(by_key))
    if missing:
        return {"status": "INCOMPLETE", "missing": missing, "numerical_recommendation": None}
    def usable(row):
        metrics = row.get("metrics", {})
        return (row.get("complete") is True and row.get("finite") is True and
                all(math.isfinite(metrics.get(m, float("nan"))) for m in ("ARI", "NMI_arithmetic", "RI", "ACC")) and
                math.isfinite(row.get("fit_seconds", float("nan"))) and row["fit_seconds"] > 0)
    legacy = VARIANTS[0]
    if not all(usable(by_key[(d, legacy, s)]) for d in PILOT_DATASETS for s in PILOT_SEEDS):
        return {"status": "INVALID_LEGACY_CONTROL", "numerical_recommendation": None}
    def values(variant):
        rows = [by_key[(d, variant, s)] for d in PILOT_DATASETS for s in PILOT_SEEDS]
        stats = {}
        for d in PILOT_DATASETS:
            part = [by_key[(d, variant, s)] for s in PILOT_SEEDS]
            stats[d] = {"ARI_mean": statistics.mean(r["metrics"]["ARI"] for r in part),
                        "ARI_sd": statistics.stdev(r["metrics"]["ARI"] for r in part),
                        "NMI_mean": statistics.mean(r["metrics"]["NMI_arithmetic"] for r in part),
                        "metrics": {m: {"mean": statistics.mean(r["metrics"][m] for r in part),
                                        "sample_sd_ddof1": statistics.stdev(r["metrics"][m] for r in part)}
                                    for m in ("ARI", "NMI_arithmetic", "RI", "ACC")}}
        return {"datasets": stats, "macro_ARI": statistics.mean(s["ARI_mean"] for s in stats.values()),
                "mean_ARI_sd": statistics.mean(s["ARI_sd"] for s in stats.values()),
                "collapsed_fits": sum(bool(r.get("collapse_original") or r.get("collapse_augmented")) for r in rows),
                "median_fit_seconds": statistics.median(r["fit_seconds"] for r in rows)}
    base = values(legacy)
    evaluated, eligible = {}, []
    for variant in VARIANTS[1:]:
        if not all(usable(by_key[(d, variant, s)]) for d in PILOT_DATASETS for s in PILOT_SEEDS):
            evaluated[variant] = {"eligible": False, "reason": "incomplete/nonfinite alternative"}
            continue
        item = values(variant)
        delta = {d: item["datasets"][d]["ARI_mean"] - base["datasets"][d]["ARI_mean"] for d in PILOT_DATASETS}
        mean_delta = statistics.mean(delta.values())
        nmi_delta = statistics.mean(item["datasets"][d]["NMI_mean"] - base["datasets"][d]["NMI_mean"] for d in PILOT_DATASETS)
        new_collapse, eliminated, eliminated_tasks = False, 0, set()
        for d in PILOT_DATASETS:
            for s in PILOT_SEEDS:
                a, b = by_key[(d, variant, s)], by_key[(d, legacy, s)]
                for field in ("collapse_original", "collapse_augmented"):
                    new_collapse |= bool(a.get(field) and not b.get(field))
                if (b.get("collapse_original") or b.get("collapse_augmented")) and not (
                        a.get("collapse_original") or a.get("collapse_augmented")):
                    eliminated += 1
                    eliminated_tasks.add(d)
        sd_reduction = {d: base["datasets"][d]["ARI_sd"] - item["datasets"][d]["ARI_sd"] for d in PILOT_DATASETS}
        quality = mean_delta >= 0.01 and sum(v > 0 for v in delta.values()) >= 6 and sum(v >= 0.01 for v in delta.values()) >= 3
        stability = mean_delta >= -0.01 and (
            (sum(v > 0 for v in sd_reduction.values()) >= 6 and statistics.mean(sd_reduction.values()) >= 0.01) or
            (eliminated >= 2 and len(eliminated_tasks) >= 2))
        safety = not new_collapse and min(delta.values()) >= -0.03 and nmi_delta >= -0.01
        dominated = (item["macro_ARI"] <= base["macro_ARI"] and
                     item["mean_ARI_sd"] >= base["mean_ARI_sd"] and
                     item["collapsed_fits"] >= base["collapsed_fits"] and
                     item["median_fit_seconds"] >= base["median_fit_seconds"] and
                     (item["macro_ARI"] < base["macro_ARI"] or
                      item["mean_ARI_sd"] > base["mean_ARI_sd"] or
                      item["collapsed_fits"] > base["collapsed_fits"] or
                      item["median_fit_seconds"] > base["median_fit_seconds"]))
        item.update(eligible=safety and (quality or stability) and not dominated,
                    delta_by_dataset=delta, macro_delta=mean_delta, macro_NMI_delta=nmi_delta,
                    positive_datasets=sum(v > 0 for v in delta.values()),
                    quality_route=quality, stability_route=stability,
                    safety_gate=safety, dominated=dominated)
        evaluated[variant] = item
        if item["eligible"]:
            eligible.append(variant)
    choice = legacy
    if len(eligible) == 1:
        choice = eligible[0]
    elif len(eligible) == 2:
        a, b = eligible
        x, y = evaluated[a], evaluated[b]
        if x["collapsed_fits"] != y["collapsed_fits"]:
            choice = a if x["collapsed_fits"] < y["collapsed_fits"] else b
        elif abs(x["macro_ARI"] - y["macro_ARI"]) >= 0.01:
            choice = a if x["macro_ARI"] > y["macro_ARI"] else b
        elif abs(x["mean_ARI_sd"] - y["mean_ARI_sd"]) >= 0.01:
            choice = a if x["mean_ARI_sd"] < y["mean_ARI_sd"] else b
        elif x["positive_datasets"] != y["positive_datasets"]:
            choice = a if x["positive_datasets"] > y["positive_datasets"] else b
        elif max(x["median_fit_seconds"], y["median_fit_seconds"]) / min(x["median_fit_seconds"], y["median_fit_seconds"]) >= 1.05:
            choice = a if x["median_fit_seconds"] < y["median_fit_seconds"] else b
        else:
            choice = VARIANTS[1]
    return {"status": "COMPLETE", "legacy": base, "alternatives": evaluated,
            "numerical_recommendation": choice,
            "conceptual_hypothesis_review_required": True, "method_automatically_frozen": False,
            "threshold_sensitivity": "Describe 0/.005/.02 mean-gain sensitivity without changing registered decision"}
