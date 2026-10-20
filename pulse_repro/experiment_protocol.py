"""Fail-closed evaluation protocol for PULSE ACDC patient-level fold reports.

All metrics are reconstructed from ED/ES *3D-volume* outputs. Fold checkpoints
were selected using fold validation: these are internally selected OOF-like
cross-validation results, NOT untouched external test outcomes.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CLASS_NAMES = ("RV", "Myo", "LV")
METRICS = tuple(f"{name}_{metric}" for name in CLASS_NAMES
                for metric in ("dice", "hd95_mm", "assd_mm")) + ("Mean_Dice",)
PHASES = frozenset(("ED", "ES"))
STATUSES = frozenset(("valid", "one_empty", "both_empty"))


class ProtocolError(ValueError):
    """Refuse to generate a statistical report for noncomparable runs."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProtocolError(message)


def _plain_json(path: Path) -> dict:
    if not path.is_file():
        raise ProtocolError(f"Missing {path}")
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ProtocolError(f"Invalid JSON {path}: {e}") from e
    _require(isinstance(record, dict), f"Expected JSON object in {path}")
    return record


def _validate_surface(df: pd.DataFrame, name: str) -> None:
    status = f"{name}_status"
    _require(status in df.columns, f"Missing {status}; cannot audit undefined metrics")
    _require(df[status].notna().all(), f"Null {status}")
    values = set(df[status].tolist())
    _require(values <= STATUSES, f"Unexpected {status}: {values - STATUSES}")
    d = pd.to_numeric(df[f"{name}_dice"], errors="coerce").to_numpy(dtype=float)
    h = pd.to_numeric(df[f"{name}_hd95_mm"], errors="coerce").to_numpy(dtype=float)
    a = pd.to_numeric(df[f"{name}_assd_mm"], errors="coerce").to_numpy(dtype=float)
    valid = df[status].eq("valid").to_numpy()
    one_empty = df[status].eq("one_empty").to_numpy()
    both_empty = df[status].eq("both_empty").to_numpy()
    _require(np.isfinite(d[valid | one_empty]).all(), f"Invalid {name} Dice for nonempty GT/pred")
    _require(((d[valid | one_empty] >= 0) & (d[valid | one_empty] <= 1)).all(),
             f"{name} Dice not in [0,1]")
    _require(np.isnan(d[both_empty]).all(), f"Expected undefined Dice for {name} both_empty")
    _require(np.isfinite(h[valid]).all() and np.isfinite(a[valid]).all(),
             f"Invalid {name} surface metric when status=valid")
    _require((h[valid] >= 0).all() and (a[valid] >= 0).all(),
             f"Negative {name} surface distance")
    _require(np.isnan(h[~valid]).all() and np.isnan(a[~valid]).all(),
             f"Undefined {name} surface distances must be NaN (not zero)")


@dataclass(frozen=True)
class FoldReport:
    directory: Path
    fold: int
    variant: str
    control: str
    tta: bool
    checkpoint_hash: str
    phases: pd.DataFrame  # (patient,phase) MultiIndex, numeric and status columns
    patients: pd.DataFrame  # indexed by patient, both phases equally weighted
    summary: dict


def load_fold(directory: Path | str, *, allow_counterfactual: bool = False,
              split_manifest: dict | None = None) -> FoldReport:
    directory = Path(directory)
    summary = _plain_json(directory / "summary.json")
    _require("fold" in summary and type(summary["fold"]) is int, "Missing integer fold")
    fold = summary["fold"]
    _require(0 <= fold < 5, f"Invalid fold {fold}")
    _require(summary.get("seed") == 42, "Only PULSE patient split seed=42 supported")
    _require(summary.get("tta") is True, "Comparable runs must use 4-flip TTA")
    variant = summary.get("model_variant", summary.get("variant"))
    _require(isinstance(variant, str) and bool(variant.strip()), "Missing model variant")
    control = summary.get("temporal_control", "correct")
    _require(control == "correct" or (allow_counterfactual and control in
             ("swap", "repeat-target")),
             f"Temporal control '{control}' is diagnostic; explicitly opt in")
    _require(summary.get("full_fold", False) is True,
             "Incomplete fold / smoke test: cannot be used for formal comparison")
    checkpoint_hash = summary.get("weights_sha256", summary.get("checkpoint_sha256"))
    _require(isinstance(checkpoint_hash, str) and len(checkpoint_hash) == 64 and
             all(c in "0123456789abcdefABCDEF" for c in checkpoint_hash),
             "Missing/invalid checkpoint SHA256 in summary")
    if (directory / "per_volume.csv").is_file():
        df = pd.read_csv(directory / "per_volume.csv", dtype={"patient": str, "phase": str})
    else:
        raise ProtocolError(f"Missing {directory / 'per_volume.csv'}")
    cols = {"patient", "phase", *[col for col in METRICS if col != "Mean_Dice"],
            *[f"{name}_status" for name in CLASS_NAMES]}
    _require(cols <= set(df.columns), f"Missing columns: {sorted(cols - set(df.columns))}")
    _require(len(df) == 40, f"Expected 40 ED/ES volumes, got {len(df)}")
    _require(df[["patient", "phase"]].notna().all().all(), "Missing patient/phase")
    _require(df.phase.isin(PHASES).all(), "Unexpected phase label")
    _require(not df.duplicated(["patient", "phase"]).any(), "Duplicate patient/phase")
    _require(df.patient.nunique() == 20, "Expected 20 unique patients per fold")
    patient_phases = df.groupby("patient")["phase"].agg(lambda x: frozenset(x))
    _require(patient_phases.map(lambda value: value == PHASES).all(), "Each patient must have one ED and one ES")
    expected = summary.get("val_patients", summary.get("validation_patients"))
    _require(isinstance(expected, list) and len(expected) == 20 and
             len(set(expected)) == 20 and set(df.patient) == set(expected),
             "CSV patient list disagrees with summary validation patient list")
    _require(summary.get("n_volumes") == 40, "Summary does not declare exactly 40 volumes")
    n_patients = summary.get("n_valid_patients", summary.get("n_patients"))
    _require(n_patients == 20, "Summary does not declare exactly 20 patients")
    if split_manifest is not None:
        _require(split_manifest.get("seed") == 42 and split_manifest.get("nfolds") == 5,
                 "Invalid split manifest protocol")
        expected_manifest = split_manifest.get("folds", {}).get(str(fold), {}).get("val_patients")
        _require(isinstance(expected_manifest, list) and set(expected_manifest) == set(df.patient),
                 "Evaluation patients disagree with frozen patient split manifest")
    for name in CLASS_NAMES:
        _validate_surface(df, name)
    df = df.set_index(["patient", "phase"]).sort_index()
    numeric = [metric for metric in METRICS if metric != "Mean_Dice"]
    for metric in numeric:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
    # Patient first: mean ED+ES, then macro-average RV/Myo/LV.
    patient_means = df[numeric].groupby(level="patient", sort=True).mean()
    patient_means["Mean_Dice"] = patient_means[[f"{c}_dice" for c in CLASS_NAMES]].mean(
        axis=1, skipna=False
    )
    stated = summary.get("per_patient_means")
    _require(isinstance(stated, dict), "Missing published per_patient_means summary")
    for metric in numeric:
        observed = float(patient_means[metric].mean())
        advertised = stated.get(metric)
        if np.isnan(observed):
            _require(advertised is None, f"Expected null summary for undefined {metric}")
        else:
            _require(advertised is not None and np.isfinite(float(advertised)) and
                     abs(observed - float(advertised)) < 1e-6,
                     f"Summary mismatch for {metric}: recomputed {observed}, published {advertised}")
    # Source evaluator's means skip invalid ED/ES surfaces. For formal paired
    # statistics we instead require *both* phases finite per patient; a single
    # valid phase must not silently stand in for a missing ED/ES measurement.
    for name in CLASS_NAMES:
        incomplete = (df[f"{name}_status"] != "valid").groupby(level="patient").any()
        patient_means.loc[incomplete[incomplete].index,
                          [f"{name}_hd95_mm", f"{name}_assd_mm"]] = np.nan
    counts = summary.get("undefined_surface_case_counts")
    _require(isinstance(counts, dict), "Missing undefined-surface counts")
    for name in CLASS_NAMES:
        observed = int((df[f"{name}_status"] != "valid").sum())
        _require(counts.get(name) == observed,
                 f"Undefined-surface status counts mismatch for {name}")
    return FoldReport(directory, fold, variant, control, True, checkpoint_hash,
                      df, patient_means, summary)


def _ensure_foldset(reports: list[FoldReport]) -> None:
    _require(len(reports) >= 1 and len(reports) <= 5, "Require 1..5 fold reports")
    folds = [r.fold for r in reports]
    _require(len(folds) == len(set(folds)), "Duplicate fold report")
    variants = {r.variant for r in reports}
    controls = {r.control for r in reports}
    _require(len(variants) == 1, "Do not mix method variants in one aggregate")
    _require(len(controls) == 1, "Do not mix temporal controls in one aggregate")
    all_pids = [pid for r in reports for pid in r.patients.index]
    _require(len(all_pids) == len(set(all_pids)),
             "Patient leakage across fold reports (duplicate validation patient)")
    if len(reports) == 5:
        _require(set(folds) == set(range(5)) and len(all_pids) == 100,
                 "Five folds must cover exactly 100 distinct ACDC patients")


def _stack(reports: list[FoldReport]) -> pd.DataFrame:
    _ensure_foldset(reports)
    frames = []
    for r in sorted(reports, key=lambda r: r.fold):
        frame = r.patients.copy()
        frame.insert(0, "fold", r.fold)
        frames.append(frame)
    return pd.concat(frames).sort_index()


def pooled_summary(reports: list[FoldReport]) -> tuple[pd.DataFrame, dict]:
    """Equal weight per patient; no implicit reweighting for slice counts."""
    data = _stack(reports)
    stats = {c: {"mean": None if not data[c].notna().any() else float(data[c].mean()),
                 "sd_between_patients": None if data[c].notna().sum() < 2 else float(data[c].std(ddof=1)),
                 "n_finite": int(data[c].notna().sum())} for c in METRICS}
    return data, {"scope": "five_fold_internal_cv" if len(reports) == 5 else "partial_internal_cv",
                  "warning": "Best checkpoints were chosen using these fold validation patients; not independent test",
                  "folds": sorted(r.fold for r in reports),
                  "n_patients": len(data), "n_volumes": 2 * len(data),
                  "variant": reports[0].variant, "temporal_control": reports[0].control,
                  "metric_definitions": "ED/ES volume-level 3D; equal patient weighting; mean three classes",
                  "metrics": stats,
                  "checkpoint_sha256_by_fold": {str(r.fold):r.checkpoint_hash for r in reports}}


def paired_compare(reference: list[FoldReport], candidate: list[FoldReport], *,
                   bootstrap_reps: int = 10000, seed: int = 20261021,
                   allow_undefined_surface: bool = False) -> tuple[pd.DataFrame, dict]:
    """Two-sided percentile CI of mean paired change; no inferred significance p-values.

    + improvement means candidate better: delta dice, or negative delta distance.
    Cluster resampling unit is patient; ED and ES are always paired within patient.
    """
    _require(bootstrap_reps >= 1000, "At least 1000 bootstrap resamples required")
    _ensure_foldset(reference);_ensure_foldset(candidate)
    _require({r.fold for r in reference} == {r.fold for r in candidate},
             "Reference/candidate fold IDs differ")
    ref, cand = _stack(reference), _stack(candidate)
    _require(ref.index.equals(cand.index),
             "Reference/candidate patients differ; cannot conduct paired test")
    _require((ref.fold == cand.fold).all(), "Reference/candidate patient fold assignment differs")
    _require(all(r.control == "correct" for r in reference+candidate) or
             all(r.variant == reference[0].variant for r in reference+candidate),
             "Counterfactual comparison must use the same model variant")
    for rr in reference:
        cc = next(x for x in candidate if x.fold == rr.fold)
        _require(rr.phases.index.equals(cc.phases.index),
                 f"Per-volume ED/ES keys do not match on fold {rr.fold}")
        if rr.control != cc.control:
            _require(rr.checkpoint_hash == cc.checkpoint_hash,
                     "Temporal counterfactuals require identical checkpoint SHA256")
    diffs = cand[list(METRICS)].subtract(ref[list(METRICS)]); diffs.insert(0, "fold", ref.fold)
    rng = np.random.default_rng(seed)
    metrics = {}
    for name in METRICS:
        val = diffs[name].to_numpy(dtype=float)
        finite = np.isfinite(val)
        if not finite.all() and (name == "Mean_Dice" or name.endswith("_dice") or
                                not allow_undefined_surface):
            raise ProtocolError(f"Missing paired observations for {name}: "
                                f"{int((~finite).sum())}; do not substitute zero")
        v = val[finite]
        _require(len(v) >= 2, f"Too few paired observations for {name}")
        draws = rng.integers(0, len(v), size=(bootstrap_reps, len(v)))
        sample_means = v[draws].mean(axis=1)
        low, high = np.quantile(sample_means, [0.025, 0.975])
        sign = -1.0 if name.endswith(("hd95_mm", "assd_mm")) else 1.0
        metrics[name] = {"reference_mean": float(ref[name].mean()),
                         "candidate_mean": float(cand[name].mean()),
                         "delta_candidate_minus_reference": float(v.mean()),
                         "delta_95ci": [float(low), float(high)],
                         "benefit_positive": float(sign * v.mean()),
                         "benefit_95ci": [float(sign * high), float(sign * low)] if sign < 0
                                         else [float(low), float(high)],
                         "n_complete_pairs": int(len(v)),
                         "n_missing_pairs": int((~finite).sum())}
    info = {"scope": "five_fold_internal_cv" if len(reference) == 5 else "single_or_partial_fold_internal_cv",
            "warning": "Paired patients also participated in checkpoint selection: not an independent test",
            "reference_variant":reference[0].variant,
            "candidate_variant":candidate[0].variant,
            "reference_control":reference[0].control,
            "candidate_control":candidate[0].control,
            "folds":sorted(r.fold for r in reference),
            "n_patients":len(ref),"n_volumes":2*len(ref),
            "bootstrap_unit":"patient (ED/ES kept together)",
            "bootstrap_reps":bootstrap_reps,"bootstrap_seed":seed,
            "ci_method":"paired nonparametric percentile bootstrap",
            "metrics":metrics}
    return diffs, info


def build_split_manifest(acdc_root: Path, pulse_root: Path) -> dict:
    """Reconstruct the *exact* upstream seed42 grouped fold assignment from ACDC."""
    from .runtime import file_sha256, upstream
    _, source = upstream(pulse_root)
    records = source.list_acdc(acdc_root / "training")
    ids = [r["pid"] for r in records]
    _require(len(records) == 200 and len(set(ids)) == 100,
             "Expected exactly 200 ED/ES records for 100 ACDC patients")
    for pid in set(ids):
        phases = {r["phase"] for r in records if r["pid"] == pid}
        _require(phases == PHASES, f"Missing ED or ES for {pid}")
    folds = {}
    for k in range(5):
        train, val = source.stratified_kfold_pids(records, 5, k, seed=42)
        _require(train.isdisjoint(val) and len(train) == 80 and len(val) == 20,
                 f"Unexpected fold {k} size or split leak")
        folds[str(k)] = {"train_patients": sorted(train), "val_patients": sorted(val)}
    vals = [p for f in folds.values() for p in f["val_patients"]]
    _require(set(vals) == set(ids) and len(vals) == len(set(vals)) == 100,
             "Fold validation sets must partition all 100 patients")
    canonical_records = [{k:r[k] for k in ("pid", "phase", "group")}
                         for r in sorted(records, key=lambda d:(d["pid"],d["phase"]))]
    record_digest = hashlib.sha256(json.dumps(canonical_records, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()
    upstream_source = pulse_root / "pulse" / "pulse_seg.py"
    return {"dataset": "ACDC_100_labeled_ED_ES", "split":"PULSE disease-stratified fivefold",
            "seed":42, "nfolds":5,
            "warning":"These 100 patients are used for train/validation folds, not the unlabeled public test data",
            "source_script_sha256":file_sha256(upstream_source),
            "patient_phase_group_sha256":record_digest,
            "folds":folds}
