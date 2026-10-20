"""ACDC PULSE: strict patient-paired 3D Dice/HD95/ASSD fold auditing and reporting.

Usage: python -m pulse_repro.experiment_report {freeze-split,audit,aggregate,compare} --help
Does not load the network, run inference or require a GPU.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .experiment_protocol import (ProtocolError, build_split_manifest,
    load_fold, paired_compare, pooled_summary)


def _write(output_dir: Path, name: str, obj: dict) -> None:
    path = output_dir / name
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print("Wrote", path)


def _write_csv(output_dir: Path, name: str, frame) -> None:
    path = output_dir / name
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index_label="patient", float_format="%.10g")
    print("Wrote", path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sp = ap.add_subparsers(dest="operation", required=True)
    freeze = sp.add_parser("freeze-split", help="Freeze original PULSE 100-patient split; no training")
    freeze.add_argument("--acdc-root", type=Path, required=True)
    freeze.add_argument("--pulse-root", type=Path, default=Path("/kaggle/working/PULSE"))
    freeze.add_argument("--out-dir", type=Path, required=True)

    audit = sp.add_parser("audit", help="Validate one 20-patient full-fold inference report")
    audit.add_argument("--eval-dir", type=Path, required=True)
    audit.add_argument("--split-manifest", type=Path)
    audit.add_argument("--allow-counterfactual", action="store_true")

    agg = sp.add_parser("aggregate", help="Aggregate 1 to 5 nonoverlapping validation folds")
    agg.add_argument("--eval-dirs", type=Path, nargs="+", required=True)
    agg.add_argument("--out-dir", type=Path, required=True)
    agg.add_argument("--split-manifest", type=Path)
    agg.add_argument("--allow-counterfactual", action="store_true")

    comp = sp.add_parser("compare", help="Paired reference/candidate by patient, phase and fold")
    comp.add_argument("--reference", type=Path, nargs="+", required=True)
    comp.add_argument("--candidate", type=Path, nargs="+", required=True)
    comp.add_argument("--out-dir", type=Path, required=True)
    comp.add_argument("--split-manifest", type=Path)
    comp.add_argument("--allow-counterfactual", action="store_true")
    comp.add_argument("--allow-undefined-surface", action="store_true")
    comp.add_argument("--bootstrap", type=int, default=10000)
    comp.add_argument("--seed", type=int, default=20261021)
    args = ap.parse_args(argv)

    if args.operation == "freeze-split":
        manifest = build_split_manifest(args.acdc_root, args.pulse_root)
        _write(args.out_dir, "acdc_fivefold_manifest.json", manifest)
        return 0

    manifest = None
    if args.split_manifest is not None:
        manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    allowed = getattr(args, "allow_counterfactual", False)
    def load(paths):
        return [load_fold(p, allow_counterfactual=allowed, split_manifest=manifest) for p in paths]

    if args.operation == "audit":
        r = load([args.eval_dir])[0]
        print(json.dumps({"audit":"passed", "fold":r.fold,
                          "variant":r.variant, "temporal_control":r.control,
                          "n_patients":len(r.patients),"n_volumes":len(r.phases),
                          "Mean_Dice":float(r.patients["Mean_Dice"].mean()),
                          "warning":"Internally selected validation fold, not independent test"},indent=2))
    elif args.operation == "aggregate":
        data, report = pooled_summary(load(args.eval_dirs))
        _write(args.out_dir, "aggregate_summary.json", report)
        _write_csv(args.out_dir, "aggregate_per_patient.csv", data)
        import pandas as pd
        table = pd.DataFrame([{"metric":k, **v} for k,v in report["metrics"].items()])
        _write_csv(args.out_dir, "table_aggregate_metrics.csv", table.set_index("metric"))
    elif args.operation == "compare":
        per_patient, report = paired_compare(
            load(args.reference), load(args.candidate),bootstrap_reps=args.bootstrap,
            seed=args.seed,allow_undefined_surface=args.allow_undefined_surface)
        _write(args.out_dir, "paired_summary.json", report)
        _write_csv(args.out_dir, "paired_per_patient_delta.csv", per_patient)
        import pandas as pd
        table = pd.DataFrame([
            {"metric":k, "reference_mean":v["reference_mean"],
             "candidate_mean":v["candidate_mean"],
             "delta":v["delta_candidate_minus_reference"],
             "delta_ci_low":v["delta_95ci"][0],"delta_ci_high":v["delta_95ci"][1],
             "benefit_positive":v["benefit_positive"],
             "n_complete_pairs":v["n_complete_pairs"],
             "n_missing_pairs":v["n_missing_pairs"]}
            for k,v in report["metrics"].items()])
        _write_csv(args.out_dir, "table_paired_metrics.csv", table.set_index("metric"))
        mean_dice = report["metrics"]["Mean_Dice"]
        print("Patient-paired Mean Dice difference: ",mean_dice["delta_candidate_minus_reference"],
              "95% CI",mean_dice["delta_95ci"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
