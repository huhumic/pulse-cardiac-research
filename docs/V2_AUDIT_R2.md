# V2/V3 R2 — second-pass audit and acceptance ledger

**Scope:** feature/cine-temporal-reliability-v2 only. Baseline main and ongoing V1 experiments must not be modified. No actual Kaggle ACDC cine training has been executed in this review session.

## Changes tied to the earlier review

| Finding | R2 correction | Evidence / caveat |
|---|---|---|
| Past/future feature streams unordered | Direction embeddings + signed ±1/T and ED-anchored target cyclic sin/cos time code | CPU swap test shows *capacity* for noncommutativity after a controlled nonzero fusion projection; requires learned-model counterfactual |
| 4D cine only checked by dimensions | 100-patient preflight verifies `NbFrame`, affine, GT geometry, ED/ES frame-to-index NCC; detects wrong and ambiguous indexing | Fake-NIfTI regression tests; real original ACDC preflight remains pending |
| Loss described as unlabeled-target learning | Precisely defines probe agreement on labeled target ED/ES with unlabeled neighboring images as evidence | No intermediate-mask claim; any true unlabeled-target segmentation requires distinct work |
| Out-of-bound grid samples could be trusted | `warp_2d(..., return_valid=True)` with zero padding; invalid sampling points suppressed before 3-way softmax; masked JSD trust | CPU invalid warp tests and weight sum test; real grid drift needs diagnostics |
| `--no-consistency` turned off two losses | Independent `--no-consistency`, `--no-probe-supervision`, `--no-temporal`, `--no-structure` switches | Training code guard enforces temporal-free ablation; same PULSE supervised loss retained |
| No test for temporal use | `--temporal-control swap` and `repeat-target`, mean skip/past/future weights and valid fractions in evaluation report | Fixed-checkpoint controls only; need trained-model data |
| Evaluation provenance | Explicit `full_fold`, 20 patient internal validation warning, fold and checkpoint SHA256 | Five folds or untouched test dataset still needed for journal-grade conclusions |
| No true end-to-end run | 28 CPU tests across V1 and V2 contracts passed in isolated environment | Kaggle preflight, 3-step GPU forward/backward + single-case evaluation still required |

Additional R2.1 correction: **probe cross-entropy now masks invalidly warped pixels**;
previously the reliability gate and agreement term honored the sampling mask,
but probe CE still penalized out-of-bounds evidence. The masked loss returns
zero safely if no valid pixels exist. Verified with a dedicated regression test.

## Remaining boundaries (not claims of completeness)

1. The local environment lacks `nibabel`/ACDC/HF model weights. Preflight behavior was verified with a mock NIfTI data proxy and deliberately corrupted indexed frame; a *real* 100-patient Kaggle preflight is mandatory before training.
2. Temporal fusion adds access to 4D neighboring frames unavailable to plain single-frame PULSE; claims about comparative performance must disclose this input-information difference.
3. The time-fusion branch can choose to ignore neighbors. Direction-aware architecture does not ensure learned direction dependence. Compare `correct`/`swap`/`repeat-target` and inspect weights, Dice, HD95, ASSD under identical evaluation.
4. Target-frame JSD does not automatically establish reliable correspondence or unsupervised learning at mid-time frames. No learned offset ground truth is provided; avoid claiming registration accuracy.
5. Fold_0 validation is used to choose the best checkpoint. Do not label its 40 volumes as independent test samples. For formal research, aggregate genuinely out-of-fold patients across all folds or evaluate on a disjoint labeled benchmark.
6. Unit tests check interface and finite-valued gradients, not convergence, parameter fairness, temporal utilization, or publication novelty. Real GPU integration remains a blocker.

## R2 code acceptance state

- Static compilation: passed.
- Tests: 28 passed, targeting standalone V2/legacy V1 and exact PULSE patch contracts.
- Data: strict patient-level preflight implemented but not exercised on the user's mounted Kaggle dataset.
- Training: architecture source assembled but **not yet approved for full 120 epochs** until preflight and real GPU smoke run pass.
- Main/V1: no files touched by this branch.
