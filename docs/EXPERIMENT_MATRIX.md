# PULSE second-paper experimental matrix (pre-registered planning table)

| ID | Network / input | M1 structure | M2 temporal reliability | M3 agreement | Probe CE | Main question | Status |
|---|---|---|---|---|---|---|---|
| B0 | Self-trained PULSE, 2.5D spatial | no | no | no | no | Fully reproduced architecture baseline | Fold0 120-epoch trained, 3D evaluated |
| B1 | PULSE + V1 joint | yes | no | no | no | Spatial enhancement contribution | Fold0 training underway; final evaluation pending |
| T0 | PULSE + temporal concatenation | no | basic concatenation | no | no | Is extra image input alone sufficient? | Not implemented |
| T1 | PULSE + M2, rejection disabled | no | yes, no reject | no | no | Is reliability decision useful? | Not implemented |
| T2 | PULSE + M2 | no | yes | no | no | Standalone temporal mechanism | Training candidate after ACDC preflight |
| T3 | PULSE + M2 + probes | no | yes | no | yes | Does target-frame probe supervision help? | Available variant candidate |
| T4 | PULSE + M2 + M3 | no | yes | yes | yes | Added reliability-guided agreement | Available variant candidate |
| Full | PULSE + M1 + M2 + M3 | yes | yes | yes | yes | Joint benefit | Available code, not yet trained |
| C1 | Full, neighbor images swapped | same checkpoint | direction images swapped | same | same | Direction/time causal diagnostic | Evaluator supports; not trained |
| C2 | Full, neighbor images replaced by target | same checkpoint | no correct temporal evidence | same | same | Is model exploiting temporal content? | Evaluator supports; not trained |
| X1 | nnU-Net v2, matched fold | different architecture | no | no | no | Strong baseline | To reproduce |
| X2 | CENet, matched fold | alternative anatomy module | no | no | no | Structural related work | To reproduce |
| X3 | CSTM or other cardiac-time method | may differ | temporal | may differ | varies | Temporal related work | Needs confirmed code/data/task compatibility |

**Non-negotiable:** No direct numeric ranking against papers using 70/10/20, public leaderboard, other HD95 implementations, or different patient splits. For temporal vs spatial-only methods disclose input information differences. For model selection note 20 validation patients per fold, best epoch selected there; not untouched test.

## Required output tables

1. Table 1 — ACDC fivefold patient-weighted RV/Myo/LV Dice, Mean Dice, HD95, ASSD; mark nonmatching inputs.
2. Table 2 — M1/M2/M3 and probe CE ablation, independent training for each variant.
3. Table 3 — M2 naïve fusion vs aligned vs rejection enabled; compare same extra inputs.
4. Table 4 — temporal controls (correct / swap / repeat-target), matched checkpoint and patient IDs, reliability weights; not a standard trained-model leaderboard.
5. Table 5 — external cine dataset, only after target labels/time series and transformation are verified.
6. Table 6 — parameters, inference latency including the additional temporal input, TTA, GPU, VRAM and FLOPs if measured.

## Promotion gate to full fivefold runs

- 4D cine of all required ACDC patients verified against ED/ES images (no fake time frames).
- Single-fold Kaggle forward/backward, strict checkpoint reload and 40-volume inference succeed.
- Mean Dice and surface metrics are compared with B0 on the same 20 patients.
- M2 evidence truly affects predictions; M3 and reject gate individually ablated.
- Compute budget and expected gain justify running the remaining four folds.
