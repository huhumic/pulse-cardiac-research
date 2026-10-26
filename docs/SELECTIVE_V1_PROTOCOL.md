# V1 — selective anatomical feature enhancement (research experiment)

## Scope
PULSE is kept as the 2.5D **spatial** cardiac MRI segmentation baseline. V1
only modifies two high-resolution skip-fusion operations of its DPT decoder
using original, lightweight implementations inspired by the general multi-scale
edge and context ideas in CENet (MICCAI 2025). The new residual branch has an
all-zero output projection, so V1 must exactly equal loaded PULSE at step zero.

No cardiac-time input, DCR-MI-generated image, unlabelled supervision or
inference postprocessing change is used in V1. Temporal modeling is deferred to
V2 so scientific claims about time cannot be conflated with spatial fusion.

## Preserved
- Original PULSE DINOv2 backbone, original DPT decoder layers/heads and 2.5D
  adjacent-**spatial**-slice input.
- Original normalization / segmentation labels RV=1, Myo=2, LV=3.
- ACDC 100 training patients; disease-stratified seed=42, fold=0 with
  **80 train patients and 20 held-out validation patients**.
- Original four-flip TTA and postprocessing for official-mode validation.
- No patient masks or pretrained network weights added to Git.

## Modification
At fusion points using intermediate hidden features 1 and 0:
1. Extract multi-scale feature detail with `skip-avgpool3(skip)` and
   `skip-avgpool5(skip)` plus depthwise filtering.
2. Extract dilated decoder context from the current decoder feature.
3. Predict per-pixel, per-channel sigmoid gate from `[decoder, skip]`.
4. Add **zero-initialized gated residual update** at the existing summation.

This is not a direct insertion of the original CENet DSEB/CFAM code and does not
make any claim of novel or superior temporal modeling.

## Optimisation
- Initialise from an official `fold_0` checkpoint (or user's trained PULSE
  fold=0 checkpoint) **only after that checkpoint is complete**.
- Keep original PULSE backbone, decoder and auxiliary heads frozen and in
  `eval()` mode. Optimise only new adapters with AMP and AdamW.
- Use the original supervised ED/ES data and main PULSE segmentation loss;
  frozen auxiliary heads do not receive training gradients.
- 1 epoch / 3 steps is a smoke test, **not** an effectiveness experiment.
- Initial research run: 15 epochs with batch size 4; only adjust after data.

## Baseline and evaluation
Original **published fold_0 checkpoint**, patient-averaged full-volume Dice,
four-flip TTA, 20 validation patients / 40 ED+ES volumes:

| RV | Myo | LV | Mean | Mean time |
| --- | --- | --- | --- | --- |
| 0.907239 | 0.843840 | 0.917309 | 0.889463 | 2.28s/volume |

The training `val_mean` is the **PULSE sampled-slice** metric and must not be
mixed with full-volume Dice; always use `pulse_repro.evaluate` with
`--model-variant selective-v1 --tta` for comparable research results.

## For the DCR-MI-to-segmentation thesis connection (V2/V3)
ACDC ED/ES masks are the only GT used here. Its full 4D cine time axis is not
equivalent to 2.5D spatial neighbors. V2 must read verified true neighboring
**cardiac time frames**, make the time-gap/time-direction explicit, and perform
a patient-disjoint ablation of temporal reliability gating. V3 could use
DCR-MI-generated intermediate cine frames only when their training and
evaluation patient provenance supports leakage-free testing. Do not attach
generated frames with pseudo-GT as if they were human-labelled.
