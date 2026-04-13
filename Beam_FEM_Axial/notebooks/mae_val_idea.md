Yes, you can see it clearly in those curves. The tell is the scale ratio:

  - val_mae_U3 converges around 0.5–0.6 mm
  - val_mae_U1 converges around 0.03–0.04 mm

  U3 is ~15× larger error than U1. That looks like the model is learning U3 well and U1 poorly — but it's actually the opposite problem: U3 is
   the bending component, so 0.5 mm MAE on a field that reaches 90 mm is fine (~0.5%). But U1 is the axial component, where the true values
  are 0.001–1 mm, so 0.03 mm MAE is catastrophic for low-force cases.

  The dead giveaway you'd look for after the Y_mean=0 fix is:

  ┌─────────────────────────┬─────────────────────────────────────────────────────┬─────────────────────────────────────────────────┐
  │         Signal          │                     Before fix                      │                    After fix                    │
  ├─────────────────────────┼─────────────────────────────────────────────────────┼─────────────────────────────────────────────────┤
  │ val_mae_U3 for X+ cases │ artificially inflated (model outputs Y_mean[2] ≠ 0) │ drops toward 0                                  │
  ├─────────────────────────┼─────────────────────────────────────────────────────┼─────────────────────────────────────────────────┤
  │ val_mae_U1 for Z cases  │ artificially inflated                               │ drops toward 0                                  │
  ├─────────────────────────┼─────────────────────────────────────────────────────┼─────────────────────────────────────────────────┤
  │ U3/U1 MAE ratio         │ ~15×                                                │ should collapse toward the true stiffness ratio │
  └─────────────────────────┴─────────────────────────────────────────────────────┴─────────────────────────────────────────────────┘

  What you cannot see from these curves alone is the direction confusion — because they average over all load directions together. A
  val_mae_U3 = 0.5 mm hides the fact that for X+ cases specifically, U3 should be 0 but the model was predicting Y_mean[2] ≈ 20 mm.

  To properly diagnose it after retraining, you'd want val_mae_U3 split by load direction — if the fix works, U3 for X+ cases should be near
  zero.

❯ can we enforce this metric to be in pourcetage soo like you said ?

● Good idea. The fix is to compute MAE per direction, per component, as % of the true RMS for that direction during the validation loop. That
  way you can see both:
  - the active component accuracy (U3 for Z, U1 for X+/X-)
  - the inactive component contamination (U3 for X+/X-) — after the fix this should collapse to ~0%

