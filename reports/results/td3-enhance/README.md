# CALF-Enhance TD3 reproduction evidence

The wrapper launches the CleanRL TD3 trainer copied from the pinned
`aidagroup/calf-enhance@afb5edc49427054c99d6fbfe87b603d126724eb8`
source tree with its frozen dependency lock. The local integration changes add
checkpoint persistence and allow execution from a vendored, non-Git directory;
the TD3 update and environment logic are unchanged.

## Historical robot run

`robot-seed1-50k.json` compares a new 50,000-step robot seed-1 run against the
historical CALF-Enhance run `89aca0e282ec4363bf29c1679b71f01b` from commit
`97cb5804c8f4d81ac63e13b607a3d7f094d246c4`. All 18 deterministic metric
histories are exactly equal through step 49,999: 78 episode points for each
episode/environment metric and 249 points for each actor/critic loss metric.
Only `charts/SPS`, which measures wall-clock throughput, is excluded.

`robot-seed1-checkpointed-50k.json` repeats the same comparison while saving a
checkpoint every 10,000 steps in run
`d75e31eee4e049f9b7adc86d14fcf27b`. All 18 histories (2,430 metric points)
remain exactly equal, verifying that checkpoint persistence does not change
the training trajectory.

## Underwater deterministic reset

The historical underwater run was produced before CALF-Enhance fixed seeded
environment resets and is therefore not a valid exact numerical target. The new
matrix uses the pinned post-fix revision. `underwater-seed1-repeat-5k.json`
compares two independent runs of that revision on the same GPU. All eight
available deterministic histories, comprising 24 metric points, are exactly
equal. A 50,000-step repeat should be completed before promoting the
underwater configuration into the final paper experiment set.
