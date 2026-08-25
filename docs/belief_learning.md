# Learned belief baseline

## Scope

The first Stage 3 model learns the mapping from observed dynamic response
tracks to a 2-D event posterior. It intentionally leaves the existing
strict-action planner unchanged:

```text
dual-view RGB-D
  -> existing RGB-D grounder and anonymous association
  -> learned cue direction / uncertainty / reliability
  -> explicit 61 x 61 belief fusion
  -> existing analytical action planner
```

This isolates the scientific question "can the event belief update be
learned?" from raw visual detection and action-policy failures. It is an
entity-token upper baseline, not the final end-to-end method.

This delivery trains and evaluates the posterior on recorded episodes. It does
not yet replace the online GTA expert inside `run_expert_episode`; wiring a
trained checkpoint to the unchanged analytical planner is the next integration
step after selecting a checkpoint on held-out anchors.

## Input contract

For every grounded track and frozen observation, the loader provides:

- semantic class (`FIRE_TRUCK`, `PEDESTRIAN`, `FIRE_SOURCE`, or `UNKNOWN`);
- start-local 3-D position recovered from RGB-D;
- measured horizontal motion between exactly adjacent observations;
- projected bounding-box location/size and named view;
- current start-local camera odometry and remaining horizon.

Stable track IDs are used only to reconstruct adjacent motion; the ID itself
is never a feature. The following privileged/generated fields are forbidden
model inputs:

- event coordinates and evaluation truth;
- event affiliation, GTA velocity, and GTA task state;
- teacher `motion_evidence` and `inferred_event_direction`;
- GTA handles and world camera/view matrices.

The event cell is the sole training target, and it is used only at the final
valid observation of each complete episode. The recorded teacher belief is
never a training target or model input; it remains available only as an
evaluation diagnostic.

## Model and fusion

An MLP over each semantic entity token predicts four interpretable values:

- a parallel component relative to measured motion;
- a perpendicular component relative to measured motion;
- angular uncertainty;
- evidence reliability.

The parallel component may become positive for a responder moving toward the
event or negative for a responder moving away. This sign is learned rather
than initialized from the hand-written fire-truck/pedestrian rule.

For track position `p`, learned unit event direction `d`, angular scale
`sigma`, reliability `w`, and candidate grid cell `x`, the evidence is:

```text
theta(x) = angle(normalize(x - p), d)
log L(x) = w * max(-0.5 * (theta(x) / sigma)^2, log(epsilon))
```

All track log likelihoods are summed and added to the previous log posterior,
then normalized over the circular 120 m map. The network cannot replace the
posterior through a hidden recurrent state; the explicit belief map is its
only temporal state.

Training minimizes only the negative log probability assigned to the true
event cell at the terminal observation of each complete episode. There is no
teacher-belief loss and no intermediate event-cell supervision. Because the
terminal posterior is produced by the explicit sequence of Bayesian updates,
its gradient still reaches cue predictions made at earlier observations.
Evaluation may report teacher KL, event NLL, belief MAP error, and event-cell
rank for all observations, observations after the first valid motion cue, and
the terminal observation on anchor-disjoint locations; teacher KL is diagnostic
only and does not affect checkpoint selection.

## Commands

```powershell
python -m pip install -r learning\requirements.txt

python validation\validate_learned_belief.py dataset\stage2e_multi_anchor

python learning\train_belief_updater.py `
  dataset\stage2e_multi_anchor `
  --output learning\checkpoints\stage3_belief_updater.pt

python learning\evaluate_belief_updater.py `
  dataset\stage2e_multi_anchor `
  learning\checkpoints\stage3_belief_updater.pt `
  --split validation
```

Checkpoints record the exact semantic classes, feature layout, grid, and
train/validation anchor lists. Evaluation rejects incompatible contracts.
Checkpoint directories are Git-ignored.

## Current limitation

The learned map is intentionally 2-D because the existing Stage 2 expert data
is overwhelmingly planar. It cannot validate altitude-aware belief or learned
`ASCEND`/`DESCEND` behavior. A future benchmark audit must deliberately add
vertical ambiguity, altitude-stratified starts and goals, 3-D observability,
and sufficient vertical expert actions before extending this baseline to a
voxel, layered, or factorized 3-D posterior.

The inspected validation anchor is also heavily responder-imbalanced: it has
`1657` valid-motion pedestrian tokens but only `2` valid-motion fire-truck
tokens. The current data can demonstrate learning the pedestrian-away relation,
but cannot support a reliable fire-truck cue conclusion without more coverage.

The categorical semantic embedding also cannot by itself produce the desired
open-vocabulary generalization to unseen response concepts. That later method
will require visual/language semantic features from a pretrained encoder while
preserving the same explicit belief bottleneck and intervention tests.
