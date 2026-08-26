# Learned belief baseline

## Scope

Stage 3A learns the mapping from observed dynamic response tracks to a 2-D
event posterior while leaving the strict-action planner unchanged:

```text
dual-view RGB-D
  -> existing RGB-D grounder and anonymous association
  -> source-blind track/candidate spatial evidence
  -> explicit 61 x 61 belief state
  -> existing analytical action planner
```

Two offline baselines are retained. `LearnedCueBeliefUpdater` predicts
interpretable cue direction, uncertainty, and reliability and adds their
likelihoods to the posterior. `SpatialRecurrentBeliefUpdater` instead uses a
pairwise track-cell encoder and belief-only Spatial ConvGRU, so it can preserve,
correct, or replace an earlier hypothesis. Both are entity-token upper
baselines, not end-to-end perception models.

Stage 3B additionally runs a trained Spatial RNN in an online lockstep episode.
It leaves the Stage 2 expert entry point unchanged and composes the RNN posterior
with a shared fixed `BeliefNavigationController`.

## Input contract

For every grounded track and frozen observation, the loader provides:

- semantic class (`FIRE_TRUCK`, `PEDESTRIAN`, or `UNKNOWN` for Spatial RNN input);
- start-local 3-D position recovered from RGB-D;
- measured horizontal motion between exactly adjacent observations;
- projected bounding-box location/size and named view;
- continuous evidence age and same-class count.

Stable track IDs are used only to reconstruct adjacent motion; the ID itself
is never a feature. The following privileged/generated fields are forbidden
model inputs:

- event coordinates and evaluation truth;
- `FIRE_SOURCE` tracks for the Spatial RNN;
- event affiliation, GTA velocity, and GTA task state;
- teacher `motion_evidence` and `inferred_event_direction`;
- GTA handles and world camera/view matrices.

The Spatial RNN also cannot read camera pose. Coordinates have already been
deterministically transformed into the start-local frame by the data pipeline.
The event cell is the sole training target. The recorded teacher belief is
never a training target, input, or checkpoint-selection criterion.

## Source-blind supervision

The loader rejects episodes without a non-empty interval from the first valid
non-source motion cue through the step immediately before the first grounded
`FIRE_SOURCE`. Event-cell NLL is averaged over that interval inside each
episode, then averaged across episodes. Belief is frozen before the first cue,
at and after first source visibility, during padding, and whenever an update
step has no valid motion evidence.

## Incremental baseline

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
only temporal state. The old terminal-only checkpoint contract remains
readable. Fair comparison to the Spatial RNN requires retraining with
`--supervision inference`.

## Spatial RNN baseline

For every valid moving track and candidate cell, a shared MLP encodes normalized
distance, motion-relative cosine/sine, displacement, relative height, bounding
box span, named view, evidence age, count, and a categorical embedding. Each
cell aggregates tracks with `mean + max + log-count` and projects the result to
a 16-channel evidence map. Track order therefore has no semantic meaning.

The recurrent step receives only the previous one-channel log-belief, current
evidence map, and valid-grid mask. Ephemeral convolutions predict reset/update
gates and a complete candidate log-belief. The result is normalized over the
circular grid after every update. There is no recurrent hidden tensor alongside
the belief map, and downstream components may consume only normalized belief.
D4 rotations/reflections are applied jointly to training tracks, motion,
teacher diagnostics, and event cell. Validation data is not augmented.

Checkpoint selection uses held-out-anchor inference-window NLL. Reports include
the whole inference interval and its final source-blind step: NLL, MAP error,
event-cell rank, top-10/50/100 recall, entropy, and 50/80/90 percent credible
region coverage and area. Teacher KL is explicitly diagnostic.


## Online runtime

`StreamingGroundedTrackEncoder` is shared by the Dataset and online runtime.
Its state contains only the immediately previous anonymous tracks, evidence
ages, and source boundary. `OnlineSpatialBeliefRuntime` calls `forward_step`
once per observation and owns one recurrent tensor: the normalized one-channel
log-belief. It does not replay the sequence prefix.

The online agent is split into strict boundaries:

```text
visibility-assisted RGB-D grounder
  -> source-blind streaming features
  -> Spatial RNN belief
  -> shared fixed navigation controller
  -> one strict action
```

The controller sees normalized belief, odometry, grounded tracks, action limits,
and static collision queries. It cannot read scenario snapshots, event truth,
visibility target lists, teacher belief, evidence maps, or recurrent gates.
`FIRE_SOURCE` is handled only by the independent two-observation source
confirmation/approach/STOP logic. The RNN belief freezes at its first grounded
appearance.

`shadow` mode retains the hand-written Stage 2 belief/controller for actions
while recording the RNN posterior. `control` mode replaces only the belief
backend; the fixed action policy is shared. The current planar posterior does
not learn altitude control, so ascent/descent remain bounded heuristics.

## Commands

```powershell
python -m pip install -r learning\requirements.txt

python validation\validate_learned_belief.py dataset\stage2e_multi_anchor

python learning\train_belief_updater.py `
  dataset\stage2e_multi_anchor `
  --supervision inference `
  --output learning\checkpoints\stage3_incremental_inference.pt

python learning\train_spatial_belief.py `
  dataset\stage2e_multi_anchor `
  --output learning\checkpoints\stage3_spatial_rnn.pt

python learning\evaluate_spatial_belief.py `
  dataset\stage2e_multi_anchor `
  learning\checkpoints\stage3_spatial_rnn.pt

python learning\compare_belief_models.py `
  dataset\stage2e_multi_anchor `
  learning\checkpoints\stage3_incremental_inference.pt `
  learning\checkpoints\stage3_spatial_rnn.pt

python validation\validate_spatial_belief.py dataset\stage2e_multi_anchor
```

Online feature/step parity and GTA execution:

```powershell
python validation\validate_online_spatial_belief_offline.py `
  dataset\stage2e_5x20_v4 --devices auto

python validation\validate_online_spatial_belief.py `
  --anchor 129.64151 -9.242669 80.02359 `
  --checkpoint learning\checkpoints\stage3_spatial_rnn.pt `
  --mode control --episodes 1 --max-steps 65 --device cuda
```

Checkpoints record the exact semantic classes, feature layout, grid, and
train/validation anchor lists, supervision boundary, model dimensions, and
training hyperparameters. Evaluation rejects incompatible contracts. Checkpoint
directories are Git-ignored.

Learned control does not treat a numerically maximal grid cell as a reliable
navigation target while the posterior is still diffuse. `FOLLOW_BELIEF` is
enabled only when both conditions hold:

- entropy is at most `6.5`;
- the 80% credible region is at most `8000 m2`.

Before that boundary, the controller may perform its finite cue scan, motion
confirmation, altitude transition, or `HOLD`, but it cannot plan toward the
single MAP cell. Shadow mode and the Stage 2 expert do not use this gate.
The thresholds and decision reason are written into every online recording.

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
