# Validation guide

## Principles

Validators operate online against one GTA instance unless explicitly called an
offline visualizer. They consume capture payloads in memory and write nothing
unless an output or recording directory is supplied. Tests expose failures;
they do not return cached observations, retry old frames, weaken thresholds, or
fall back to realtime execution.

Press `F10` before online tests. If cleanup cannot complete, press `F11` in GTA
to reset the scenario, exit lockstep, restore time, stop the camera, and return
to the player.

## Camera and RGB-D

```powershell
python validation\validate_camera_lifecycle.py
python validation\validate_pose_control.py
python validation\validate_rgbd_yaw_sync.py
python validation\validate_rgbd_stability.py --count 1000
```

These cover camera activation/inactive errors, absolute pose application,
view-matrix agreement, alternating-yaw RGB/depth freshness, monotonic frame
IDs, finite metric depth, latency, and process-memory growth.

For collision rejection, face a nearby wall and run:

```powershell
python validation\validate_pose_control.py --expect-forward-collision 10
```

## Geometry and dual view

```powershell
python validation\validate_rgbd_pointcloud.py `
  --pixel-stride 4 --max-view-depth 200

python validation\validate_dual_view_rgbd.py `
  --anchor 234 324 100 --pairs 40
```

The point-cloud test rotates through eight headings and merges colored points
in memory. The dual-view test checks common simulation time and camera center,
the `-45°/-90°` axes, frame freshness, pose restoration, and depth validity.
Use `--pairs 1000` for stress testing or `--show-pointcloud` for interactive
geometry inspection.

## Lockstep clock

```powershell
python validation\validate_lockstep_clock.py --anchor 234 324 100
```

The full validator checks frozen world state, cumulative 250 ms advances,
clock overshoot, repeated enter/exit, scenario evolution, matched realtime and
lockstep dynamics, RGB-D availability, recovery, and memory growth.

## Fire scenario

```powershell
python validation\validate_fire_scenario.py --anchor 234 324 100
python validation\validate_fire_scenario.py `
  --anchor 234 324 100 --cycles 50
python validation\validate_fire_scenario.py `
  --anchor 234 324 100 --verify-seed-isolation
```

The scenario validator covers lifecycle, entity roles and stable IDs, response
activation, direction and progress, Reset, invalid requests, repeated cycles,
population control, and blueprint seed isolation. Formal collection should
also use `--require-clean-area`.

Audit fire-particle appearance independently of geometric task success:

```powershell
python validation\validate_fire_visual_coverage.py `
  --anchor 234 324 100 --show
```

It samples controlled radii, heights, and azimuths at one frozen simulation
time. The temporal-activity metric is diagnostic only and never becomes a
visibility label.

## Visibility and starts

```powershell
python validation\validate_visibility_starts.py `
  --anchor 234 324 100 --strata both --verify-determinism
```

Use `--show-starts` to inspect projected source, diagnostic envelope, and responder boxes.
Use `--queries 1000` to stress repeated frozen visibility queries without
saving payloads.

Validate the source-shadow batch protocols, pool invariants, persistence
digest, and optional scenario-seed isolation:

```powershell
python validation\validate_anchor_start_pool.py `
  --anchor 234 324 100 --verify-seed-isolation
```

The validator keeps all RGB-D and visibility data in memory. It checks the
120 m ray contract, the dense radius/AGL/azimuth grid (up to 160 retained
entries), source occlusion, goal views,
lockstep invariants, pool reload/tamper rejection, and atomic anchor-row
removal semantics.

## Spatiotemporal feasibility audit

```powershell
python validation\validate_spatiotemporal_feasibility.py `
  --anchor 234 324 100
```

This evaluation-only search audits strict discrete actions, cue continuity,
source-view reachability, horizon accounting, structural replay, and STOP. It
is not the dataset expert. A finite search failure means no witness was found
within that search, not mathematical unreachability.

Exploratory repeated witness search is available through
`find_stage2d_witness.py`. It is useful for diagnosing search coverage but is
not the formal benchmark acceptance test.

## Expert and dataset path

Validate one no-payload expert rollout:

```powershell
python validation\validate_stage2e_expert.py --anchor 234 324 100
```

Record a compact visual diagnostic only when needed:

```powershell
python validation\validate_stage2e_expert.py `
  --anchor 234 324 100 `
  --record-dir recordings\stage2e_validation

python validation\visualize_stage2e_trajectory.py `
  recordings\stage2e_validation
```

Generate and replay successful dataset episodes:

```powershell
# In manual camera mode, press F8 at each desired location first.
python validation\generate_stage2e_experts.py `
  --anchor-file "C:\path\to\Grand Theft Auto V\data\DroneSim_anchors.jsonl" `
  --scenes-per-anchor 1 `
  --starts-per-scene 5 `
  --max-attempts-per-scenario 15 `
  --output-dir dataset\stage2e_multi_anchor

python validation\visualize_stage2e_dataset.py `
  dataset\stage2e_multi_anchor --loop
```

`--max-steps` may audit a noncanonical budget explicitly. The chosen value is
shared by start-budget checking, teacher rollout, and strict execution. The
canonical benchmark default remains 80 including STOP.

## Stage 3A offline belief validation

```powershell
python validation\validate_spatial_belief.py dataset\stage2e_multi_anchor
```

The default run audits every schema-4 episode's source-blind boundary, source
and pose exclusion, track permutation invariance, exact identity updates,
normalization, D4 round trips, episode-balanced NLL, variable-length backward,
tiny overfit, checkpoint reload, and five full-data smoke epochs. Use
`--training-smoke-epochs 0 --overfit-steps 2` for a fast structural check.

After training the two checkpoints, `learning/compare_belief_models.py` reports
the uniform prior, inference-supervised incremental updater, and Spatial RNN on
the identical anchor split. It reports performance differences without making
one model winning an implementation acceptance condition. Teacher KL is always
diagnostic.

## Storage behavior

- ordinary validators: no payload files;
- point clouds and Matplotlib diagnostics: memory-only interactive windows;
- compact trajectory recording: JPEG RGB plus structured metadata, no depth;
- dataset generation: successful episode RGB-D only;
- failed dataset attempts: compact JSON diagnostics only;
- `dataset/` and `recordings/`: Git-ignored.
