# Dataset format

## Generation policy

`validation/generate_stage2e_experts.py` performs bounded scenario attempts and
retains payloads only for episodes that satisfy the strict terminal task.
Failed attempts append compact diagnostics to `failures.jsonl` and remove
their partial payload directory. Every attempt appends timing information to
`timings.jsonl`.

The batch output root may already exist, but every generated episode name and
its `.partial` directory must be new. Using a fresh root per collection run is
recommended. Dataset and recording roots are ignored by Git because RGB-D
collections can grow quickly.

Grouped collection mode uses manifest schema 4 and stores a
`dataset_manifest.json`, one reusable `start_pool.json` per accepted anchor,
and directories organized by anchor and fixed scenario blueprint. The ASI
manual camera saves
anchor rows to `<GTA V>\data\DroneSim_anchors.jsonl` when `F8` is pressed; the
generator reads that file with `--anchor-file`. `--scenes-per-anchor` controls
the number of seed-distinct blueprints at each coordinate and
`--starts-per-scene` controls the successful episode quota for each blueprint.
The older names `--scenario-count` and `--episodes-per-scenario` are aliases.
Each scene has an independent attempt budget.

The anchor file is UTF-8 JSONL with one finite numeric XYZ record per line:

```json
{"x":234.123456,"y":324.123456,"z":100.123456}
```

Blank lines are ignored. Malformed records and duplicate coordinates are
rejected before the generator connects to GTA.

For a fresh grouped collection, all anchors are classified before the manifest
is created. This phase prepares a source-only static scene with zero responders,
so pedestrian/vehicle spawn probability cannot invalidate anchor geometry.
Static rays identify building-shadow sectors; candidate points pass ground,
2 m camera clearance, source-vehicle occlusion, and task-observable goal-view
budget checks. The current deterministic grid covers 42--60 m radius in 2 m
increments, 25--60 m AGL, and dense azimuths within supported shadow sectors;
farthest-point sampling retains at most 160 entries. Fire-envelope masks are
stored as diagnostics but do not reject
a start. `UNSUITABLE` rows are atomically removed from an anchor JSONL while
retained text and order are preserved. Any `UNKNOWN` failure leaves the file
untouched.

For each scene slot, projected response visibility first creates a deterministic
truth-level shortlist at `t=250 ms`. Every shortlisted entry is then checked
once with the real dual-view camera and the episode RGB-D grounder.
`starts_per_scene` is the hard acceptance quota; the generator attempts to
certify `scene_catalog_reserve` additional starts, but a reserve shortfall only
produces a diagnostic warning. The certified catalog, its digest, attempted IDs, and successful IDs are
stored in the manifest. An insufficient empty scene slot advances to the next
deterministic scenario seed, bounded by `max_scene_seed_candidates`; protocol,
capture, and implementation errors stop collection immediately.

A later `--resume` validates the collection configuration, every completed
episode, absence of partial payloads, and the immutable blueprint signature.
The plugin's runtime blueprint ID is persisted to accelerate a Python-only
resume, but a GTA/plugin restart triggers a same-anchor/same-seed rebuild and
signature check before any payload is appended. It also loads each pool,
verifies its digest, and rechecks event position and static occlusion masks on
first use. Schema 1/2/3 batches remain supported by offline visualization but
cannot be resumed for generation.

## Batch layout

```text
stage2e_fire/
  dataset_manifest.json
  anchor_000/
    start_pool.json
    scene_000_seed_1/
      episode_0000_attempt_0000_start_<id>/
        agent/
          episode.json
          steps.jsonl
          rgb/
          depth/
        teacher/
          episode.json
          awareness.jsonl
          beliefs.npz
        evaluation_truth/
          episode.json
          steps.jsonl
    scene_001_candidate_001_seed_17/
  anchor_001/
  ...
  failures.jsonl
  timings.jsonl
```

RGB is stored as compressed JPEG. Metric depth is stored per view as compressed
NumPy `float16` arrays in metres. Capture, grounding, planning, and online
validation continue to use `float32`; the lossy conversion happens only at the
dataset writer. It is not clipped, normalized, or downsampled. At the current
depth range, binary16 resolution is approximately 3 cm near 50--60 m, 6 cm at
120 m, and 0.5 m at 800 m. The dtype and units are declared in
`agent/episode.json`. Episode and step metadata use JSON/JSONL; belief grids
use NPZ.

`start_pool.json` contains only evaluation/generation state: requested anchor,
resolved event position, algorithm version, accepted camera positions, AGL,
bearing, static source visibility plus diagnostic envelope samples,
task-observable goal views, lower-bound budget, timing, and a content digest.
It is never exposed in the Agent stream. The schema-4 manifest additionally
stores each scene's real-RGB-D-certified catalog and rejected seed history.
Within one scene every catalog start ID is attempted at most once and every
successful ID is unique; different seed-distinct scenes may reuse the same
static position and select a different responder-facing yaw. The manifest and
teacher metadata record the close-source STOP policy (`30 m`, `64 px`, two
consecutive grounded observations); resume rejects collections created under a
different policy instead of mixing supervision semantics.

## Agent stream

The agent stream contains the information available along the expert
trajectory:

- synchronized oblique and nadir RGB-D references;
- start-local odometry and relative yaw;
- lockstep step and simulation time;
- selected strict research action and action budget state.

It excludes absolute world coordinates, raw view matrices, event truth,
entity handles, visibility truth, and GTA task state.

## Teacher stream

The teacher stream contains structured decision state aligned with the agent
step:

- grounded visible tracks and recovered motion evidence;
- spatial belief and connected belief mode;
- intent, temporary subgoal, support, contradiction, and uncertainty;
- local-planner status and selected action.

This is generated supervision. Future Awareness experiments must place any
claimed reasoning state on the actual decision path and test it through direct
intervention; free-text plausibility alone is insufficient.

## Evaluation truth

Privileged truth records the immutable task blueprint, event location,
registered entity states, task status, cue validity, source observability,
terminal estimate, and success/error metrics. It exists for dataset auditing
and scoring and must remain separated from the agent stream.

## Compact validation recording

`validate_stage2e_expert.py --record-dir PATH` writes a smaller visual-debug
format. It stores both RGB streams as JPEG, actions, odometry, belief,
Awareness, projected boxes, and compact truth. It never stores depth. Unlike
dataset generation, a failed or exceptional validation rollout is retained so
the failure can be inspected.

Use:

```powershell
python validation\visualize_stage2e_trajectory.py PATH
```

## Offline dataset playback

The dataset visualizer accepts either one successful episode directory or the
batch root:

```powershell
python validation\visualize_stage2e_dataset.py dataset\stage2e_multi_anchor
```

It joins the agent, teacher, truth, and belief streams, but deliberately does
not load metric depth. It writes no converted output.

Controls:

- `Space`: pause/play;
- `Left/Right`: previous/next frame;
- `Home/End`: first/last frame;
- `Up/Down`: previous/next episode;
- `Q`: close.

## Timing records

Generation separately reports `anchor_prepare`, `shadow_rays`,
`ground_clearance`, `fire_occlusion`, `goal_audit`,
`dynamic_response_query`, `yaw_selection`, `real_camera_verify`, and
`rgbd_grounding`, followed by teacher planning, recording, pose control, Advance,
Capture, finalization, and cleanup. Timing is observational and does not alter
candidate ordering or task semantics.

Summarize a completed batch with:

```powershell
python validation\summarize_stage2e_timings.py dataset\stage2e_multi_anchor
```
