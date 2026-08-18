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

Grouped collection mode stores a `dataset_manifest.json` and one directory per
fixed scenario blueprint. Each scene has an independent success quota and
attempt budget. A later `--resume` validates the collection configuration,
every completed episode, absence of partial payloads, and the blueprint
signature rebuilt from the same anchor and seed before appending data. Runtime
blueprint IDs are never trusted across GTA/plugin restarts.

## Batch layout

```text
stage2e_fire/
  dataset_manifest.json
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
  scene_001_seed_2/
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
python validation\visualize_stage2e_dataset.py dataset\stage2e_fire
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

Generation reports scenario preparation, lockstep setup, start generation,
visibility, grounding, teacher planning, recording, pose control, Advance,
Capture, finalization, and cleanup. Timing is observational and does not alter
candidate ordering or task semantics.

Summarize a completed batch with:

```powershell
python validation\summarize_stage2e_timings.py dataset\stage2e_fire
```
