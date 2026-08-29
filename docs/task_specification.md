# Task specification

## Research problem

The event source is initially hidden. The agent must use event-induced dynamic
responses, such as a fire truck approaching a fire or pedestrians fleeing from
it, to maintain a spatial hypothesis and choose actions that eventually expose
the source.

This is not oracle ObjectNav. The agent never receives the resolved event
coordinate, a goal bearing, GTA entity handles, scenario truth, visibility
truth, GTA velocities, task states, or a world-coordinate camera matrix.

## Coordinate frame

Every episode defines a start-local frame:

- origin: initial camera center;
- positive X: initial body-forward;
- positive Y: initial body-right;
- positive Z: GTA world up.

GTA positive yaw turns left. A relative yaw `theta` therefore has start-local
horizontal heading `(cos(theta), -sin(theta))`. At `+90°`, `FORWARD` moves
toward negative start-local right.

The event estimate supplied to `STOP` is a finite 3D coordinate in this local
frame. Evaluation transforms it to GTA world coordinates only after the agent
terminates.

## Initial condition

The first observation is acquired at `t = 250 ms` after scenario Start and
does not consume an agent action.

At that observation:

- every geometry sample on the fire-source vehicle is occluded from the camera
  center;
- fire/smoke envelope visibility is retained as diagnostic truth only and does
  not accept or reject a start, because GTA particle rendering depends on LOD,
  distance, and view state.

Current expert-data generation uses `POTENTIAL_CUE_VISIBLE` starts. The camera
is 40--60 metres from the event, and at least one RGB-D-grounded responder is
active or scheduled to activate within two seconds. This condition is not
itself valid dynamic evidence.

The visibility audit also defines:

- `CUE_VISIBLE`: at least one affiliated responder is initially
  task-observable;
- `CUE_HIDDEN`: all affiliated responders are initially hidden.

These are initial visibility strata, not statements that a complete episode
is solvable.

## Observation

An observation contains:

- `oblique` RGB and metric depth at pitch `-45°`;
- `nadir` RGB and metric depth at pitch `-90°`;
- start-local camera position and relative yaw;
- the current lockstep index and action budget state.

The two views share camera center, yaw, roll, FOV, clip planes, and GTA
simulation time. They use different render frame IDs. Capture restores the
camera to the canonical `-45°` pitch afterward.

Raw world-to-view matrices are retained by evaluation code for geometry
checks but removed from the agent-facing observation.

## Action space

The research action space is strictly discrete and mutually exclusive:

| Action | Effect |
|---|---|
| `FORWARD` | Move exactly 1 m along current body-forward |
| `ASCEND` | Move exactly 1 m along GTA world up |
| `DESCEND` | Move exactly 1 m along GTA world down |
| `TURN_LEFT` | Increase GTA yaw by exactly 15° |
| `TURN_RIGHT` | Decrease GTA yaw by exactly 15° |
| `HOLD` | Keep pose fixed and acquire a later observation |
| `STOP(event_estimate_local)` | End on the current frozen observation |

There is no backward or lateral movement, diagonal translation, variable
magnitude, or combined translation and yaw. The lower-level absolute pose API
is not the agent action space.

Every non-terminal action advances exactly `250 ms` of GTA simulation time and
then returns a frozen dual-view observation. `STOP` consumes one action and
does not advance time. The canonical horizon is 80 actions including the
reserved terminal action.

## Task observability

A geometry target is `task-observable` in a named view when:

- at least four samples are unoccluded and in the camera frustum;
- the projected clear-sample span is at least 24 pixels;
- the clear-sample box remains at least 12 pixels inside every image border.

A target is task-observable for the dual-view observation if either named view
satisfies the definition. Fire particles and smoke are diagnostic appearance;
terminal source observability is currently defined on the source vehicle.

## Dynamic cue

A valid consecutive response cue requires:

- the same associated responder in two adjacent frozen observations;
- task-observability in both observations;
- ACTIVE response state in evaluation truth;
- at least `0.4 m` of horizontal displacement over the interval;
- event-relative horizontal direction cosine of at least `0.5`.

The fire-truck direction points toward the event. A fleeing pedestrian's
direction points away from it. Association and evaluation status are
privileged; image location and 3D motion used by the expert are recovered from
RGB-D.

## Success

An episode succeeds only if all conditions hold at one terminal observation:

```text
STOP is issued
AND the controlled fire remains active
AND the fire-source vehicle is task-observable
AND the start-local 3D estimate is finite
AND its world-space Euclidean error is at most 5 m
```

Seeing only a smoke column does not satisfy terminal source confirmation.
Initial visibility, cue accessibility, and a reachable source viewpoint are
reported separately; none alone proves full task solvability.

## Expert boundary

The current teacher receives truth-assisted anonymous association and visible
sample pixels. It recovers actor positions and motion from metric depth and
updates a broad spatial belief. It cannot read event coordinates, event
affiliation, GTA velocities, world matrices, or the start-budget audit.

This teacher is intended to construct training trajectories and structured
decision records. It is not evidence that a learned policy uses causal
response cues; that claim requires counterfactual intervention experiments.

The first learned baseline deliberately replaces only the teacher's
cue-to-belief update. It consumes structured tracks recovered from the episode
RGB-D stream and retains the analytical strict-action planner. This is an
oracle-association/entity-token baseline: stable track association is provided
by the dataset, while event coordinates, event affiliation, GTA velocity,
task state, and teacher-inferred event direction remain excluded.

Stage 3A trains and evaluates this learned posterior offline. Stage 3B runs the
same posterior online behind the unchanged analytical planner as a controlled
baseline. Stage 3C adds a separate planner-free policy whose only route from
response tracks to action is the explicit predicted belief. Its other inputs
are observation-derived dual-view Depth geometry, odometry/action history, and
a separately grounded fire-source token used for terminal localization. The
Stage 3C control path does not call the analytical planner, confidence state
machine, GTA geometry action mask, teacher belief, or event truth. The current
visibility-assisted RGB-D grounder remains a structured perception upper bound.

Its posterior is 2-D. Although the environment action contract remains 3-D,
the present Stage 2 trajectories contain essentially no vertical exploration;
the learned baseline must not be presented as a learned ascent/descent policy.

The current teacher emits `STOP` only after the same RGB-D-grounded source has
been observed in two adjacent observations, its horizontal range from the camera
is at most `30 m`, and its clear projected box spans at least `64 px`. A farther
grounded source is passed to the existing short-range planner; once horizontally
close, the teacher descends if needed to enlarge the source observation.
Successful dataset retention also requires a valid dynamic cue and a
cue-sensitivity audit. These are stricter expert-quality filters, not additional
environment `TaskSuccess` conditions.
