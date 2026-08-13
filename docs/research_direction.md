# Research direction

## Problem

The target event is initially invisible. A drone must infer its location from
dynamic agents whose behavior was induced by the event, such as an emergency
vehicle approaching a fire or pedestrians moving away from it. It must maintain
spatial uncertainty and choose observations that distinguish competing event
hypotheses.

The strict terminal task succeeds only when the agent issues STOP while the
fire-source vehicle is task-observable and its start-local 3D event estimate
is within 5 metres of the resolved GTA event position. Smoke alone is not
terminal confirmation. At the initial observation, the source vehicle must be
physically occluded and the fire/smoke envelope must not be task-observable in
either named view; partial envelope line of sight is retained as diagnostic
truth. Initial visibility, spatiotemporal cue accessibility, and goal-view
reachability are reported separately and are not themselves claims that the
task is solvable.

This is not ordinary object navigation. The desired inference is:

```text
hidden event
  -> dynamic social and traffic responses
  -> partial drone observations
  -> event-location belief
  -> informative next observation
```

## Evidence requirements

Model navigation success cannot establish that GTA contains a useful response
ecology. Response reliability must first be measured independently of any
learned policy:

- entity identity, type, position, velocity, heading, and trajectory
- event affiliation and intended response role
- response latency, duration, failure rate, and visibility
- direction agreement relative to the hidden event
- stability across locations, seeds, weather, and interventions

Oracle trajectories that read the event coordinate are not evidence that a
model uses response cues. Free-text explanations and representation alignment
are also insufficient evidence of causal use.

## Implementation sequence

1. Maintain the minimal camera, control, and synchronized RGB-D runtime.
2. Add a fire-only scenario lifecycle and a structured event/entity registry.
3. Freeze inference time with a fixed-step lockstep simulation clock.
4. Acquire named oblique and nadir RGB-D views from one simulation instant.
5. Add building occlusion, response-entity visibility truth, and deterministic
   start-local task blueprints.
6. Measure spatiotemporal cue accessibility and goal-view reachability without
   calling either one full task solvability.
7. Collect state-only telemetry and statistically validate native and
   explicitly controlled response modes.
8. Add paired semantic interventions: cue removal, distractors, conflicting
   cues, and changed event affiliation.
9. Establish oracle-entity and explicit Bayesian belief baselines.
10. Add visual perception and tracking noise.
11. Compare learned temporal models only after the task and signals are shown to
   be valid.
12. Add structured Awareness as a decision bottleneck and test it through direct
   intervention.

Stages 1, 2A, and 2B implement the controlled response kernel, lockstep clock,
and dual-view observation foundation. They deliberately do not claim that
GTA's native response ecology is reliable. The next milestone is explicit
occlusion and visibility truth, followed by spatiotemporal cue accessibility
and goal-view reachability before response-ecology measurement. Neither
diagnostic is called full task solvability. The old static anomaly generators
and oracle collection code must not be restored.

The research action budget uses seven mutually exclusive actions: fixed
2-metre FORWARD, fixed 2-metre ASCEND and DESCEND, fixed 15-degree TURN_LEFT
and TURN_RIGHT, HOLD, and STOP. A movement or turn advances one 250 ms
simulation step, HOLD advances time without changing pose, and STOP terminates
from the current observation. There is no backward, lateral, diagonal, or
combined movement and rotation. The canonical horizon is 65 actions including
STOP.
A valid consecutive response cue requires the same ACTIVE actor to remain
task-observable across two adjacent observations, move at least 0.4 metres
horizontally, and agree with its expected event-relative direction by a
horizontal cosine of at least 0.5.
Stage 2D establishes that cue inside the search rollout. Independent replay
keeps camera, action, clock, direct-goal, STOP, and localization checks strict,
while reporting same-step cue reproduction as a diagnostic because GTA AI
trajectories are not specified to be frame-identical across reconstructed
rollouts. Cross-rollout cue reliability belongs to response-ecology
measurement.

## Current Stage 2D TODO

The immediate engineering target is a stable strict witness for
`CUE_VISIBLE`. Because that stratum guarantees at least one response actor is
observable at the initial frozen observation, the audit must first support
repeated HOLD observations at the start pose before invoking spatial search.

`CUE_HIDDEN` joint-witness planning is deferred. In the current fixed-action
search, ten exploratory episodes with an 80-action horizon produced no joint
witness and almost never found even a first task-observable response view.
This is recorded as a planner/search-coverage TODO, not evidence that the task
or stratum is mathematically impossible. Its visibility thresholds and cue
definition must not be silently weakened to manufacture a pass.

## Stage 2E expert-data milestone

Stage 2D is a finite feasibility audit, not the expert used to create the
learning dataset. Stage 2E replaces joint oracle witness search with an online
cue-grounded teacher:

```text
visible RGB-D-grounded response tracks
  -> broad event-location belief
  -> temporary subgoal
  -> collision-only strict local planning
  -> direct source verification and STOP
```

Task starts are 40--60 metres from the event. Dataset generation verifies at
least one clear source-observable goal view and applies an optimistic strict
action lower bound that must retain 15 actions for evidence collection and
obstacle detours. This budget audit is not a reachability proof; the actual
cue-grounded rollout decides whether the episode is retained.
`--start-audit-timeout` bounds only this lightweight check. A Stage 2E start is called
`POTENTIAL_CUE_VISIBLE`: at least one responder is initially task-observable
and is active or scheduled to activate within two seconds. It is not a valid
dynamic cue until two adjacent RGB-D observations recover at least 0.4 metres
of motion.

The controlled pedestrian response field contains eight actors in each
event-distance band 8--20, 20--35, 35--50, and 50--65 metres. Actors exist
from Prepare but activate according to a deterministic distance-dependent
wave stored in the immutable blueprint. This is controlled experimental
structure, not evidence about GTA's native response ecology.

The teacher receives truth-assisted anonymous association and visible sample
pixels, but entity positions are recovered from the corresponding metric
Depth. It cannot read the event coordinate, event affiliation, GTA velocity,
task state, world view matrices, or the static goal budget audit. Every action
produces a new observation and Awareness update. A collision-only local A*
plan may be reused while intent, belief mode, subgoal, cue availability, and
collision validity remain unchanged. Its 20-metre subgoal is retained while
the strict search budget is 32 actions, bounded by 12,000 expanded states and
15 wall-clock seconds. `SEARCH_CUE` and `REACQUIRE_CUE` are distinct: both use
a finite six-turn scan, and repeated evidence from the same track cannot reset
the scan into indefinite rotation.

Start filtering applies that same RGB-D grounder to the initial
observation and rejects raycast-visible candidates that cannot yield a real
response track scheduled to activate within two seconds. The camera faces the
selected responder. A SEARCH_CUE or REACQUIRE_CUE context permits at most two
finite scans with one intervening altitude change or HOLD; exhausting them
produces an explicit failure rather than resetting the completion state and
rotating forever.
Fire-source confirmation is likewise bounded: two failed consecutive-view
confirmation attempts produce `SOURCE_CONFIRMATION_UNSTABLE`.

For interactive Stage 2E validation, the requested anchor may be the current
scripted-camera position. Event placement still resolves to a vehicle node no
more than 30 horizontal metres away; camera altitude is deliberately excluded
from this road-proximity check. Explicit world anchors remain the reproducible
dataset-generation interface.

The 65-action horizon is the canonical benchmark default rather than a hidden
runtime constant. Stage 2E validation and bounded dataset generation expose
`--max-steps`; the value is carried in the task blueprint so start
budget audit, teacher rollout, and action execution use one budget. An
episode that reaches its STOP-reserved final action without direct source
confirmation is recorded as a horizon-exhaustion failure.

The Stage 2E planner uses GTA's positive-yaw-is-left convention. In the
start-local `(forward, right)` plane its heading is
`(cos(yaw_delta), -sin(yaw_delta))`. Online FORWARD execution checks the
observed local displacement against this relation, so planner/runtime sign
disagreement fails immediately instead of producing a mirrored expert path.

`validate_stage2e_expert.py --record-dir PATH` is an explicit visual-audit
mode. It records compressed oblique/nadir RGB, actions, odometry, grounded
boxes, structured Awareness, belief maps, and compact evaluation truth, but no
Depth. Partial failed rollouts are retained for diagnosis. The default remains
strictly in-memory so repeated validation cannot silently fill the disk.
