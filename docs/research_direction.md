# Research direction

## Central question

Can an aerial agent localize an initially hidden urban event by observing how
other agents respond to it?

The causal direction in the simulator is:

```text
hidden event
  -> emergency, pedestrian, and traffic responses
  -> partial RGB-D observations
```

The research problem is the inverse:

```text
partial dynamic responses
  -> spatial event belief
  -> informative navigation
  -> direct event observation and localization
```

This differs from ordinary ObjectNav. The event is initially occluded, its
coordinate is never an agent input, and successful termination requires direct
observation plus a precise local-coordinate estimate.

## Research hypotheses

1. Event-induced dynamic responses contain statistically useful information
   about a hidden source.
2. Combining multiple response types improves spatial belief over using one
   cue family.
3. Active viewpoint selection resolves hypotheses more efficiently than
   greedily following the current belief mode.
4. Paired interventions can distinguish genuine cue use from map, texture,
   spawn, road-layout, and static-scene shortcuts.
5. A structured decision bottleneck can provide faithful Awareness only if it
   lies on the action path and responds correctly to direct intervention.

## Required evidence

Navigation success by itself is insufficient. A credible benchmark must
separately establish:

- response direction, latency, duration, failure, and visibility statistics;
- stability across event locations, seeds, weather, and scene geometry;
- map-only, static-image, and no-dynamic-cue shortcut baselines;
- oracle-entity and explicit spatial-belief upper bounds;
- cue-removal, distractor, conflict, and affiliation-swap interventions;
- belief calibration and information-gain behavior;
- generalization to unseen locations and cue combinations.

Oracle goal coordinates may be used by evaluation and feasibility audits but
not by the dataset expert or learned policy. Free-text explanations and
representation alignment are not evidence of causal cue use.

## Current foundation

The repository currently provides:

- stable synchronized metric RGB-D capture;
- fixed-step lockstep interaction independent of inference wall time;
- simultaneous-time oblique and nadir observations;
- a controlled fire-response lifecycle and structured entity truth;
- occlusion-aware starts and evaluation-only visibility geometry;
- a strict seven-action task environment;
- a cue-grounded belief expert and replayable successful trajectories;
- validation for capture, geometry, timing, scenarios, starts, and expert data.

The controlled pedestrian wave and fire-truck task are experimental structure,
not claims about GTA's native response ecology. The current expert is a data
generator and baseline, not the final research method.

### Known 2-D / vertical-action bias

The implemented simulator exposes `ASCEND` and `DESCEND`, but the current
Stage 2 teacher, belief grid, and collected trajectories are effectively
2-D/2.5-D. In the inspected schema-4 collection (`68` episodes, `2497`
actions), only one action was `ASCEND`, none was `DESCEND`, mean within-episode
altitude range was `0.01 m`, and the maximum was `1 m`. The existing data
therefore cannot support a claim that a learned policy has acquired meaningful
vertical exploration.

This is recorded rather than hidden by a model change. The first learned
belief baseline intentionally matches the current 2-D teacher distribution.
A later benchmark audit must add altitude-stratified starts, vertically
occluded goals/cues, 3-D belief and viewpoint metrics, and enough successful
vertical actions before ascent/descent performance is trained or reported.
Until then, results apply only to the current Stage 2 planar navigation bias.

## Next research milestones

### 1. Learned explicit belief baseline under the current Stage 2 bias

Replace the hand-written cue-to-belief update before replacing the action
planner. The initial additive baseline consumes anonymous structured tracks
recovered from RGB-D and learns cue direction, angular uncertainty, and
reliability. Stage 3A adds a source-blind Spatial ConvGRU whose only recurrent
state is the explicit log-belief map. Both are evaluated over the same interval
from the first valid dynamic cue to immediately before direct source grounding.

These remain entity-token, planar baselines. They are not final perception
models and categorical embeddings are not evidence of open-vocabulary
generalization. Online planner integration follows only after offline
calibration and held-out-anchor behavior are understood.

### 2. Deferred benchmark and vertical audit

Collect state-only telemetry before training a learned policy. Measure response
latency, direction agreement, trajectory diversity, task failure, cue
visibility, redundancy among pedestrians, and road/sidewalk biases across many
locations and blueprints. Include the vertical-coverage requirements described
above before calling the benchmark 3-D.

### 3. Paired semantic interventions

Reuse immutable blueprints to construct matched conditions:

- remove a response type;
- add an unrelated emergency vehicle;
- introduce conflicting response directions;
- change event affiliation while preserving plausible motion;
- vary which responders are visible from the same start.

Interventions should preserve scene naturalness and background geometry rather
than reverse trajectories or produce obviously invalid motion.

### 4. Additional explicit belief baselines

Start with privileged entity tracks and a transparent spatial filter. Compare
map-only, static RGB, oracle cue, Bayesian belief, and entity-token recurrent
baselines before increasing model complexity.

### 5. Learned perception and temporal models

Replace privileged association with detection and tracking noise, then compare
simple recurrent, attention-based, and state-space temporal models. Architecture
choice follows evidence about sequence length and failure modes; it is not the
starting contribution.

### 6. Structured Awareness

Represent observed cues, hypotheses, support, contradiction, uncertainty, and
information need as machine-testable fields. The policy must consume this
state without a visual bypass. Intervene on the structure while holding visual
input fixed and require the action to change in the predicted direction.

## Intended benchmark scope

Fire is the first controlled event, not a sufficient final benchmark. Future
event types should provide distinct response mechanisms and controllable
counterfactuals without relying on unique static appearance. Candidate types
must first be audited for stable GTA behavior, visibility duration, and safe
reset semantics before entering the benchmark.

The intended progression is:

```text
validated simulation substrate
  -> response ecology
  -> counterfactual benchmark
  -> explicit belief baselines
  -> learned perception and policies
  -> intervention-tested Awareness
```
