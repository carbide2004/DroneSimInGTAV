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
