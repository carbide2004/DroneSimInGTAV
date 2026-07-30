# Research direction

## Problem

The target event is initially invisible. A drone must infer its location from
dynamic agents whose behavior was induced by the event, such as an emergency
vehicle approaching a fire or pedestrians moving away from it. It must maintain
spatial uncertainty and choose observations that distinguish competing event
hypotheses.

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
5. Add building occlusion and response-entity visibility truth.
6. Validate that useful cues persist long enough to be observed from reachable
   viewpoints.
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
occlusion and visibility truth, followed by reachable-viewpoint solvability
before response-ecology measurement. The old static anomaly generators and
oracle collection code must not be restored.
