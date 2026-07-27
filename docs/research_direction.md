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
3. Collect state-only telemetry and statistically validate native and
   explicitly controlled response modes.
4. Add paired semantic interventions: cue removal, distractors, conflicting
   cues, and changed event affiliation.
5. Establish oracle-entity and explicit Bayesian belief baselines.
6. Add visual perception and tracking noise.
7. Compare learned temporal models only after the task and signals are shown to
   be valid.
8. Add structured Awareness as a decision bottleneck and test it through direct
   intervention.

The next milestone after Stage 0 is therefore a new `ScenarioManager` and
event-response ground-truth interface, beginning with one fire event. The old
static anomaly generators and oracle collection code must not be restored.
