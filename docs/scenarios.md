# Controlled scenarios

## Lifecycle

The runtime supports one active scenario:

```text
EMPTY -> PREPARING -> READY -> RUNNING -> RESET -> EMPTY
                  \-> FAILED -----------/
```

Prepare is a non-blocking per-frame process. It validates the loaded area,
resolves an immutable blueprint, loads models, clears safe ambient entities,
and places scenario entities. Start records one GTA timer/frame origin,
activates the event, and issues each response task once. Reset owns all cleanup
of fire handles, vehicles, pedestrians, tasks, and model references.

Errors remain explicit until Reset. Scenario logic does not silently replace
entities, resend failed tasks, teleport responders back to route, or create a
new scenario over an active one.

## Fire experiment

The current event contains:

- one damaged frozen `blista` as the event source;
- zero to four `firetruk` vehicles with firefighter drivers;
- zero to 32 civilian pedestrians;
- one verified looped fire effect and script-managed fire state.

Fire trucks spawn on distinct road nodes and receive one task toward the
event. Pedestrians occupy four event-centric distance bands: `8--20`,
`20--35`, `35--50`, and `50--65 m`. They exist in READY but remain frozen while
pending. Deterministic activation offsets create an outward response wave over
approximately six seconds instead of releasing the entire crowd at once.

Task failures, actors disappearing, and unusual GTA navigation are recorded as
outcomes. They are not hidden by repeated commands.

## Area loading and population control

The caller teleports the player into the requested area before Prepare so GTA
loads collision and navigation data. The requested anchor must resolve to a
vehicle node within 30 horizontal metres.

Ordinary ambient vehicles and pedestrians in the controlled area are removed
when safe. Player and mission entities are never deleted; they are retained as
protected entities in the scenario snapshot. During READY and RUNNING,
population density is suppressed and periodic maintenance removes ordinary
entities entering the area. Use a formal clean-area validation when collecting
benchmark data.

## Seeds and immutable blueprints

The master seed expands into independent streams for fire-truck placement,
pedestrian placement, pedestrian models, and activation timing. It does not
change GTA's global random seed.

Prepare with `blueprint_id=0` resolves a new blueprint. Its returned ID can be
reused after Reset to instantiate the same event and any prefix of its actor
capacity. A reuse request must match the original anchor and seed and cannot
request more entities than the blueprint contains.

Blueprint reuse preserves resolved event position, actor spawn positions,
headings, models, and activation offsets. It does not promise frame-identical
GTA AI paths. The benchmark unit is therefore a controlled blueprint plus
recorded rollout, not a claim of deterministic engine navigation.

## Truth schema

`ScenarioSnapshot` contains:

- scenario, blueprint, seed, lifecycle, game timer, and frame count;
- requested anchor and resolved event position;
- common Start timer/frame and fire-active state;
- ambient/protected entity counts;
- all registered scenario entities.

Each entity records stable ID, GTA handle, model, kind, role, event
affiliation, existence, task state, position, velocity, speed, heading, spawn
and activation timing, and task target. Stable IDs are scenario-local
evaluation identifiers and are never agent observations.

## Visual effects and visibility

Fire and smoke particles communicate the event visually but do not provide
geometric truth. Particle-handle existence does not imply that visible pixels
occur from every viewpoint. `validate_fire_visual_coverage.py` therefore
audits controlled distances, heights, and azimuths separately from task
success.

Geometric target visibility uses sampled source, responder, and fire-envelope
points plus GTA shape tests. It is intended for start construction and
evaluation, not as a substitute for learned perception.
