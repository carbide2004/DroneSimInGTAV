# DroneSimInGTAV

DroneSimInGTAV is a small GTA V research runtime for controlling a scripted
camera, acquiring synchronized RGB-D observations, and running a minimal
controlled fire-response experiment. Stage 2A adds an explicit lockstep
simulation clock so model inference time does not consume GTA simulation time.
Stage 2B adds named oblique and nadir RGB-D captures from one frozen
simulation instant.

The research direction is hidden-event localization from event-induced dynamic
responses. See [docs/research_direction.md](docs/research_direction.md).

## Strict research task definition

This project studies hidden-event localization from event-induced dynamic
responses. It is not oracle ObjectNav: the agent is never given the event
coordinate, a goal bearing, GTA entity handles, scenario truth, visibility
truth, or a world-coordinate camera matrix.

An episode starts from a scripted-camera pose whose local coordinate frame is:

- origin at the initial camera center;
- positive X along the initial body-forward direction;
- positive Y along the initial body-right direction;
- positive Z along GTA world up.

GTA positive yaw turns left. Consequently, a yaw delta `theta` has the
start-local horizontal heading `(cos(theta), -sin(theta))`; at `+90 degrees`,
`FORWARD` moves along negative local-right. The Stage 2E local planner,
executor invariant checks, and trajectory player all use this convention.

The first observation is acquired at `t = 250 ms` after the controlled
responses start. At the start, every geometry sample on the fire-source
vehicle must be physically occluded from the camera center. The fixed
`8 m` radius, `25 m` high fire/smoke envelope may have partial line of sight,
but it must not be task-observable in either named view. Its clear-sample
fraction is retained as diagnostic truth. Initial conditions are reported
separately:

- `CUE_VISIBLE`: the source vehicle is physically occluded, the fire/smoke
  envelope is not task-observable, and at least one affiliated response actor
  is task-observable;
- `CUE_HIDDEN`: the source vehicle is physically occluded, the fire/smoke
  envelope is not task-observable, and all affiliated response actors are
  initially not task-observable.

Stage 2E uses the more precise name `POTENTIAL_CUE_VISIBLE`. The camera starts
40--60 metres from the event, and at least one task-observable responder is
active or scheduled to activate within two seconds. This condition does not
itself count as dynamic evidence: the same RGB-D-grounded actor must still be
observed moving across two adjacent frozen observations.

Each observation contains synchronized metric RGB-D from one frozen simulation
instant:

- `oblique`: pitch `-45 degrees`;
- `nadir`: pitch `-90 degrees`.

The canonical research action space is strictly discrete:

- `FORWARD` moves exactly `2 m` along the current body-forward direction;
- `ASCEND` moves exactly `2 m` along GTA world up;
- `DESCEND` moves exactly `2 m` along GTA world down;
- `TURN_LEFT` increases GTA yaw by exactly `15 degrees`;
- `TURN_RIGHT` decreases GTA yaw by exactly `15 degrees`;
- `HOLD` keeps the pose fixed while acquiring the next observation;
- `STOP(event_estimate_local)` terminates on the current frozen observation.

`FORWARD`, `ASCEND`, `DESCEND`, both turn actions, and `HOLD` each advance
exactly `250 ms` of GTA simulation time and produce a new dual-view
observation. `STOP` consumes one action but does not advance simulation time.
There is no backward motion, lateral translation, diagonal translation,
variable action magnitude, or combined translation and rotation. The
canonical horizon is 65 actions, including `STOP`. These limits are separate
from the low-level absolute-pose API and the `1 m / 15 degree` manual keyboard
controls.

A valid consecutive dynamic cue uses the same ACTIVE response actor in two
adjacent frozen observations. The actor must be task-observable in both,
move at least `0.4 m` horizontally over the `250 ms` interval, and have a
horizontal response-direction cosine of at least `0.5` relative to the event.
Stage 2D requires this cue in the search rollout that establishes the witness.
Independent replay strictly verifies action decomposition, camera poses,
lockstep advancement, direct source observability, STOP, and localization.
Whether an independently reconstructed GTA AI rollout presents the same cue
at the same steps is reported as `cue_reproduced`; it is not a Stage 2D hard
condition because blueprints do not promise frame-identical AI trajectories.
Cross-rollout cue reliability is measured later as response-ecology evidence.

The agent must express its event estimate in the start-local coordinate frame.
An episode succeeds only when all of the following hold at the same terminal
step:

```text
the agent issues STOP
AND the fire-source vehicle is task-observable in the final RGB-D observation
AND the agent provides a finite start-local 3D event coordinate
AND its world-space 3D Euclidean error is at most 5 m
```

## Stage 2E cue-grounded expert

The controlled response field places 32 pedestrians in four event-centric
distance bands: 8--20, 20--35, 35--50, and 50--65 metres. All actors exist in
READY, while immutable blueprint activation offsets create an outward
response wave. Pending actors remain frozen even during a lockstep Advance.

The expert never reads the event location. Visibility truth supplies only
anonymous track association and visible projected samples; positions and
motion are recovered from synchronized metric Depth. Broad firetruck and
pedestrian direction likelihoods update a 4-metre belief grid. The highest
mass connected mode selects a temporary subgoal, and a collision-only strict
local A* supplies at most 32 actions. The bounded search also enforces the
episode activity volume, a 12,000-state expansion limit, and a 15-second wall
timeout.

The teacher observes after every action. The remaining local action queue is
reused until the intent, primary belief mode, subgoal, cue availability,
terminal visibility, or collision state changes.

`SEARCH_CUE` denotes the period before any valid dynamic evidence has been
obtained; `REACQUIRE_CUE` is reserved for later cue loss or belief ambiguity.
Both use bounded scans. A scan context may perform two finite six-turn scans,
with at most one altitude change or HOLD between them. If neither scan yields
an RGB-D-grounded response track, the rollout fails explicitly with
`CUE_SEARCH_EXHAUSTED`; it cannot restart the same scan indefinitely.

`POTENTIAL_CUE_VISIBLE` certification uses the exact episode
`VisibleTrackGrounder`, not only raycast/projected-box truth. The initial pair
must contain at least one responder scheduled to activate within two seconds
whose metric Depth supports the same grounded track representation consumed
by the teacher. The initial oblique camera is aimed toward the selected
responder, with no 180-degree reversal. The validator reports
`initial_grounded_responses` for this invariant.

Direct fire-source verification still requires two adjacent grounded
observations. If a candidate source disappears after two separate confirmation
HOLDs, the rollout fails as `SOURCE_CONFIRMATION_UNSTABLE` rather than
repeating a discover/lose cycle until the horizon.

Run the no-payload online validation with:

```powershell
python validation\validate_stage2e_expert.py
```

The canonical horizon remains 65 actions including `STOP`, but online
validation can explicitly audit a different budget instead of changing source
constants:

```powershell
python validation\validate_stage2e_expert.py --max-steps 100
```

The selected value is embedded in the generated start blueprint and is shared
by the static certificate, teacher episode specification, and strict action
executor. Reaching the last reserved action without `STOP` returns
`TASK_HORIZON_EXHAUSTED_WITHOUT_STOP`; it is an episode failure, not a GTA or
protocol exception. `generate_stage2e_experts.py` exposes the same option.

When `--anchor X Y Z` is omitted, Stage 2E validation uses the current
scripted-camera position and resolves the nearest road node within 30
horizontal metres. This lets an operator fly over a desired street before
starting validation without first looking up GTA world coordinates. The
explicit option remains available for repeatable automated runs.

Trajectory recording is opt-in and the destination must not already exist:

```powershell
python validation\validate_stage2e_expert.py `
  --record-dir recordings\stage2e_validation

python validation\visualize_stage2e_trajectory.py `
  recordings\stage2e_validation
```

The compact validation recording retains both RGB streams as JPEGs, each
selected action, start-local odometry, grounded cue/source boxes, structured
Awareness, the spatial belief grid, and compact evaluation truth. It never
writes Depth. A failed or exceptional rollout retains the frames recorded up
to failure and marks the trajectory `FAILED` or `ERROR`. The offline player
connects to no GTA process; Space pauses, Left/Right steps, Home/End jumps,
and Q closes it. Each frame labels an action as `PROPOSED` or `EXECUTED`, so a
horizon-rejected final proposal is not mistaken for movement that produced
the displayed observation. Without `--record-dir`, validation still writes
nothing.

Generate a bounded number of successful episodes with:

```powershell
python validation\generate_stage2e_experts.py `
  --anchor 234 324 100 `
  --max-success-episodes 10 `
  --output-dir dataset\stage2e_fire
```

Only successful episodes retain RGB-D. Failed attempts append a compact JSON
diagnostic and remove their `.partial` payload directory.

The fire must be active at the terminal step. Seeing only a distant smoke
column is not sufficient terminal confirmation. `task-observable` means that
the target has at least four unoccluded in-frustum geometry samples, a
projected bounding span of at least 24 pixels, and a projected clear-sample
box at least 12 pixels inside every image border in either named view.

The following terms are deliberately distinct:

- `InitialVisibility`: what is observable at the first frozen observation;
- `CueAccessible`: whether response evidence can be reached while it remains
  useful;
- `GoalViewReachable`: whether the action budget permits a viewpoint that
  directly observes the fire-source vehicle;
- `TaskSuccess`: the strict terminal condition above.

Initial visibility, cue accessibility, and goal-view reachability are necessary
diagnostics, not proofs that the task is solvable. Stage 2C implements
visibility truth and task-start generation only. Stage 2D evaluates
spatiotemporal cue accessibility and goal-view reachability. Full task
solvability is tested later with an observation-only belief policy and the
strict terminal condition; it is never inferred merely from two observations
of a moving actor.

The research sequence is:

```text
lockstep and synchronized RGB-D
  -> dual-view observation
  -> visibility truth and task starts
  -> spatiotemporal cue accessibility and goal-view reachability
  -> response-ecology statistics
  -> paired counterfactual interventions
  -> explicit belief baseline
  -> learned temporal models and structured Awareness
```

## Architecture

```mermaid
flowchart LR
    subgraph Python["Python control and evaluation"]
        Agent["Agent environment"]
        Actions["ResearchActionExecutor<br/>translate / rotate / hold / stop"]
        Validation["Online validation"]
        Pair["LockstepRgbdPair<br/>oblique + nadir"]
        Starts["TaskStartGenerator<br/>virtual viewpoints and local frame"]
        Relative["RelativePoseController<br/>body-frame delta to world pose"]
        Client["DroneSimClient<br/>strict DSV3 codec"]

        Agent --> Actions --> Relative --> Client
        Agent --> Pair --> Client
        Starts --> Client
        Validation --> Client
    end

    subgraph Runtime["GTA V ASI plugin"]
        Server["ProtocolServer<br/>network thread"]
        Queue["Typed command queue<br/>request ID and completion result"]
        Script["ScriptRuntime<br/>GTA script thread and per-frame tick"]
        Camera["CameraController"]
        Clock["SimulationClock<br/>frozen inference and 250ms steps"]
        Manager["ScenarioManager"]
        Fire["FireScenario"]
        Truth["EntityRegistry<br/>structured response truth"]
        Visibility["VisibilityEvaluator<br/>virtual LOS geometry"]

        Server --> Queue --> Script
        Script --> Camera
        Script --> Clock
        Script --> Manager --> Fire --> Truth
        Script --> Visibility
        Manager --> Visibility
    end

    subgraph Capture["Synchronized RGB-D pipeline"]
        Hook["D3D11 depth hook<br/>observe current-cycle DSV"]
        Present["Present callback<br/>copy same-frame RGB and depth"]
        Ring["Three-slot staging ring<br/>event queries"]
        Worker["Worker thread<br/>RGB conversion and metric depth"]

        Hook --> Present --> Ring --> Worker
    end

    World["GTA world and ScriptHookV natives"]

    Client <-->|"DSV3 TCP"| Server
    Camera --> World
    Clock --> World
    Fire --> World
    World --> Hook
    Script -->|"same-frame camera metadata"| Present
    Worker -->|"frame ID, RGB, depth, matrices"| Server
    Truth -.->|"evaluation only"| Validation
    Visibility -.->|"evaluation only"| Starts
```

The network thread never calls GTA natives directly. It validates each request,
places a typed command in the queue, and waits for the GTA script thread to
return a correlated result. Scenario preparation and maintenance advance from
the per-frame runtime tick. RGB-D capture remains a separate render-thread
pipeline, so scenario logic cannot manufacture or cache observations.

`ResearchActionExecutor` and the stripped dual-view observation form the
agent-facing boundary. `RelativePoseController`, raw Capture V3 view matrices,
scenario snapshots, and visibility truth are privileged runtime/evaluation
interfaces and must not be included in an agent observation.

## Repository layout

```text
DroneSim/                  GTA V ASI plugin
  camera.*                 camera lifecycle and absolute pose control
  command_queue.*          request-correlated GTA-thread commands
  rgbd_capture.*           synchronized Capture V3 implementation
  scenario_manager.*       single-scenario lifecycle coordinator
  simulation_clock.*       lockstep session and fixed 250 ms advances
  fire_scenario.*          asynchronous controlled fire experiment
  entity_registry.*        stable entity IDs and response truth
  visibility.*             virtual-viewpoint occlusion geometry
  script.*                 minimal GTA script runtime
  server.*                 DSV3 TCP protocol server
agent_control/
  dronesim_client.py       strict Python client and relative-pose wrapper
  task_starts.py           evaluation-only starts and local task boundary
  research_actions.py      seven fixed research actions
  feasibility.py           Stage 2D bounded joint-witness search
  expert_starts.py         Stage 2E static start certificates
  expert_teacher.py        RGB-D-grounded belief and local planner
  expert_episode.py        strict online expert rollout
  expert_recording.py      successful-episode dataset writer
  requirements.txt         online validation dependencies
validation/
  rgbd_geometry.py
  rgbd_sync_metrics.py
  validate_dual_view_rgbd.py
  validate_pose_control.py
  validate_rgbd_pointcloud.py
  validate_rgbd_stability.py
  validate_rgbd_yaw_sync.py
  validate_fire_scenario.py
  validate_visibility_starts.py
  validate_spatiotemporal_feasibility.py
  validate_stage2e_expert.py
  generate_stage2e_experts.py
  stage2e_trajectory_recording.py
  visualize_stage2e_trajectory.py
  trajectory_recording.py
  visualize_spatiotemporal_trajectory.py
```

## Build and install

Requirements:

- GTA V Legacy with ScriptHookV
- Visual Studio 2026 with Desktop development with C++
- Windows SDK and the v145 toolset
- x64 Release configuration

Open `DroneSim.sln`, select `Release | x64`, and build. Copy the resulting
`DroneSim.asi` next to `GTA5.exe`. The repository does not automatically copy
or install the ASI.

The plugin listens on `127.0.0.5:23456` by default.

- F9 shows the current GTA world position and camera rotation in the
  bottom-left notification feed.
- F10 creates the scripted camera.
- F11 stops the scripted camera and restores the player after validation teleportation.
- W/S move forward/backward in 1-metre increments.
- A/D strafe left/right in 1-metre increments.
- Q/E turn left/right in 15-degree increments.
- Z/C move up/down in 1-metre increments.

Manual translation uses the same collision check as the network pose API.
Pressing a key applies one increment immediately; holding it repeats at a
fixed 100-ms interval independent of render FPS. Multiple held keys may be
combined for manual inspection. While the scripted camera is active, GTA
gameplay controls are suppressed so these keys do not also move or operate the
player. Manual camera keys remain disabled throughout lockstep. Pause-menu
controls remain available, and normal player input resumes after F11.

The same camera operations are available through the Python client.

## Python client

Install only the validation dependencies:

```powershell
python -m pip install -r agent_control\requirements.txt
```

Basic use:

```python
from agent_control.dronesim_client import (
    DroneSimClient,
    LockstepSession,
    RelativePoseController,
)
from agent_control.research_actions import (
    AscendAction,
    DescendAction,
    ForwardAction,
    HoldAction,
    ResearchActionExecutor,
    StopAction,
    TurnLeftAction,
    TurnRightAction,
)

client = DroneSimClient()
client.create_camera()

pose = client.get_pose()
frame = client.capture()

# The plugin receives an absolute GTA world position and yaw. Pitch and roll
# remain unchanged.
actual = client.set_camera_pose(
    pose[0],
    pose[1],
    pose[2],
    pose[5] + 45.0,
    collision_check=False,
)

# Low-level wrapper: forward, right, vertical, yaw delta.
controller = RelativePoseController(client, collision_check=True)
controller.synchronize()
controller.step_relative(0.5, 0.0, 0.0, 0.0)

# A scene must be Reset while frozen before this context can exit.
with LockstepSession(client) as lockstep:
    lockstep.advance()  # exactly one 250 ms GTA simulation step
    views = lockstep.capture_rgbd_pair()
    oblique = views.oblique
    nadir = views.nadir
```

`RelativePoseController` is a low-level utility and can still set translation
and yaw together. Research agents must use `ResearchActionExecutor`, which
accepts only the seven fixed actions above, reserves one action for `STOP`,
advances lockstep exactly once for every non-terminal action, and represents a
zero-motion observation explicitly as `HoldAction`. Its agent result contains
start-local odometry and RGB-D without the absolute world view matrices; raw
pairs remain available only to evaluation code.

All non-capture commands acknowledge only after the GTA script thread has
executed them. The Python client contains no settling sleeps. A failed command
raises `DroneSimCommandError` with one of:

- `CAMERA_INACTIVE`
- `INVALID_POSE`
- `COLLISION_BLOCKED`
- `COMMAND_TIMEOUT`
- `POSE_APPLY_FAILED`
- `POSE_MISMATCH`
- `INVALID_REQUEST`
- `INTERNAL_ERROR`
- `LOCKSTEP_ALREADY_ACTIVE`
- `LOCKSTEP_NOT_ACTIVE`
- `LOCKSTEP_SESSION_MISMATCH`
- `LOCKSTEP_ADVANCE_TIMEOUT`
- `LOCKSTEP_INTERRUPTED`
- `LOCKSTEP_CLOCK_INVARIANT_FAILED`

Capture keeps the existing V3 response: request/frame IDs, RGB, metric depth,
FOV, clip planes, projection matrix, and world-to-view matrix.

`capture_rgbd_pair()` is a Python-side composition over unchanged Capture V3.
It requires the lockstep session to be frozen, captures named `oblique`
(`-45` degree pitch) and `nadir` (`-90` degree pitch) frames, then restores the
canonical `-45` degree pitch. Both frames have the same simulation step and
camera center but different render frame IDs. With zero camera roll, the top
of the nadir image is body-forward and its right edge is body-right.

One GTA instance must be controlled by one `DroneSimClient`. The client
serializes the complete paired operation against other threads using the same
client object; separate client processes are not supported concurrently.

## Lockstep simulation time

Lockstep is a separate research mode. Enter sets GTA gameplay time scale to
zero and explicitly freezes the controlled response actors, while leaving the
script thread, camera commands, rendering, network protocol, and Capture V3
operational. `advance_lockstep()` releases those actors, temporarily runs the
world until the next cumulative 250 ms target, then freezes both the clock and
actors again before replying. The boundary records each response actor's
kinematics before freezing and restores its linear velocity on release; the
matched realtime/lockstep validation below checks whether this is sufficient
to avoid artificial repeated starts.
Inference wall time and RGB-D transfer time therefore do not age response
actors.

The intended scenario order is:

```python
scenario_id = client.prepare_fire_scenario((x, y, z), seed=1)
client.wait_scenario_ready(scenario_id)
with LockstepSession(client) as lockstep:
    client.start_scenario(scenario_id)
    initial_time = lockstep.advance()  # t = 250 ms
    views = lockstep.capture_rgbd_pair()  # simulation remains frozen

    # Reset must happen before Exit so no scene resumes during cleanup.
    client.reset_scenario(scenario_id)
```

Only one lockstep session can exist. Session IDs are checked on every clock
operation. There is no heartbeat, automatic real-time fallback, or variable
step duration. During lockstep, manual camera movement is disabled. F11 is the
emergency recovery path: it resets the scenario, restores normal time, stops
the scripted camera, and restores the player.

## Controlled fire scenario

The experiment lifecycle is `PREPARING -> READY -> RUNNING`, followed by an
explicit Reset. Prepare snaps the requested anchor to a nearby road, resolves
a deterministic blueprint, loads models, clears the controlled area, and
places a damaged `blista`, `firetruk` actors, and civilian pedestrians. Start
records one GTA timer/frame origin, activates the fire, and issues each
response task exactly once.

The Python client exposes:

```python
scenario_id = client.prepare_fire_scenario(
    anchor_world_xyz=(x, y, z),
    seed=1,
    firetruck_count=1,
    pedestrian_count=32,
)
ready = client.wait_scenario_ready(scenario_id)
start = client.start_scenario(scenario_id)
snapshot = client.get_scenario_state(scenario_id)
client.reset_scenario(scenario_id)
```

The caller must first teleport the player to the event area so GTA has loaded
collision there. Scenario truth is intended for experiment control and
evaluation; it is not exposed by `RelativePoseController`.

The supplied master seed is deterministically expanded into independent
firetruck-placement, pedestrian-placement, and pedestrian-model random streams.
Prepare with `blueprint_id=0` creates a new immutable blueprint after clearing
the controlled area. Its ID is returned in `ScenarioSnapshot.blueprint_id` and
survives Reset in a single-slot cache. A later Prepare can pass that ID and
instantiate any prefix of its actor capacity:

```python
superset_id = client.prepare_fire_scenario(
    (x, y, z), seed=1, firetruck_count=1, pedestrian_count=32
)
superset = client.wait_scenario_ready(superset_id)
client.reset_scenario(superset_id)

no_truck_id = client.prepare_fire_scenario(
    (x, y, z),
    seed=1,
    firetruck_count=0,
    pedestrian_count=32,
    blueprint_id=superset.blueprint_id,
)
```

The second instance reuses the exact pedestrian positions, headings, and
models; it does not call GTA safe-coordinate queries again. A reuse request
must match the cached anchor and seed and cannot exceed its actor counts. The
seed controls scene construction, not GTA AI pathfinding or frame-by-frame
response trajectories.

## Online validation

Every validation consumes RGB-D in memory. By default no image, depth map,
video, PLY, PCD, point cloud, or trajectory is written to disk. Stage 2D has
an explicit optional recording mode for visual inspection; it stores
compressed RGB JPEGs and compact JSON metadata, never raw Depth or RGB-D.

```powershell
python validation\validate_camera_lifecycle.py
python validation\validate_pose_control.py
python validation\validate_rgbd_yaw_sync.py
python validation\validate_dual_view_rgbd.py --anchor X Y Z
python validation\validate_dual_view_rgbd.py --anchor X Y Z --pairs 1000
python validation\validate_dual_view_rgbd.py --anchor X Y Z --show-pointcloud
python validation\validate_rgbd_pointcloud.py --pixel-stride 4 --max-view-depth 200
python validation\validate_rgbd_stability.py --count 1000
python validation\validate_fire_scenario.py --anchor X Y Z
python validation\validate_fire_scenario.py --anchor X Y Z --cycles 50
python validation\validate_fire_scenario.py --anchor X Y Z --rgbd-captures 1000
python validation\validate_fire_scenario.py --anchor X Y Z --require-clean-area
python validation\validate_fire_scenario.py --anchor X Y Z --verify-seed-isolation
python validation\validate_lockstep_clock.py --anchor X Y Z
python validation\validate_visibility_starts.py --anchor X Y Z
python validation\validate_visibility_starts.py --anchor X Y Z --show-starts
python validation\validate_visibility_starts.py --anchor X Y Z --queries 1000
python validation\validate_spatiotemporal_feasibility.py --anchor X Y Z
python validation\validate_spatiotemporal_feasibility.py --anchor X Y Z --search-timeout 120
python validation\validate_spatiotemporal_feasibility.py --anchor X Y Z --record-dir recordings\stage2d_run
python validation\find_stage2d_witness.py --anchor X Y Z --stratum CUE_VISIBLE --attempts 10 --all-attempts
python validation\find_stage2d_witness.py --anchor X Y Z --stratum CUE_VISIBLE --attempts 10 --all-attempts --record-dir recordings\visible_witness
python validation\find_stage2d_witness.py --anchor X Y Z --stratum CUE_VISIBLE --attempts 10 --all-attempts --record-dir recordings\visible_witness_all --record-all-successes
python validation\find_stage2d_witness.py --anchor X Y Z --stratum CUE_HIDDEN --attempts 10 --horizon-steps 80 --search-timeout 300
python validation\visualize_spatiotemporal_trajectory.py recordings\stage2d_run
```

`--record-dir` must name a path that does not already exist. A successful run
creates `CUE_VISIBLE` and `CUE_HIDDEN` subdirectories. Each contains the
actual replay's oblique/nadir JPEG sequence and `trajectory.json` with the
current action, world/start-local camera pose, lockstep time, entity truth,
and projected target boxes. The offline player connects to no GTA process.
It plays both strata in sequence by default; use
`--stratum CUE_VISIBLE` or `--stratum CUE_HIDDEN` to select one. Space
pauses, Left/Right steps, Home/End jumps, and Q closes the window.

Stage 2D prints setup and fixed-action search progress separately. Its
`--search-timeout` is a wall-clock limit for each visibility stratum and
defaults to 120 seconds. A timeout returns `UNKNOWN`; it does not silently
switch planners or claim that the task is unreachable. `Ctrl+C` executes the
normal scenario/lockstep/player cleanup path; F11 remains the in-game emergency
recovery control.

`find_stage2d_witness.py` is an exploratory probability audit, not the formal
two-stratum Stage 2D acceptance test. It searches one requested stratum, reuses
one fire-scene blueprint, advances `start_seed` by one per attempt, and stops
after the first strict joint witness with the required action margin.
`--all-attempts` instead runs every attempt, structurally replays each
successful witness, and reports the witness rate; only the first successful
replay is recorded by default. The explicit `--record-all-successes` flag
stores every successful path in a separate attempt subdirectory and therefore
can grow substantially; it requires both `--all-attempts` and `--record-dir`.
Without `--record-dir`, the audit remains entirely in memory. Use
`--start-seed-step 0` to retry the same task start while sampling only GTA AI
trajectory variation. `--horizon-steps` and `--search-timeout` are explicit
exploration controls; they do not change the canonical 65-action definition.
A supplied `--record-dir` records only the successful replay and still writes
no Depth payload.

The lockstep validator also reuses one immutable scenario blueprint for a
continuous realtime run and a matched `250 ms` lockstep run. It compares
response-actor path length, directional progress, and median speed so that
repeated freezing cannot silently turn vehicle motion into a sequence of
artificial restarts.

Mission entities inside the controlled radius are never deleted. They are
preserved and reported separately in `snapshot.protected_entities`; the normal
validator prints a warning and continues. Formal clean-area collection should
use `--require-clean-area`. While a scenario is active, all ambient and
scenario-ped density multipliers are set to zero each frame. Existing ordinary
entities that cross into the 120-metre controlled area are removed by periodic
area maintenance; protected mission entities remain registered.

When the scripted camera is active, the fire-scenario validator moves it once,
after Prepare reaches READY, directly above the resolved event position. This
avoids a second visible relocation while the scene is starting. The default
observer view is 40 metres above the event at a -70 degree pitch. Use
`--camera-height` and `--camera-pitch` to adjust the overview;
`--camera-pitch -90` looks straight down. Pitch control is a validation utility
and does not change the agent-facing relative-pose action.

To validate collision rejection, choose a world coordinate beyond a nearby
wall or inside solid geometry. A simpler option is to place the camera in front
of a wall, face the wall, and test a point ten metres forward:

```powershell
python validation\validate_pose_control.py --expect-forward-collision 10
```

The explicit world-coordinate form remains available as
`--expect-collision X Y Z`.

The lifecycle test stops and recreates the camera, verifies the inactive error,
and captures immediately without sleeps. `validate_pose_control.py` alternates
two absolute positions and yaws, then checks that the returned pose and Capture
V3 view matrix agree. The yaw synchronization test distinguishes RGB and depth
from two views. The point-cloud test merges eight directions into an
interactive in-memory cloud. The stability test checks 1,000 captures,
monotonic frame IDs, finite metric depth, latency, and GTA process memory.

## Protocol boundary

The transport remains DSV3. Retained commands are:

- create/stop/query camera
- get pose and set absolute position plus yaw
- set FOV, time, and weather
- teleport/protect and restore the player
- synchronized RGB-D capture
- prepare/query/start/reset a controlled fire scenario
- enter/query/advance/exit the lockstep clock
- query evaluation-only visibility from a virtual camera center
- resolve a loaded, collision-clear task-start altitude
- batch-probe 2-metre-clear camera points and segments
- batch-query selected response targets from virtual camera centres
- ping

Capture V3 is unchanged. Visibility and start probing use message IDs 31 and
32. Stage 2D evaluation-only geometry and selected-target batches use message
IDs 33 and 34. Removed message IDs remain unsupported, with no compatibility
fallback for the old scene, recording, or discrete-action interfaces.
