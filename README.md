# DroneSimInGTAV

DroneSimInGTAV is a small GTA V research runtime for controlling a scripted
camera and acquiring synchronized RGB-D observations. The current Stage 0
codebase intentionally contains no anomaly scenarios, trajectory recorder,
navigation policy, training pipeline, or dataset conversion code.

The research direction is hidden-event localization from event-induced dynamic
responses. See [docs/research_direction.md](docs/research_direction.md).

## Repository layout

```text
DroneSim/                  GTA V ASI plugin
  camera.*                 camera lifecycle and absolute pose control
  command_queue.*          request-correlated GTA-thread commands
  rgbd_capture.*           synchronized Capture V3 implementation
  script.*                 minimal GTA script runtime
  server.*                 DSV3 TCP protocol server
agent_control/
  dronesim_client.py       strict Python client and relative-pose wrapper
  requirements.txt         online validation dependencies
validation/
  validate_pose_control.py
  validate_rgbd_pointcloud.py
  validate_rgbd_stability.py
  validate_rgbd_yaw_sync.py
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
- F11 stops the scripted camera.
- W/S move forward/backward by 1 metre.
- A/D strafe left/right by 1 metre.
- Q/E turn left/right by 15 degrees.
- Z/C move up/down by 1 metre.

Manual translation uses the same collision check as the network pose API.
Each physical key press performs one step; holding a key does not repeatedly
move the camera.

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
    RelativePoseController,
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

# Agent-facing wrapper: forward, right, vertical, yaw delta.
controller = RelativePoseController(client, collision_check=True)
controller.synchronize()
controller.step_relative(0.5, 0.0, 0.0, 10.0)
```

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

Capture keeps the existing V3 response: request/frame IDs, RGB, metric depth,
FOV, clip planes, projection matrix, and world-to-view matrix.

## Online validation

Every validation consumes RGB-D in memory. No image, depth map, video, PLY,
PCD, or point cloud is written to disk.

```powershell
python validation\validate_camera_lifecycle.py
python validation\validate_pose_control.py
python validation\validate_rgbd_yaw_sync.py
python validation\validate_rgbd_pointcloud.py --pixel-stride 4 --max-view-depth 200
python validation\validate_rgbd_stability.py --count 1000
```

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
- ping

Removed message IDs are unsupported. There is no compatibility fallback for
the old scene, recording, or discrete-action interfaces.
