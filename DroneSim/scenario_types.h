#pragma once

#include <cstdint>
#include <string>
#include <vector>

enum class ScenarioLifecycle : std::uint32_t {
    Empty = 0,
    Preparing = 1,
    Ready = 2,
    Running = 3,
    Failed = 4,
};

enum class ScenarioEntityKind : std::uint32_t {
    Vehicle = 1,
    Pedestrian = 2,
};

enum class ScenarioEntityRole : std::uint32_t {
    FireSourceVehicle = 1,
    FireTruck = 2,
    FirefighterDriver = 3,
    FleeingPedestrian = 4,
};

enum class ScenarioTaskState : std::uint32_t {
    None = 0,
    Pending = 1,
    Active = 2,
    Succeeded = 3,
    Failed = 4,
    Lost = 5,
};

enum class ScenarioOperationStatus {
    Ok,
    AlreadyActive,
    InvalidConfig,
    AreaNotReady,
    IdMismatch,
    NotReady,
    PrepareFailed,
    StartFailed,
};

struct ScenarioVector3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

struct FireScenarioConfig {
    ScenarioVector3 anchor;
    std::uint64_t seed = 0;
    std::uint16_t firetruck_count = 1;
    std::uint16_t pedestrian_count = 32;
};

struct ScenarioStartInfo {
    std::uint64_t scenario_id = 0;
    std::uint32_t game_timer_ms = 0;
    std::uint32_t frame_count = 0;
};

struct ScenarioEntitySnapshot {
    std::uint64_t stable_id = 0;
    std::int32_t gta_handle = 0;
    std::uint32_t model_hash = 0;
    ScenarioEntityKind kind = ScenarioEntityKind::Vehicle;
    ScenarioEntityRole role =
        ScenarioEntityRole::FireSourceVehicle;
    std::uint64_t event_id = 0;
    ScenarioTaskState task_state = ScenarioTaskState::None;
    bool exists = false;
    ScenarioVector3 position;
    ScenarioVector3 velocity;
    float speed = 0.0f;
    float heading = 0.0f;
    std::uint32_t spawn_game_timer_ms = 0;
    std::uint32_t task_start_game_timer_ms = 0;
    std::uint32_t response_start_game_timer_ms = 0;
    ScenarioVector3 task_target;
};

struct ScenarioProtectedEntitySnapshot {
    std::int32_t gta_handle = 0;
    std::uint32_t model_hash = 0;
    ScenarioEntityKind kind = ScenarioEntityKind::Vehicle;
    bool exists = false;
    ScenarioVector3 position;
};

struct ScenarioSnapshot {
    std::uint64_t scenario_id = 0;
    std::uint64_t seed = 0;
    ScenarioLifecycle lifecycle = ScenarioLifecycle::Empty;
    std::uint32_t game_timer_ms = 0;
    std::uint32_t frame_count = 0;
    std::uint32_t start_game_timer_ms = 0;
    std::uint32_t start_frame_count = 0;
    ScenarioVector3 requested_anchor;
    ScenarioVector3 event_position;
    bool event_active = false;
    std::uint32_t removed_pedestrians = 0;
    std::uint32_t removed_vehicles = 0;
    std::uint32_t ambient_pedestrians = 0;
    std::uint32_t ambient_vehicles = 0;
    std::string failure_message;
    std::vector<ScenarioProtectedEntitySnapshot>
        protected_entities;
    std::vector<ScenarioEntitySnapshot> entities;
};
