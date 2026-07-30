#pragma once

#include "scenario_types.h"
#include "simulation_clock.h"

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

enum class RuntimeCommandType {
    CreateCamera,
    StopCamera,
    GetCameraState,
    GetCameraPose,
    SetCameraPose,
    SetCameraPitch,
    SetFov,
    SetTime,
    SetWeather,
    TeleportPlayer,
    RestorePlayer,
    Capture,
    PrepareFireScenario,
    GetScenarioState,
    StartScenario,
    ResetScenario,
    EnterLockstep,
    GetLockstepState,
    AdvanceLockstep,
    ExitLockstep,
};

enum class RuntimeCommandStatus : std::uint32_t {
    Ok = 0,
    CameraInactive = 1,
    InvalidPose = 2,
    CollisionBlocked = 3,
    CommandTimeout = 4,
    PoseApplyFailed = 5,
    PoseMismatch = 6,
    InvalidRequest = 7,
    InternalError = 8,
    ScenarioAlreadyActive = 9,
    ScenarioNotFound = 10,
    ScenarioNotReady = 11,
    ScenarioAreaNotReady = 12,
    ScenarioPrepareFailed = 13,
    ScenarioStartFailed = 14,
    WorldAreaNotReady = 15,
    LockstepAlreadyActive = 16,
    LockstepNotActive = 17,
    LockstepSessionMismatch = 18,
    LockstepAdvanceTimeout = 19,
    LockstepInterrupted = 20,
    LockstepClockInvariantFailed = 21,
};

struct RuntimePose {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float pitch = 0.0f;
    float roll = 0.0f;
    float yaw = 0.0f;
};

struct RuntimeCommandResult {
    RuntimeCommandStatus status = RuntimeCommandStatus::InternalError;
    std::string message;
    RuntimePose pose;
    ScenarioSnapshot scenario_snapshot;
    ScenarioStartInfo scenario_start;
    LockstepSnapshot lockstep_snapshot;
    std::uint64_t value = 0;
    bool bool_value = false;
};

struct RuntimeCommand {
    RuntimeCommand(RuntimeCommandType command_type, std::uint64_t id)
        : type(command_type), request_id(id) {}

    RuntimeCommandType type;
    std::uint64_t request_id;
    std::array<float, 4> floats{};
    std::array<std::int32_t, 3> integers{};
    bool flag = false;
    std::string text;
    FireScenarioConfig fire_scenario_config;
    std::uint64_t scenario_id = 0;
    std::uint64_t lockstep_session_id = 0;

    std::atomic<bool> cancelled{false};
    std::mutex completion_mutex;
    std::condition_variable completion_cv;
    bool completed = false;
    RuntimeCommandResult result;
};

using RuntimeCommandPtr = std::shared_ptr<RuntimeCommand>;

RuntimeCommandPtr make_runtime_command(
    RuntimeCommandType type,
    std::uint64_t request_id);
void enqueue_command(const RuntimeCommandPtr& command);
bool try_dequeue_command(RuntimeCommandPtr& command);
bool wait_for_command(
    const RuntimeCommandPtr& command,
    std::chrono::milliseconds timeout,
    RuntimeCommandResult& result);
void complete_command(
    const RuntimeCommandPtr& command,
    RuntimeCommandResult result);
std::size_t command_queue_size();
