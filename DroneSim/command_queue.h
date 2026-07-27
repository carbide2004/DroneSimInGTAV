#pragma once

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
    SetFov,
    SetTime,
    SetWeather,
    TeleportPlayer,
    RestorePlayer,
    Capture,
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
