#pragma once

#include "command_queue.h"
#include "types.h"

#include <atomic>
#include <string>

enum class CameraPoseStatus {
    Ok,
    CameraInactive,
    InvalidPose,
    CollisionBlocked,
    ApplyFailed,
    PoseMismatch,
};

class CameraController {
public:
    static CameraController& instance();

    bool create(std::uint64_t& camera_id, std::string& error);
    bool stop(std::string& error);
    bool is_active() const;
    bool get_pose(RuntimePose& pose, std::string& error) const;
    CameraPoseStatus set_pose(
        float x,
        float y,
        float z,
        float yaw_degrees,
        bool collision_check,
        const std::atomic<bool>& cancelled,
        RuntimePose& actual_pose,
        std::string& error);
    bool set_fov(float fov_degrees, std::string& error);

private:
    CameraController() = default;

    Any camera_handle_ = 0;
};
