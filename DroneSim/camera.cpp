#include "camera.h"

#include "logging.h"
#include "main.h"
#include "natives.h"

#include <cmath>

namespace {

constexpr float kPositionToleranceMeters = 1.0e-3f;
constexpr float kAngleToleranceDegrees = 1.0e-2f;
constexpr int kMaximumPoseApplyFrames = 120;
constexpr int kPlayerControlGroup = 0;
constexpr int kPauseControl = 199;
constexpr int kAlternatePauseControl = 200;

void suppress_gameplay_controls_for_frame() {
    CONTROLS::DISABLE_ALL_CONTROL_ACTIONS(kPlayerControlGroup);
    CONTROLS::ENABLE_CONTROL_ACTION(
        kPlayerControlGroup,
        kPauseControl,
        TRUE);
    CONTROLS::ENABLE_CONTROL_ACTION(
        kPlayerControlGroup,
        kAlternatePauseControl,
        TRUE);
}

bool finite_pose(float x, float y, float z, float yaw) {
    return std::isfinite(x) && std::isfinite(y) && std::isfinite(z) &&
        std::isfinite(yaw);
}

float wrap_degrees(float value) {
    float wrapped = std::fmod(value + 180.0f, 360.0f);
    if (wrapped < 0.0f) {
        wrapped += 360.0f;
    }
    return wrapped - 180.0f;
}

float angle_error(float actual, float expected) {
    return std::fabs(wrap_degrees(actual - expected));
}

enum class RaycastStatus {
    Clear,
    Hit,
    Failed,
};

RaycastStatus raycast_world(
    const RuntimePose& from,
    float to_x,
    float to_y,
    float to_z,
    const std::atomic<bool>& cancelled,
    std::string& error) {
    const int handle = WORLDPROBE::_0x7EE9F5D83DD4F90E(
        from.x,
        from.y,
        from.z,
        to_x,
        to_y,
        to_z,
        1,
        0,
        4);
    if (handle == 0) {
        error = "GTA did not create a world raycast";
        return RaycastStatus::Failed;
    }

    BOOL hit = FALSE;
    Vector3 hit_position{};
    Vector3 hit_normal{};
    Entity hit_entity = 0;
    for (int frame = 0; frame < 8; ++frame) {
        suppress_gameplay_controls_for_frame();
        if (cancelled.load(std::memory_order_acquire)) {
            error = "Camera pose request was cancelled";
            return RaycastStatus::Failed;
        }
        const int state = WORLDPROBE::_GET_RAYCAST_RESULT(
            handle,
            &hit,
            &hit_position,
            &hit_normal,
            &hit_entity);
        if (state == 2) {
            if (hit == TRUE) {
                error =
                    "World geometry blocks the requested camera segment at (" +
                    std::to_string(hit_position.x) + ", " +
                    std::to_string(hit_position.y) + ", " +
                    std::to_string(hit_position.z) + ")";
                return RaycastStatus::Hit;
            }
            return RaycastStatus::Clear;
        }
        if (state != 1) {
            error = "GTA returned an invalid world-raycast state";
            return RaycastStatus::Failed;
        }
        WAIT(0);
    }

    error = "World raycast did not complete within eight game frames";
    return RaycastStatus::Failed;
}

}  // namespace

CameraController& CameraController::instance() {
    static CameraController controller;
    return controller;
}

bool CameraController::create(
    std::uint64_t& camera_id,
    std::string& error) {
    if (is_active()) {
        camera_id = static_cast<std::uint64_t>(camera_handle_);
        return true;
    }
    if (camera_handle_ != 0 && CAM::DOES_CAM_EXIST(camera_handle_)) {
        CAM::DESTROY_CAM(camera_handle_, FALSE);
        camera_handle_ = 0;
    }

    const Ped player = PLAYER::PLAYER_PED_ID();
    if (player == 0 || !ENTITY::DOES_ENTITY_EXIST(player)) {
        error = "Player entity is unavailable";
        return false;
    }
    const Vector3 location =
        ENTITY::GET_OFFSET_FROM_ENTITY_IN_WORLD_COORDS(player, 0.0f, 0.0f, 10.0f);
    camera_handle_ = CAM::CREATE_CAM_WITH_PARAMS(
        "DEFAULT_SCRIPTED_CAMERA",
        location.x,
        location.y,
        location.z,
        0.0f,
        0.0f,
        ENTITY::GET_ENTITY_HEADING(player),
        40.0f,
        TRUE,
        2);
    if (camera_handle_ == 0 || !CAM::DOES_CAM_EXIST(camera_handle_)) {
        camera_handle_ = 0;
        error = "GTA failed to create the scripted camera";
        return false;
    }

    CAM::SET_CAM_ACTIVE(camera_handle_, TRUE);
    CAM::RENDER_SCRIPT_CAMS(TRUE, FALSE, 0, TRUE, FALSE);
    if (!CAM::IS_CAM_ACTIVE(camera_handle_)) {
        CAM::DESTROY_CAM(camera_handle_, FALSE);
        camera_handle_ = 0;
        error = "GTA created the scripted camera but did not activate it";
        return false;
    }

    camera_id = static_cast<std::uint64_t>(camera_handle_);
    LOGI(
        "camera",
        "Camera active at (" + std::to_string(location.x) + ", " +
            std::to_string(location.y) + ", " +
            std::to_string(location.z) + ")");
    return true;
}

bool CameraController::stop(std::string& error) {
    if (camera_handle_ == 0) {
        return true;
    }
    if (CAM::DOES_CAM_EXIST(camera_handle_)) {
        CAM::SET_CAM_ACTIVE(camera_handle_, FALSE);
        CAM::RENDER_SCRIPT_CAMS(FALSE, FALSE, 0, TRUE, FALSE);
        CAM::DESTROY_CAM(camera_handle_, FALSE);
    }
    camera_handle_ = 0;
    LOGI("camera", "Scripted camera stopped");
    return true;
}

bool CameraController::is_active() const {
    return camera_handle_ != 0 &&
        CAM::DOES_CAM_EXIST(camera_handle_) &&
        CAM::IS_CAM_ACTIVE(camera_handle_);
}

void CameraController::suppress_player_controls_for_frame() const {
    if (is_active()) {
        suppress_gameplay_controls_for_frame();
    }
}

bool CameraController::get_pose(
    RuntimePose& pose,
    std::string& error) const {
    if (!is_active()) {
        error = "Scripted camera is inactive";
        return false;
    }
    const Vector3 position = CAM::GET_CAM_COORD(camera_handle_);
    const Vector3 rotation = CAM::GET_CAM_ROT(camera_handle_, 2);
    pose.x = position.x;
    pose.y = position.y;
    pose.z = position.z;
    pose.pitch = rotation.x;
    pose.roll = rotation.y;
    pose.yaw = rotation.z;
    if (!finite_pose(pose.x, pose.y, pose.z, pose.yaw) ||
        !std::isfinite(pose.pitch) || !std::isfinite(pose.roll)) {
        error = "GTA returned a non-finite camera pose";
        return false;
    }
    return true;
}

CameraPoseStatus CameraController::set_pose(
    float x,
    float y,
    float z,
    float yaw_degrees,
    bool collision_check,
    const std::atomic<bool>& cancelled,
    RuntimePose& actual_pose,
    std::string& error) {
    if (!finite_pose(x, y, z, yaw_degrees)) {
        error = "Camera position and yaw must be finite";
        return CameraPoseStatus::InvalidPose;
    }

    RuntimePose current;
    if (!get_pose(current, error)) {
        return CameraPoseStatus::CameraInactive;
    }

    const float segment_x = x - current.x;
    const float segment_y = y - current.y;
    const float segment_z = z - current.z;
    const float segment_length_squared =
        segment_x * segment_x +
        segment_y * segment_y +
        segment_z * segment_z;
    if (collision_check && segment_length_squared > 1.0e-8f) {
        const RaycastStatus raycast =
            raycast_world(current, x, y, z, cancelled, error);
        if (raycast == RaycastStatus::Hit) {
            return CameraPoseStatus::CollisionBlocked;
        }
        if (raycast == RaycastStatus::Failed) {
            return CameraPoseStatus::ApplyFailed;
        }
    }

    if (cancelled.load(std::memory_order_acquire)) {
        error = "Camera pose request was cancelled before application";
        return CameraPoseStatus::ApplyFailed;
    }
    const float target_yaw = wrap_degrees(yaw_degrees);
    CAM::SET_CAM_COORD(camera_handle_, x, y, z);
    CAM::SET_CAM_ROT(
        camera_handle_,
        current.pitch,
        current.roll,
        target_yaw,
        2);

    float position_error = 0.0f;
    float yaw_error = 0.0f;
    for (int frame = 0; frame < kMaximumPoseApplyFrames; ++frame) {
        // GTA camera writes become observable on a later game tick. Yield to
        // the engine and poll the actual pose instead of sleeping for a fixed
        // wall-clock duration.
        WAIT(0);
        suppress_gameplay_controls_for_frame();
        if (cancelled.load(std::memory_order_acquire)) {
            CAM::SET_CAM_COORD(
                camera_handle_,
                current.x,
                current.y,
                current.z);
            CAM::SET_CAM_ROT(
                camera_handle_,
                current.pitch,
                current.roll,
                current.yaw,
                2);
            error = "Camera pose request was cancelled while awaiting GTA";
            return CameraPoseStatus::ApplyFailed;
        }
        if (!get_pose(actual_pose, error)) {
            error =
                "Camera became inactive while applying the requested pose";
            return CameraPoseStatus::ApplyFailed;
        }

        const float dx = actual_pose.x - x;
        const float dy = actual_pose.y - y;
        const float dz = actual_pose.z - z;
        position_error = std::sqrt(dx * dx + dy * dy + dz * dz);
        yaw_error = angle_error(actual_pose.yaw, target_yaw);
        if (position_error <= kPositionToleranceMeters &&
            yaw_error <= kAngleToleranceDegrees) {
            return CameraPoseStatus::Ok;
        }
    }

    error =
        "Requested camera pose did not settle within " +
        std::to_string(kMaximumPoseApplyFrames) +
        " game frames; position error=" +
        std::to_string(position_error) +
        " m, yaw error=" +
        std::to_string(yaw_error) +
        " deg";
    return CameraPoseStatus::PoseMismatch;
}

bool CameraController::set_fov(
    float fov_degrees,
    std::string& error) {
    if (!is_active()) {
        error = "Scripted camera is inactive";
        return false;
    }
    if (!std::isfinite(fov_degrees) ||
        fov_degrees <= 1.0f ||
        fov_degrees >= 179.0f) {
        error = "FOV must be finite and within (1, 179) degrees";
        return false;
    }
    CAM::SET_CAM_FOV(camera_handle_, fov_degrees);
    const float actual = CAM::GET_CAM_FOV(camera_handle_);
    if (!std::isfinite(actual) ||
        std::fabs(actual - fov_degrees) > 1.0e-2f) {
        error = "GTA did not apply the requested camera FOV";
        return false;
    }
    return true;
}
