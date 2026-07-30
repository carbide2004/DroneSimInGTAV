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

float camera_orientation_error(
    const RuntimePose& actual,
    float expected_pitch,
    float expected_yaw) {
    if (expected_pitch <= -90.0f + kAngleToleranceDegrees) {
        return angle_error(
            actual.yaw - actual.roll,
            expected_yaw);
    }
    if (expected_pitch >= 90.0f - kAngleToleranceDegrees) {
        return angle_error(
            actual.yaw + actual.roll,
            expected_yaw);
    }
    const float roll_error = angle_error(actual.roll, 0.0f);
    const float yaw_error = angle_error(actual.yaw, expected_yaw);
    return roll_error > yaw_error ? roll_error : yaw_error;
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
    canonical_pitch_degrees_ = 0.0f;
    canonical_yaw_degrees_ = wrap_degrees(
        ENTITY::GET_ENTITY_HEADING(player));
    camera_handle_ = CAM::CREATE_CAM_WITH_PARAMS(
        "DEFAULT_SCRIPTED_CAMERA",
        location.x,
        location.y,
        location.z,
        0.0f,
        0.0f,
        canonical_yaw_degrees_,
        40.0f,
        TRUE,
        2);
    if (camera_handle_ == 0 || !CAM::DOES_CAM_EXIST(camera_handle_)) {
        camera_handle_ = 0;
        error = "GTA failed to create the scripted camera";
        return false;
    }

    CAM::SET_CAM_ACTIVE(camera_handle_, TRUE);
    CAM::SET_CAM_MOTION_BLUR_STRENGTH(camera_handle_, 0.0f);
    CAM::SET_CAM_DOF_STRENGTH(camera_handle_, 0.0f);
    CAM::SET_CAM_USE_SHALLOW_DOF_MODE(camera_handle_, FALSE);
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
    canonical_pitch_degrees_ = 0.0f;
    canonical_yaw_degrees_ = 0.0f;
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

bool CameraController::get_capture_pose(
    RuntimePose& pose,
    std::string& error) const {
    RuntimePose actual;
    if (!get_pose(actual, error)) {
        return false;
    }
    const float pitch_error =
        std::fabs(actual.pitch - canonical_pitch_degrees_);
    const float orientation_error = camera_orientation_error(
        actual,
        canonical_pitch_degrees_,
        canonical_yaw_degrees_);
    if (pitch_error > kAngleToleranceDegrees ||
        orientation_error > kAngleToleranceDegrees) {
        error =
            "Camera is not at its canonical capture orientation; "
            "pitch error=" +
            std::to_string(pitch_error) +
            " deg, physical orientation error=" +
            std::to_string(orientation_error) +
            " deg";
        return false;
    }
    pose = actual;
    pose.pitch = canonical_pitch_degrees_;
    pose.roll = 0.0f;
    pose.yaw = canonical_yaw_degrees_;
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
    const float previous_yaw = canonical_yaw_degrees_;
    canonical_yaw_degrees_ = target_yaw;
    CAM::SET_CAM_COORD(camera_handle_, x, y, z);
    CAM::SET_CAM_ROT(
        camera_handle_,
        canonical_pitch_degrees_,
        0.0f,
        target_yaw,
        2);

    float position_error = 0.0f;
    float roll_error = 0.0f;
    float yaw_error = 0.0f;
    float orientation_error = 0.0f;
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
                canonical_pitch_degrees_,
                0.0f,
                previous_yaw,
                2);
            canonical_yaw_degrees_ = previous_yaw;
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
        roll_error = angle_error(actual_pose.roll, 0.0f);
        yaw_error = angle_error(actual_pose.yaw, target_yaw);
        orientation_error = camera_orientation_error(
            actual_pose,
            canonical_pitch_degrees_,
            target_yaw);
        if (position_error <= kPositionToleranceMeters &&
            orientation_error <= kAngleToleranceDegrees) {
            return CameraPoseStatus::Ok;
        }
    }

    error =
        "Requested camera pose did not settle within " +
        std::to_string(kMaximumPoseApplyFrames) +
        " game frames; position error=" +
        std::to_string(position_error) +
        " m, roll error=" +
        std::to_string(roll_error) +
        " deg, yaw error=" +
        std::to_string(yaw_error) +
        " deg, physical orientation error=" +
        std::to_string(orientation_error) +
        " deg";
    return CameraPoseStatus::PoseMismatch;
}

CameraPoseStatus CameraController::set_pitch(
    float pitch_degrees,
    const std::atomic<bool>& cancelled,
    RuntimePose& actual_pose,
    std::string& error) {
    if (!std::isfinite(pitch_degrees) ||
        pitch_degrees < -90.0f ||
        pitch_degrees > 90.0f) {
        error = "Camera pitch must be finite and within [-90, 90] degrees";
        return CameraPoseStatus::InvalidPose;
    }

    RuntimePose current;
    if (!get_pose(current, error)) {
        return CameraPoseStatus::CameraInactive;
    }
    if (cancelled.load(std::memory_order_acquire)) {
        error = "Camera pitch request was cancelled before application";
        return CameraPoseStatus::ApplyFailed;
    }

    const float previous_pitch = canonical_pitch_degrees_;
    canonical_pitch_degrees_ = pitch_degrees;
    CAM::SET_CAM_ROT(
        camera_handle_,
        pitch_degrees,
        0.0f,
        canonical_yaw_degrees_,
        2);

    float pitch_error = 0.0f;
    float roll_error = 0.0f;
    float yaw_error = 0.0f;
    float orientation_error = 0.0f;
    for (int frame = 0; frame < kMaximumPoseApplyFrames; ++frame) {
        WAIT(0);
        suppress_gameplay_controls_for_frame();
        if (cancelled.load(std::memory_order_acquire)) {
            CAM::SET_CAM_ROT(
                camera_handle_,
                previous_pitch,
                0.0f,
                canonical_yaw_degrees_,
                2);
            canonical_pitch_degrees_ = previous_pitch;
            error =
                "Camera pitch request was cancelled while awaiting GTA";
            return CameraPoseStatus::ApplyFailed;
        }
        if (!get_pose(actual_pose, error)) {
            error =
                "Camera became inactive while applying the requested pitch";
            return CameraPoseStatus::ApplyFailed;
        }
        pitch_error = std::fabs(actual_pose.pitch - pitch_degrees);
        roll_error = angle_error(actual_pose.roll, 0.0f);
        yaw_error = angle_error(
            actual_pose.yaw,
            canonical_yaw_degrees_);
        orientation_error = camera_orientation_error(
            actual_pose,
            pitch_degrees,
            canonical_yaw_degrees_);
        if (pitch_error <= kAngleToleranceDegrees &&
            orientation_error <= kAngleToleranceDegrees) {
            return CameraPoseStatus::Ok;
        }
    }

    error =
        "Requested camera pitch did not settle within " +
        std::to_string(kMaximumPoseApplyFrames) +
        " game frames; pitch error=" +
        std::to_string(pitch_error) +
        " deg, roll error=" +
        std::to_string(roll_error) +
        " deg, yaw error=" +
        std::to_string(yaw_error) +
        " deg, physical orientation error=" +
        std::to_string(orientation_error) +
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
