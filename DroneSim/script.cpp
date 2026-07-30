#include "script.h"

#include "camera.h"
#include "command_queue.h"
#include "keyboard.h"
#include "logging.h"
#include "main.h"
#include "natives.h"
#include "rgbd_capture.h"
#include "scenario_manager.h"
#include "server.h"
#include "simulation_clock.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <exception>
#include <string>
#include <utility>

namespace {

constexpr float kManualTranslationStepMeters = 1.0f;
constexpr float kManualYawStepDegrees = 15.0f;
constexpr float kDegreesToRadians = 0.01745329251994329577f;

void show_notification(const std::string& text);
void perform_lockstep_emergency_recovery();

RuntimeCommandResult ok_result() {
    RuntimeCommandResult result;
    result.status = RuntimeCommandStatus::Ok;
    return result;
}

RuntimeCommandResult error_result(
    RuntimeCommandStatus status,
    std::string message) {
    RuntimeCommandResult result;
    result.status = status;
    result.message = std::move(message);
    return result;
}

RuntimeCommandStatus map_pose_status(CameraPoseStatus status) {
    switch (status) {
        case CameraPoseStatus::Ok:
            return RuntimeCommandStatus::Ok;
        case CameraPoseStatus::CameraInactive:
            return RuntimeCommandStatus::CameraInactive;
        case CameraPoseStatus::InvalidPose:
            return RuntimeCommandStatus::InvalidPose;
        case CameraPoseStatus::CollisionBlocked:
            return RuntimeCommandStatus::CollisionBlocked;
        case CameraPoseStatus::ApplyFailed:
            return RuntimeCommandStatus::PoseApplyFailed;
        case CameraPoseStatus::PoseMismatch:
            return RuntimeCommandStatus::PoseMismatch;
        default:
            return RuntimeCommandStatus::InternalError;
    }
}

RuntimeCommandStatus map_scenario_status(
    ScenarioOperationStatus status) {
    switch (status) {
        case ScenarioOperationStatus::Ok:
            return RuntimeCommandStatus::Ok;
        case ScenarioOperationStatus::AlreadyActive:
            return RuntimeCommandStatus::ScenarioAlreadyActive;
        case ScenarioOperationStatus::InvalidConfig:
            return RuntimeCommandStatus::InvalidRequest;
        case ScenarioOperationStatus::AreaNotReady:
            return RuntimeCommandStatus::ScenarioAreaNotReady;
        case ScenarioOperationStatus::IdMismatch:
            return RuntimeCommandStatus::ScenarioNotFound;
        case ScenarioOperationStatus::NotReady:
            return RuntimeCommandStatus::ScenarioNotReady;
        case ScenarioOperationStatus::PrepareFailed:
            return RuntimeCommandStatus::ScenarioPrepareFailed;
        case ScenarioOperationStatus::StartFailed:
            return RuntimeCommandStatus::ScenarioStartFailed;
        default:
            return RuntimeCommandStatus::InternalError;
    }
}

RuntimeCommandStatus map_lockstep_status(
    LockstepOperationStatus status) {
    switch (status) {
        case LockstepOperationStatus::Ok:
            return RuntimeCommandStatus::Ok;
        case LockstepOperationStatus::AlreadyActive:
            return RuntimeCommandStatus::LockstepAlreadyActive;
        case LockstepOperationStatus::NotActive:
            return RuntimeCommandStatus::LockstepNotActive;
        case LockstepOperationStatus::SessionMismatch:
            return RuntimeCommandStatus::LockstepSessionMismatch;
        case LockstepOperationStatus::AdvanceTimeout:
            return RuntimeCommandStatus::LockstepAdvanceTimeout;
        case LockstepOperationStatus::Interrupted:
            return RuntimeCommandStatus::LockstepInterrupted;
        case LockstepOperationStatus::InvariantFailed:
            return RuntimeCommandStatus::LockstepClockInvariantFailed;
        default:
            return RuntimeCommandStatus::InternalError;
    }
}

RuntimeCommandResult submit_capture_camera(std::uint64_t request_id) {
    if (!CameraController::instance().is_active()) {
        return error_result(
            RuntimeCommandStatus::CameraInactive,
            "RGB-D capture requires an active scripted camera");
    }
    const Any camera = CAM::GET_RENDERING_CAM();
    if (camera == 0 || !CAM::IS_CAM_ACTIVE(camera)) {
        return error_result(
            RuntimeCommandStatus::CameraInactive,
            "GTA has no active rendering camera");
    }

    const Vector3 position = CAM::GET_CAM_COORD(camera);
    const Vector3 rotation = CAM::GET_CAM_ROT(camera, 2);
    CaptureCamera capture_camera;
    std::string error;
    const CaptureStatus status = build_capture_camera(
        CAM::GET_CAM_FOV(camera),
        CAM::GET_CAM_NEAR_CLIP(camera),
        CAM::GET_CAM_FAR_CLIP(camera),
        position.x,
        position.y,
        position.z,
        rotation.x,
        rotation.y,
        rotation.z,
        capture_camera,
        error);
    if (status != CaptureStatus::Ok) {
        return error_result(
            RuntimeCommandStatus::InvalidPose,
            "Invalid capture camera: " + error);
    }
    RgbdCapture::instance().submit_camera(request_id, capture_camera);
    return ok_result();
}

RuntimeCommandResult teleport_player(const RuntimeCommand& command) {
    if (ScenarioManager::instance().lifecycle() !=
        ScenarioLifecycle::Empty) {
        return error_result(
            RuntimeCommandStatus::ScenarioAlreadyActive,
            "Reset the active scenario before teleporting the player");
    }
    const float x = command.floats[0];
    const float y = command.floats[1];
    const float z = command.floats[2];
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
        return error_result(
            RuntimeCommandStatus::InvalidPose,
            "Player position must be finite");
    }
    const Ped player = PLAYER::PLAYER_PED_ID();
    if (player == 0 || !ENTITY::DOES_ENTITY_EXIST(player)) {
        return error_result(
            RuntimeCommandStatus::InternalError,
            "Player entity is unavailable");
    }

    PLAYER::SET_PLAYER_INVINCIBLE(PLAYER::PLAYER_ID(), TRUE);
    ENTITY::SET_ENTITY_VISIBLE(player, FALSE, FALSE);
    ENTITY::SET_ENTITY_CAN_BE_DAMAGED(player, FALSE);
    ENTITY::SET_ENTITY_PROOFS(
        player,
        TRUE,
        TRUE,
        TRUE,
        TRUE,
        TRUE,
        TRUE,
        TRUE,
        TRUE);
    PLAYER::SET_MAX_WANTED_LEVEL(0);
    PLAYER::SET_POLICE_IGNORE_PLAYER(PLAYER::PLAYER_ID(), TRUE);
    ENTITY::SET_ENTITY_COORDS(player, x, y, z, TRUE, FALSE, FALSE, TRUE);

    const auto deadline =
        std::chrono::steady_clock::now() +
        std::chrono::seconds(5);
    while (std::chrono::steady_clock::now() < deadline) {
        if (command.cancelled.load(std::memory_order_acquire)) {
            return error_result(
                RuntimeCommandStatus::CommandTimeout,
                "Player teleport was cancelled while loading collision");
        }
        STREAMING::REQUEST_COLLISION_AT_COORD(x, y, z);
        if (ENTITY::HAS_COLLISION_LOADED_AROUND_ENTITY(player)) {
            return ok_result();
        }
        CameraController::instance()
            .suppress_player_controls_for_frame();
        WAIT(0);
    }
    return error_result(
        RuntimeCommandStatus::WorldAreaNotReady,
        "Collision did not load around the teleported player within 5 seconds");
}

RuntimeCommandResult restore_player() {
    const Ped player = PLAYER::PLAYER_PED_ID();
    if (player == 0 || !ENTITY::DOES_ENTITY_EXIST(player)) {
        return error_result(
            RuntimeCommandStatus::InternalError,
            "Player entity is unavailable");
    }
    PLAYER::SET_PLAYER_INVINCIBLE(PLAYER::PLAYER_ID(), FALSE);
    ENTITY::SET_ENTITY_VISIBLE(player, TRUE, FALSE);
    ENTITY::SET_ENTITY_CAN_BE_DAMAGED(player, TRUE);
    ENTITY::SET_ENTITY_PROOFS(
        player,
        FALSE,
        FALSE,
        FALSE,
        FALSE,
        FALSE,
        FALSE,
        FALSE,
        FALSE);
    PLAYER::SET_MAX_WANTED_LEVEL(5);
    PLAYER::SET_POLICE_IGNORE_PLAYER(PLAYER::PLAYER_ID(), FALSE);
    return ok_result();
}

RuntimeCommandResult execute_command(const RuntimeCommand& command) {
    auto& camera = CameraController::instance();
    std::string error;

    switch (command.type) {
        case RuntimeCommandType::CreateCamera: {
            RuntimeCommandResult result = ok_result();
            if (!camera.create(result.value, error)) {
                return error_result(
                    RuntimeCommandStatus::InternalError,
                    error);
            }
            return result;
        }
        case RuntimeCommandType::StopCamera:
            if (!camera.stop(error)) {
                return error_result(
                    RuntimeCommandStatus::InternalError,
                    error);
            }
            return ok_result();
        case RuntimeCommandType::GetCameraState: {
            RuntimeCommandResult result = ok_result();
            result.bool_value = camera.is_active();
            return result;
        }
        case RuntimeCommandType::GetCameraPose: {
            RuntimeCommandResult result = ok_result();
            if (!camera.get_pose(result.pose, error)) {
                return error_result(
                    RuntimeCommandStatus::CameraInactive,
                    error);
            }
            return result;
        }
        case RuntimeCommandType::SetCameraPose: {
            RuntimeCommandResult result = ok_result();
            const CameraPoseStatus status = camera.set_pose(
                command.floats[0],
                command.floats[1],
                command.floats[2],
                command.floats[3],
                command.flag,
                command.cancelled,
                result.pose,
                error);
            result.status = map_pose_status(status);
            result.message = error;
            return result;
        }
        case RuntimeCommandType::SetCameraPitch: {
            RuntimeCommandResult result = ok_result();
            const CameraPoseStatus status = camera.set_pitch(
                command.floats[0],
                command.cancelled,
                result.pose,
                error);
            result.status = map_pose_status(status);
            result.message = error;
            return result;
        }
        case RuntimeCommandType::SetFov:
            if (!camera.is_active()) {
                return error_result(
                    RuntimeCommandStatus::CameraInactive,
                    "Scripted camera is inactive");
            }
            if (!camera.set_fov(command.floats[0], error)) {
                return error_result(
                    RuntimeCommandStatus::InvalidRequest,
                    error);
            }
            return ok_result();
        case RuntimeCommandType::SetTime:
            TIME::SET_CLOCK_TIME(
                command.integers[0],
                command.integers[1],
                command.integers[2]);
            return ok_result();
        case RuntimeCommandType::SetWeather:
            if (command.text.empty()) {
                return error_result(
                    RuntimeCommandStatus::InvalidRequest,
                    "Weather name cannot be empty");
            }
            GAMEPLAY::CLEAR_WEATHER_TYPE_PERSIST();
            GAMEPLAY::SET_WEATHER_TYPE_NOW_PERSIST(
                const_cast<char*>(command.text.c_str()));
            return ok_result();
        case RuntimeCommandType::TeleportPlayer:
            return teleport_player(command);
        case RuntimeCommandType::RestorePlayer:
            return restore_player();
        case RuntimeCommandType::Capture:
            return submit_capture_camera(command.request_id);
        case RuntimeCommandType::PrepareFireScenario: {
            RuntimeCommandResult result = ok_result();
            std::uint64_t scenario_id = 0;
            const ScenarioOperationStatus status =
                ScenarioManager::instance().prepare_fire(
                    command.fire_scenario_config,
                    scenario_id,
                    error);
            result.status = map_scenario_status(status);
            result.message = error;
            result.value = scenario_id;
            return result;
        }
        case RuntimeCommandType::GetScenarioState: {
            RuntimeCommandResult result = ok_result();
            const ScenarioOperationStatus status =
                ScenarioManager::instance().snapshot(
                    command.scenario_id,
                    result.scenario_snapshot,
                    error);
            result.status = map_scenario_status(status);
            result.message = error;
            return result;
        }
        case RuntimeCommandType::StartScenario: {
            RuntimeCommandResult result = ok_result();
            const ScenarioOperationStatus status =
                ScenarioManager::instance().start(
                    command.scenario_id,
                    result.scenario_start,
                    error);
            result.status = map_scenario_status(status);
            result.message = error;
            return result;
        }
        case RuntimeCommandType::ResetScenario: {
            RuntimeCommandResult result = ok_result();
            const ScenarioOperationStatus status =
                ScenarioManager::instance().reset(
                    command.scenario_id,
                    error);
            result.status = map_scenario_status(status);
            result.message = error;
            return result;
        }
        case RuntimeCommandType::EnterLockstep: {
            RuntimeCommandResult result = ok_result();
            const LockstepOperationStatus status =
                SimulationClock::instance().enter(
                    result.lockstep_snapshot,
                    error);
            result.status = map_lockstep_status(status);
            result.message = error;
            return result;
        }
        case RuntimeCommandType::GetLockstepState: {
            RuntimeCommandResult result = ok_result();
            const LockstepOperationStatus status =
                SimulationClock::instance().snapshot(
                    command.lockstep_session_id,
                    result.lockstep_snapshot,
                    error);
            result.status = map_lockstep_status(status);
            result.message = error;
            return result;
        }
        case RuntimeCommandType::AdvanceLockstep: {
            RuntimeCommandResult result = ok_result();
            const LockstepOperationStatus status =
                SimulationClock::instance().advance(
                    command.lockstep_session_id,
                    command.cancelled,
                    result.lockstep_snapshot,
                    error);
            result.status = map_lockstep_status(status);
            result.message = error;
            return result;
        }
        case RuntimeCommandType::ExitLockstep: {
            RuntimeCommandResult result = ok_result();
            const LockstepOperationStatus status =
                SimulationClock::instance().exit(
                    command.lockstep_session_id,
                    error);
            result.status = map_lockstep_status(status);
            result.message = error;
            return result;
        }
        default:
            return error_result(
                RuntimeCommandStatus::InvalidRequest,
                "Unsupported runtime command");
    }
}

void process_command(const RuntimeCommandPtr& command) {
    if (!command ||
        command->cancelled.load(std::memory_order_acquire)) {
        return;
    }
    RuntimeCommandResult result;
    try {
        result = execute_command(*command);
    } catch (const std::exception& exception) {
        result = error_result(
            RuntimeCommandStatus::InternalError,
            exception.what());
    } catch (...) {
        result = error_result(
            RuntimeCommandStatus::InternalError,
            "Unknown GTA script-thread exception");
    }
    if (SimulationClock::instance()
            .take_emergency_recovery_request()) {
        perform_lockstep_emergency_recovery();
    }
    complete_command(command, std::move(result));
}

void show_notification(const std::string& text) {
    UI::_SET_NOTIFICATION_TEXT_ENTRY("STRING");
    UI::_ADD_TEXT_COMPONENT_STRING(
        const_cast<char*>(text.c_str()));
    UI::_DRAW_NOTIFICATION(FALSE, TRUE);
}

void discard_manual_camera_presses() {
    MoveForward.consume_press();
    MoveBackward.consume_press();
    StrafeRight.consume_press();
    StrafeLeft.consume_press();
    MoveUp.consume_press();
    MoveDown.consume_press();
    YawLeft.consume_press();
    YawRight.consume_press();
}

void perform_lockstep_emergency_recovery() {
    ScenarioManager::instance().force_reset();
    SimulationClock::instance().force_exit();

    std::string error;
    bool succeeded = true;
    if (!CameraController::instance().stop(error)) {
        succeeded = false;
        LOGE(
            "script",
            "Lockstep emergency camera stop failed: " + error);
    }
    const RuntimeCommandResult restore_result = restore_player();
    if (restore_result.status != RuntimeCommandStatus::Ok) {
        succeeded = false;
        LOGE(
            "script",
            "Lockstep emergency player restore failed: " +
                restore_result.message);
    }
    discard_manual_camera_presses();
    if (succeeded) {
        show_notification(
            "Lockstep stopped; scenario, camera and player restored");
        LOGI("script", "Completed F11 lockstep emergency recovery");
    } else {
        show_notification(
            "Lockstep stopped with recovery errors; check DroneSim.log");
    }
}

void process_manual_camera_controls() {
    float forward_steps = 0.0f;
    float right_steps = 0.0f;
    float vertical_steps = 0.0f;
    float yaw_steps = 0.0f;

    if (MoveForward.consume_press()) {
        forward_steps += 1.0f;
    }
    if (MoveBackward.consume_press()) {
        forward_steps -= 1.0f;
    }
    if (StrafeRight.consume_press()) {
        right_steps += 1.0f;
    }
    if (StrafeLeft.consume_press()) {
        right_steps -= 1.0f;
    }
    if (MoveUp.consume_press()) {
        vertical_steps += 1.0f;
    }
    if (MoveDown.consume_press()) {
        vertical_steps -= 1.0f;
    }
    if (YawLeft.consume_press()) {
        yaw_steps += 1.0f;
    }
    if (YawRight.consume_press()) {
        yaw_steps -= 1.0f;
    }

    if (forward_steps == 0.0f &&
        right_steps == 0.0f &&
        vertical_steps == 0.0f &&
        yaw_steps == 0.0f) {
        return;
    }

    auto& camera = CameraController::instance();
    RuntimePose current;
    std::string error;
    if (!camera.get_pose(current, error)) {
        show_notification(
            "DroneSim camera is inactive; press F10 first");
        return;
    }

    const float yaw_radians = current.yaw * kDegreesToRadians;
    const float forward_x = -std::sin(yaw_radians);
    const float forward_y = std::cos(yaw_radians);
    const float right_x = std::cos(yaw_radians);
    const float right_y = std::sin(yaw_radians);
    const float forward_distance =
        forward_steps * kManualTranslationStepMeters;
    const float right_distance =
        right_steps * kManualTranslationStepMeters;
    const float vertical_distance =
        vertical_steps * kManualTranslationStepMeters;
    const float target_x =
        current.x +
        forward_x * forward_distance +
        right_x * right_distance;
    const float target_y =
        current.y +
        forward_y * forward_distance +
        right_y * right_distance;
    const float target_z = current.z + vertical_distance;
    const float target_yaw =
        current.yaw + yaw_steps * kManualYawStepDegrees;
    const bool position_changed =
        forward_steps != 0.0f ||
        right_steps != 0.0f ||
        vertical_steps != 0.0f;

    std::atomic<bool> never_cancelled{false};
    RuntimePose actual;
    const CameraPoseStatus status = camera.set_pose(
        target_x,
        target_y,
        target_z,
        target_yaw,
        position_changed,
        never_cancelled,
        actual,
        error);
    if (status == CameraPoseStatus::CollisionBlocked) {
        show_notification("Drone camera movement blocked by collision");
        return;
    }
    if (status != CameraPoseStatus::Ok) {
        show_notification("Drone camera control failed: " + error);
        LOGE("script", "Manual camera control failed: " + error);
    }
}

void process_keyboard() {
    std::string error;
    if (F9.consume_press()) {
        RuntimePose pose;
        if (CameraController::instance().get_pose(pose, error)) {
            char text[256]{};
            std::snprintf(
                text,
                sizeof(text),
                "Drone pose | X %.2f  Y %.2f  Z %.2f | "
                "Pitch %.1f  Roll %.1f  Yaw %.1f",
                pose.x,
                pose.y,
                pose.z,
                pose.pitch,
                pose.roll,
                pose.yaw);
            show_notification(text);
            LOGI("script", text);
        } else {
            show_notification(
                "DroneSim camera is inactive; press F10 first");
        }
    }
    if (F11.consume_press()) {
        if (SimulationClock::instance().is_active()) {
            perform_lockstep_emergency_recovery();
            return;
        }
        bool succeeded = true;
        if (!CameraController::instance().stop(error)) {
            succeeded = false;
            LOGE("script", "F11 camera stop failed: " + error);
            show_notification(
                "DroneSim camera stop failed: " + error);
        }
        const RuntimeCommandResult restore_result =
            restore_player();
        if (restore_result.status != RuntimeCommandStatus::Ok) {
            succeeded = false;
            LOGE(
                "script",
                "F11 player restore failed: " +
                    restore_result.message);
            show_notification(
                "DroneSim player restore failed: " +
                    restore_result.message);
        }
        if (succeeded) {
            show_notification(
                "DroneSim camera stopped; player restored");
        }
        return;
    }
    if (SimulationClock::instance().is_active()) {
        if (F10.consume_press()) {
            show_notification(
                "F10 is disabled while lockstep is active; use F11");
        }
        discard_manual_camera_presses();
        return;
    }
    if (F10.consume_press()) {
        std::uint64_t camera_id = 0;
        if (!CameraController::instance().create(camera_id, error)) {
            LOGE("script", "F10 camera creation failed: " + error);
            show_notification(
                "DroneSim camera creation failed: " + error);
        } else {
            show_notification("DroneSim camera mode active");
        }
    }
    process_manual_camera_controls();
}

}  // namespace

void scriptMain() {
    InitializeServer();
    LOGI("script", "DroneSim Stage 1 runtime started");

    while (true) {
        CameraController::instance().suppress_player_controls_for_frame();
        process_keyboard();

        RuntimeCommandPtr command;
        while (try_dequeue_command(command)) {
            process_command(command);
        }
        ScenarioManager::instance().tick();
        WAIT(0);
    }
}
