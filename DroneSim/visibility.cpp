#include "visibility.h"

#include "camera.h"
#include "keyboard.h"
#include "main.h"
#include "natives.h"
#include "scenario_manager.h"
#include "simulation_clock.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

namespace {

constexpr int kRaycastFlags = 511;
constexpr int kRaycastOption = 4;
constexpr std::size_t kMaximumOutstandingRaycasts = 8;
constexpr int kMaximumRaycastFrames = 8;
constexpr float kMinimumModelExtentMeters = 0.01f;
constexpr float kDuplicatePointToleranceSquared = 1.0e-4f;
constexpr float kFireEnvelopeRadiusMeters = 8.0f;
constexpr float kFireEnvelopeHeightMeters = 25.0f;
constexpr float kCameraClearanceRadiusMeters = 1.0f;
constexpr float kProfileClearanceRadiusMeters = 2.0f;
constexpr std::size_t kMaximumGeometryBatchItems = 256;
constexpr std::size_t kMaximumTargetVisibilityCases = 64;
constexpr float kPi = 3.14159265358979323846f;

constexpr std::array<int, 8> kPedestrianBones = {
    31086,  // SKEL_Head
    24816,  // SKEL_Spine2
    11816,  // SKEL_Pelvis
    18905,  // SKEL_L_Hand
    57005,  // SKEL_R_Hand
    14201,  // SKEL_L_Foot
    52301,  // SKEL_R_Foot
    39317,  // SKEL_Neck_1
};

bool finite_vector(const ScenarioVector3& value) {
    return std::isfinite(value.x) &&
        std::isfinite(value.y) &&
        std::isfinite(value.z);
}

bool evaluation_interrupted(
    const std::atomic<bool>& cancelled) {
    if (cancelled.load(std::memory_order_acquire)) {
        return true;
    }
    if (F11.consume_press()) {
        SimulationClock::instance().request_emergency_recovery();
        return true;
    }
    return false;
}

ScenarioVector3 to_scenario_vector(const Vector3& value) {
    return {value.x, value.y, value.z};
}

float distance_squared(
    const ScenarioVector3& left,
    const ScenarioVector3& right) {
    const float dx = left.x - right.x;
    const float dy = left.y - right.y;
    const float dz = left.z - right.z;
    return dx * dx + dy * dy + dz * dz;
}

bool append_unique_point(
    std::vector<VisibilitySampleSnapshot>& samples,
    const ScenarioVector3& position) {
    if (!finite_vector(position)) {
        return false;
    }
    for (const VisibilitySampleSnapshot& existing : samples) {
        if (distance_squared(existing.position, position) <=
            kDuplicatePointToleranceSquared) {
            return false;
        }
    }
    VisibilitySampleSnapshot sample;
    sample.position = position;
    samples.push_back(sample);
    return true;
}

bool build_vehicle_samples(
    const ScenarioEntitySnapshot& entity,
    VisibilityTargetRole role,
    VisibilityTargetSnapshot& target,
    std::string& error) {
    Vector3 minimum{};
    Vector3 maximum{};
    GAMEPLAY::GET_MODEL_DIMENSIONS(
        entity.model_hash,
        &minimum,
        &maximum);
    const float extent_x = maximum.x - minimum.x;
    const float extent_y = maximum.y - minimum.y;
    const float extent_z = maximum.z - minimum.z;
    if (!std::isfinite(extent_x) ||
        !std::isfinite(extent_y) ||
        !std::isfinite(extent_z) ||
        extent_x <= kMinimumModelExtentMeters ||
        extent_y <= kMinimumModelExtentMeters ||
        extent_z <= kMinimumModelExtentMeters) {
        error =
            "GTA returned invalid model dimensions for vehicle entity " +
            std::to_string(entity.stable_id);
        return false;
    }

    target.stable_id = entity.stable_id;
    target.gta_handle = entity.gta_handle;
    target.role = role;

    const float center_x = (minimum.x + maximum.x) * 0.5f;
    const float center_y = (minimum.y + maximum.y) * 0.5f;
    const float center_z = (minimum.z + maximum.z) * 0.5f;
    const std::array<std::array<float, 3>, 15> offsets = {{
        {minimum.x, minimum.y, minimum.z},
        {minimum.x, minimum.y, maximum.z},
        {minimum.x, maximum.y, minimum.z},
        {minimum.x, maximum.y, maximum.z},
        {maximum.x, minimum.y, minimum.z},
        {maximum.x, minimum.y, maximum.z},
        {maximum.x, maximum.y, minimum.z},
        {maximum.x, maximum.y, maximum.z},
        {minimum.x, center_y, center_z},
        {maximum.x, center_y, center_z},
        {center_x, minimum.y, center_z},
        {center_x, maximum.y, center_z},
        {center_x, center_y, minimum.z},
        {center_x, center_y, maximum.z},
        {center_x, center_y, center_z},
    }};

    for (const auto& offset : offsets) {
        const Vector3 world =
            ENTITY::GET_OFFSET_FROM_ENTITY_IN_WORLD_COORDS(
                entity.gta_handle,
                offset[0],
                offset[1],
                offset[2]);
        if (!append_unique_point(
                target.samples,
                to_scenario_vector(world))) {
            error =
                "Vehicle visibility geometry contains an invalid or "
                "duplicate point for entity " +
                std::to_string(entity.stable_id);
            return false;
        }
    }
    return true;
}

bool build_pedestrian_samples(
    const ScenarioEntitySnapshot& entity,
    VisibilityTargetSnapshot& target,
    std::string& error) {
    target.stable_id = entity.stable_id;
    target.gta_handle = entity.gta_handle;
    target.role = VisibilityTargetRole::FleeingPedestrian;
    for (int bone : kPedestrianBones) {
        const Vector3 world = PED::GET_PED_BONE_COORDS(
            entity.gta_handle,
            bone,
            0.0f,
            0.0f,
            0.0f);
        if (!append_unique_point(
                target.samples,
                to_scenario_vector(world))) {
            error =
                "Pedestrian visibility geometry contains an invalid or "
                "duplicate bone point for entity " +
                std::to_string(entity.stable_id);
            return false;
        }
    }
    return true;
}

VisibilityTargetSnapshot build_fire_envelope(
    const ScenarioVector3& event_position) {
    VisibilityTargetSnapshot target;
    target.role = VisibilityTargetRole::FireEnvelope;
    const std::array<float, 3> heights = {
        5.0f,
        15.0f,
        kFireEnvelopeHeightMeters,
    };
    for (float height : heights) {
        VisibilitySampleSnapshot center;
        center.position = {
            event_position.x,
            event_position.y,
            event_position.z + height,
        };
        target.samples.push_back(center);
        for (int index = 0; index < 8; ++index) {
            const float angle =
                2.0f * kPi * static_cast<float>(index) / 8.0f;
            VisibilitySampleSnapshot ring;
            ring.position = {
                event_position.x +
                    std::cos(angle) * kFireEnvelopeRadiusMeters,
                event_position.y +
                    std::sin(angle) * kFireEnvelopeRadiusMeters,
                event_position.z + height,
            };
            target.samples.push_back(ring);
        }
    }
    return target;
}

struct RaycastWork {
    std::size_t target_index = 0;
    std::size_t sample_index = 0;
    int handle = 0;
    bool completed = false;
};

bool evaluate_raycasts(
    const ScenarioVector3& camera_center,
    Entity ignored_player,
    Entity source_vehicle,
    const std::atomic<bool>& cancelled,
    std::vector<VisibilityTargetSnapshot>& targets,
    std::string& error) {
    std::vector<std::pair<std::size_t, std::size_t>> work_items;
    for (std::size_t target_index = 0;
         target_index < targets.size();
         ++target_index) {
        for (std::size_t sample_index = 0;
             sample_index < targets[target_index].samples.size();
             ++sample_index) {
            work_items.emplace_back(target_index, sample_index);
        }
    }

    for (std::size_t offset = 0;
         offset < work_items.size();
         offset += kMaximumOutstandingRaycasts) {
        if (evaluation_interrupted(cancelled)) {
            error = "Visibility query was interrupted";
            return false;
        }
        const std::size_t count = (std::min)(
            kMaximumOutstandingRaycasts,
            work_items.size() - offset);
        std::vector<RaycastWork> batch;
        batch.reserve(count);
        for (std::size_t batch_index = 0;
             batch_index < count;
             ++batch_index) {
            const auto indices = work_items[offset + batch_index];
            const ScenarioVector3& point =
                targets[indices.first]
                    .samples[indices.second]
                    .position;
            RaycastWork work;
            work.target_index = indices.first;
            work.sample_index = indices.second;
            work.handle = WORLDPROBE::_0x7EE9F5D83DD4F90E(
                camera_center.x,
                camera_center.y,
                camera_center.z,
                point.x,
                point.y,
                point.z,
                kRaycastFlags,
                ignored_player,
                kRaycastOption);
            if (work.handle == 0) {
                error =
                    "GTA did not create visibility raycast " +
                    std::to_string(offset + batch_index) +
                    " of " +
                    std::to_string(work_items.size());
                return false;
            }
            batch.push_back(work);
        }

        bool complete = false;
        for (int frame = 0;
             frame < kMaximumRaycastFrames && !complete;
             ++frame) {
            complete = true;
            for (RaycastWork& work : batch) {
                if (work.completed) {
                    continue;
                }
                BOOL hit = FALSE;
                Vector3 hit_position{};
                Vector3 hit_normal{};
                Entity hit_entity = 0;
                const int state = WORLDPROBE::_GET_RAYCAST_RESULT(
                    work.handle,
                    &hit,
                    &hit_position,
                    &hit_normal,
                    &hit_entity);
                if (state == 1) {
                    complete = false;
                    continue;
                }
                if (state != 2) {
                    error = "GTA returned an invalid visibility-ray state";
                    return false;
                }
                VisibilityTargetSnapshot& target =
                    targets[work.target_index];
                VisibilitySampleSnapshot& sample =
                    target.samples[work.sample_index];
                sample.hit_entity =
                    hit == TRUE ? static_cast<std::int32_t>(hit_entity) : 0;
                sample.clear_line_of_sight =
                    hit != TRUE ||
                    hit_entity == target.gta_handle ||
                    (target.role ==
                         VisibilityTargetRole::FireEnvelope &&
                     hit_entity == source_vehicle);
                work.completed = true;
            }
            if (!complete) {
                if (evaluation_interrupted(cancelled)) {
                    error = "Visibility query was interrupted";
                    return false;
                }
                CameraController::instance()
                    .suppress_player_controls_for_frame();
                WAIT(0);
            }
        }
        if (!complete) {
            error =
                "Visibility raycasts did not complete within eight "
                "render frames";
            return false;
        }
        if (offset + count < work_items.size()) {
            if (evaluation_interrupted(cancelled)) {
                error = "Visibility query was interrupted";
                return false;
            }
            CameraController::instance()
                .suppress_player_controls_for_frame();
            WAIT(0);
        }
    }
    return true;
}

bool build_entity_target(
    const ScenarioEntitySnapshot& entity,
    VisibilityTargetSnapshot& target,
    std::string& error) {
    switch (entity.role) {
        case ScenarioEntityRole::FireSourceVehicle:
            return build_vehicle_samples(
                entity,
                VisibilityTargetRole::FireSourceVehicle,
                target,
                error);
        case ScenarioEntityRole::FireTruck:
            return build_vehicle_samples(
                entity,
                VisibilityTargetRole::FireTruck,
                target,
                error);
        case ScenarioEntityRole::FleeingPedestrian:
            return build_pedestrian_samples(
                entity,
                target,
                error);
        case ScenarioEntityRole::FirefighterDriver:
            error =
                "Firefighter drivers are not visibility targets";
            return false;
        default:
            error = "Scenario contains an unknown visibility role";
            return false;
    }
}

struct GeometryRaycastWork {
    bool point = false;
    std::size_t index = 0;
    int handle = 0;
    bool completed = false;
};

bool evaluate_geometry_batch(
    const std::vector<ScenarioVector3>& points,
    const std::vector<GeometrySegment>& segments,
    const std::atomic<bool>& cancelled,
    std::vector<bool>& point_clear,
    std::vector<bool>& segment_clear,
    std::string& error) {
    point_clear.assign(points.size(), false);
    segment_clear.assign(segments.size(), false);
    const Ped player = PLAYER::PLAYER_PED_ID();
    const std::size_t total = points.size() + segments.size();
    for (std::size_t offset = 0;
         offset < total;
         offset += kMaximumOutstandingRaycasts) {
        if (evaluation_interrupted(cancelled)) {
            error = "Camera-geometry batch was interrupted";
            return false;
        }
        const std::size_t count = (std::min)(
            kMaximumOutstandingRaycasts,
            total - offset);
        std::vector<GeometryRaycastWork> batch;
        batch.reserve(count);
        for (std::size_t batch_index = 0;
             batch_index < count;
             ++batch_index) {
            const std::size_t flat_index = offset + batch_index;
            GeometryRaycastWork work;
            ScenarioVector3 start;
            ScenarioVector3 end;
            if (flat_index < points.size()) {
                work.point = true;
                work.index = flat_index;
                start = points[flat_index];
                end = points[flat_index];
                start.z -= 0.01f;
                end.z += 0.01f;
            } else {
                work.index = flat_index - points.size();
                start = segments[work.index].start;
                end = segments[work.index].end;
            }
            work.handle = static_cast<int>(
                WORLDPROBE::_CAST_3D_RAY_POINT_TO_POINT(
                    start.x,
                    start.y,
                    start.z,
                    end.x,
                    end.y,
                    end.z,
                    kProfileClearanceRadiusMeters,
                    1,
                    player,
                    kRaycastOption));
            if (work.handle == 0) {
                error =
                    "GTA did not create camera-geometry shape test " +
                    std::to_string(flat_index) + " of " +
                    std::to_string(total);
                return false;
            }
            batch.push_back(work);
        }

        bool complete = false;
        for (int frame = 0;
             frame < kMaximumRaycastFrames && !complete;
             ++frame) {
            complete = true;
            for (GeometryRaycastWork& work : batch) {
                if (work.completed) {
                    continue;
                }
                BOOL hit = FALSE;
                Vector3 hit_position{};
                Vector3 hit_normal{};
                Entity hit_entity = 0;
                const int state = WORLDPROBE::_GET_RAYCAST_RESULT(
                    work.handle,
                    &hit,
                    &hit_position,
                    &hit_normal,
                    &hit_entity);
                if (state == 1) {
                    complete = false;
                    continue;
                }
                if (state != 2) {
                    error =
                        "GTA returned an invalid camera-geometry "
                        "shape-test state";
                    return false;
                }
                if (work.point) {
                    point_clear[work.index] = hit != TRUE;
                } else {
                    segment_clear[work.index] = hit != TRUE;
                }
                work.completed = true;
            }
            if (!complete) {
                if (evaluation_interrupted(cancelled)) {
                    error = "Camera-geometry batch was interrupted";
                    return false;
                }
                CameraController::instance()
                    .suppress_player_controls_for_frame();
                WAIT(0);
            }
        }
        if (!complete) {
            error =
                "Camera-geometry shape tests did not complete within "
                "eight render frames";
            return false;
        }
        if (offset + count < total) {
            if (evaluation_interrupted(cancelled)) {
                error = "Camera-geometry batch was interrupted";
                return false;
            }
            CameraController::instance()
                .suppress_player_controls_for_frame();
            WAIT(0);
        }
    }
    return true;
}

}  // namespace

VisibilityEvaluator& VisibilityEvaluator::instance() {
    static VisibilityEvaluator evaluator;
    return evaluator;
}

VisibilityOperationStatus VisibilityEvaluator::query(
    std::uint64_t scenario_id,
    std::uint64_t lockstep_session_id,
    const ScenarioVector3& camera_center,
    const std::atomic<bool>& cancelled,
    VisibilitySnapshot& output,
    std::string& error) const {
    if (!finite_vector(camera_center)) {
        error = "Visibility camera center must be finite";
        return VisibilityOperationStatus::InvalidRequest;
    }
    if (!CameraController::instance().is_active()) {
        error = "Visibility query requires an active scripted camera";
        return VisibilityOperationStatus::InvalidRequest;
    }

    LockstepSnapshot clock;
    const LockstepOperationStatus clock_status =
        SimulationClock::instance().snapshot(
            lockstep_session_id,
            clock,
            error);
    if (clock_status != LockstepOperationStatus::Ok) {
        return clock_status ==
                LockstepOperationStatus::SessionMismatch
            ? VisibilityOperationStatus::LockstepSessionMismatch
            : VisibilityOperationStatus::LockstepNotActive;
    }

    ScenarioSnapshot scenario;
    const ScenarioOperationStatus scenario_status =
        ScenarioManager::instance().snapshot(
            scenario_id,
            scenario,
            error);
    if (scenario_status != ScenarioOperationStatus::Ok) {
        return VisibilityOperationStatus::ScenarioNotFound;
    }
    if (scenario.lifecycle != ScenarioLifecycle::Ready &&
        scenario.lifecycle != ScenarioLifecycle::Running) {
        error = "Visibility query requires a READY or RUNNING scenario";
        return VisibilityOperationStatus::ScenarioNotReady;
    }

    output = {};
    output.scenario_id = scenario_id;
    output.lockstep_session_id = lockstep_session_id;
    output.step_index = clock.step_index;
    output.game_timer_ms = clock.game_timer_ms;
    output.frame_count =
        static_cast<std::uint32_t>(GAMEPLAY::GET_FRAME_COUNT());
    output.camera_center = camera_center;

    Entity source_vehicle = 0;
    for (const ScenarioEntitySnapshot& entity : scenario.entities) {
        if (!entity.exists ||
            entity.gta_handle == 0 ||
            !ENTITY::DOES_ENTITY_EXIST(entity.gta_handle)) {
            error =
                "Scenario entity disappeared during visibility query: " +
                std::to_string(entity.stable_id);
            return VisibilityOperationStatus::GeometryInvalid;
        }
        VisibilityTargetSnapshot target;
        bool include = false;
        bool built = false;
        switch (entity.role) {
            case ScenarioEntityRole::FireSourceVehicle:
                source_vehicle = entity.gta_handle;
                include = true;
                built = build_vehicle_samples(
                    entity,
                    VisibilityTargetRole::FireSourceVehicle,
                    target,
                    error);
                break;
            case ScenarioEntityRole::FireTruck:
                include = true;
                built = build_vehicle_samples(
                    entity,
                    VisibilityTargetRole::FireTruck,
                    target,
                    error);
                break;
            case ScenarioEntityRole::FleeingPedestrian:
                include = true;
                built = build_pedestrian_samples(
                    entity,
                    target,
                    error);
                break;
            case ScenarioEntityRole::FirefighterDriver:
                break;
            default:
                error = "Scenario contains an unknown visibility role";
                return VisibilityOperationStatus::GeometryInvalid;
        }
        if (include) {
            if (!built) {
                return VisibilityOperationStatus::GeometryInvalid;
            }
            output.targets.push_back(std::move(target));
        }
    }
    if (source_vehicle == 0) {
        error = "Scenario has no fire-source vehicle";
        return VisibilityOperationStatus::GeometryInvalid;
    }
    output.targets.push_back(
        build_fire_envelope(scenario.event_position));

    const Ped player = PLAYER::PLAYER_PED_ID();
    if (!evaluate_raycasts(
            camera_center,
            player,
            source_vehicle,
            cancelled,
            output.targets,
            error)) {
        return cancelled.load(std::memory_order_acquire) ||
                SimulationClock::instance()
                    .emergency_recovery_requested()
            ? VisibilityOperationStatus::Interrupted
            : VisibilityOperationStatus::RaycastFailed;
    }
    return VisibilityOperationStatus::Ok;
}

VisibilityOperationStatus
VisibilityEvaluator::probe_camera_geometry_batch(
    std::uint64_t lockstep_session_id,
    const std::vector<ScenarioVector3>& points,
    const std::vector<GeometrySegment>& segments,
    const std::atomic<bool>& cancelled,
    GeometryBatchSnapshot& output,
    std::string& error) const {
    const std::size_t total = points.size() + segments.size();
    if (total == 0 || total > kMaximumGeometryBatchItems) {
        error =
            "Camera-geometry batch must contain 1..256 total items";
        return VisibilityOperationStatus::InvalidRequest;
    }
    if (!CameraController::instance().is_active()) {
        error =
            "Camera-geometry batch requires an active scripted camera";
        return VisibilityOperationStatus::InvalidRequest;
    }
    for (const ScenarioVector3& point : points) {
        if (!finite_vector(point)) {
            error =
                "Camera-geometry batch contains a non-finite point";
            return VisibilityOperationStatus::InvalidRequest;
        }
    }
    for (const GeometrySegment& segment : segments) {
        if (!finite_vector(segment.start) ||
            !finite_vector(segment.end)) {
            error =
                "Camera-geometry batch contains a non-finite segment";
            return VisibilityOperationStatus::InvalidRequest;
        }
    }

    LockstepSnapshot before;
    const LockstepOperationStatus before_status =
        SimulationClock::instance().snapshot(
            lockstep_session_id,
            before,
            error);
    if (before_status != LockstepOperationStatus::Ok) {
        return before_status ==
                LockstepOperationStatus::SessionMismatch
            ? VisibilityOperationStatus::LockstepSessionMismatch
            : VisibilityOperationStatus::LockstepNotActive;
    }

    output = {};
    output.lockstep_session_id = lockstep_session_id;
    output.step_index = before.step_index;
    output.game_timer_ms = before.game_timer_ms;
    if (!evaluate_geometry_batch(
            points,
            segments,
            cancelled,
            output.point_clear,
            output.segment_clear,
            error)) {
        return cancelled.load(std::memory_order_acquire) ||
                SimulationClock::instance()
                    .emergency_recovery_requested()
            ? VisibilityOperationStatus::Interrupted
            : VisibilityOperationStatus::RaycastFailed;
    }

    LockstepSnapshot after;
    const LockstepOperationStatus after_status =
        SimulationClock::instance().snapshot(
            lockstep_session_id,
            after,
            error);
    if (after_status != LockstepOperationStatus::Ok ||
        after.step_index != before.step_index ||
        after.game_timer_ms != before.game_timer_ms) {
        if (after_status == LockstepOperationStatus::Ok) {
            error =
                "Lockstep instant changed during camera-geometry batch";
        }
        return VisibilityOperationStatus::GeometryInvalid;
    }
    output.frame_count = after.frame_count;
    return VisibilityOperationStatus::Ok;
}

VisibilityOperationStatus
VisibilityEvaluator::query_target_batch(
    std::uint64_t scenario_id,
    std::uint64_t lockstep_session_id,
    const std::vector<TargetVisibilityCase>& cases,
    const std::atomic<bool>& cancelled,
    TargetVisibilityBatchSnapshot& output,
    std::string& error) const {
    if (cases.empty() ||
        cases.size() > kMaximumTargetVisibilityCases) {
        error =
            "Target-visibility batch must contain 1..64 cases";
        return VisibilityOperationStatus::InvalidRequest;
    }
    if (!CameraController::instance().is_active()) {
        error =
            "Target-visibility batch requires an active scripted camera";
        return VisibilityOperationStatus::InvalidRequest;
    }
    for (const TargetVisibilityCase& item : cases) {
        if (item.stable_id == 0 ||
            !finite_vector(item.camera_center)) {
            error =
                "Target-visibility batch contains an invalid case";
            return VisibilityOperationStatus::InvalidRequest;
        }
    }

    LockstepSnapshot before;
    const LockstepOperationStatus clock_status =
        SimulationClock::instance().snapshot(
            lockstep_session_id,
            before,
            error);
    if (clock_status != LockstepOperationStatus::Ok) {
        return clock_status ==
                LockstepOperationStatus::SessionMismatch
            ? VisibilityOperationStatus::LockstepSessionMismatch
            : VisibilityOperationStatus::LockstepNotActive;
    }

    ScenarioSnapshot scenario;
    const ScenarioOperationStatus scenario_status =
        ScenarioManager::instance().snapshot(
            scenario_id,
            scenario,
            error);
    if (scenario_status != ScenarioOperationStatus::Ok) {
        return VisibilityOperationStatus::ScenarioNotFound;
    }
    if (scenario.lifecycle != ScenarioLifecycle::Ready &&
        scenario.lifecycle != ScenarioLifecycle::Running) {
        error =
            "Target-visibility batch requires a READY or RUNNING scenario";
        return VisibilityOperationStatus::ScenarioNotReady;
    }

    Entity source_vehicle = 0;
    for (const ScenarioEntitySnapshot& entity : scenario.entities) {
        if (entity.role == ScenarioEntityRole::FireSourceVehicle &&
            entity.exists &&
            entity.gta_handle != 0 &&
            ENTITY::DOES_ENTITY_EXIST(entity.gta_handle)) {
            source_vehicle = entity.gta_handle;
            break;
        }
    }
    if (source_vehicle == 0) {
        error = "Scenario has no live fire-source vehicle";
        return VisibilityOperationStatus::GeometryInvalid;
    }

    output = {};
    output.scenario_id = scenario_id;
    output.lockstep_session_id = lockstep_session_id;
    output.step_index = before.step_index;
    output.game_timer_ms = before.game_timer_ms;
    output.cases.reserve(cases.size());
    for (const TargetVisibilityCase& item : cases) {
        if (evaluation_interrupted(cancelled)) {
            error = "Target-visibility batch was interrupted";
            return VisibilityOperationStatus::Interrupted;
        }
        const auto found = std::find_if(
            scenario.entities.begin(),
            scenario.entities.end(),
            [&](const ScenarioEntitySnapshot& entity) {
                return entity.stable_id == item.stable_id;
            });
        if (found == scenario.entities.end()) {
            error =
                "Target-visibility stable ID was not found: " +
                std::to_string(item.stable_id);
            return VisibilityOperationStatus::GeometryInvalid;
        }
        if (!found->exists ||
            found->gta_handle == 0 ||
            !ENTITY::DOES_ENTITY_EXIST(found->gta_handle)) {
            error =
                "Target-visibility entity is not live: " +
                std::to_string(item.stable_id);
            return VisibilityOperationStatus::GeometryInvalid;
        }

        TargetVisibilityCaseSnapshot result;
        result.stable_id = item.stable_id;
        result.camera_center = item.camera_center;
        if (!build_entity_target(*found, result.target, error)) {
            return VisibilityOperationStatus::GeometryInvalid;
        }
        std::vector<VisibilityTargetSnapshot> targets;
        targets.push_back(result.target);
        if (!evaluate_raycasts(
                item.camera_center,
                PLAYER::PLAYER_PED_ID(),
                source_vehicle,
                cancelled,
                targets,
                error)) {
            return cancelled.load(std::memory_order_acquire) ||
                    SimulationClock::instance()
                        .emergency_recovery_requested()
                ? VisibilityOperationStatus::Interrupted
                : VisibilityOperationStatus::RaycastFailed;
        }
        result.target = std::move(targets.front());
        output.cases.push_back(std::move(result));
    }

    LockstepSnapshot after;
    const LockstepOperationStatus after_status =
        SimulationClock::instance().snapshot(
            lockstep_session_id,
            after,
            error);
    if (after_status != LockstepOperationStatus::Ok ||
        after.step_index != before.step_index ||
        after.game_timer_ms != before.game_timer_ms) {
        if (after_status == LockstepOperationStatus::Ok) {
            error =
                "Lockstep instant changed during target-visibility batch";
        }
        return VisibilityOperationStatus::GeometryInvalid;
    }
    output.frame_count = after.frame_count;
    return VisibilityOperationStatus::Ok;
}

CameraStartProbeStatus VisibilityEvaluator::probe_camera_start(
    float x,
    float y,
    float altitude_agl,
    const std::atomic<bool>& cancelled,
    CameraStartProbe& output,
    std::string& error) const {
    if (!std::isfinite(x) ||
        !std::isfinite(y) ||
        !std::isfinite(altitude_agl) ||
        altitude_agl <= 0.0f) {
        error =
            "Camera-start X/Y must be finite and altitude AGL must be "
            "positive";
        return CameraStartProbeStatus::InvalidRequest;
    }
    if (cancelled.load(std::memory_order_acquire)) {
        error = "Camera-start probe was cancelled";
        return CameraStartProbeStatus::Interrupted;
    }

    float ground_z = 0.0f;
    if (GAMEPLAY::GET_GROUND_Z_FOR_3D_COORD(
            x,
            y,
            1000.0f,
            &ground_z,
            FALSE) != TRUE ||
        !std::isfinite(ground_z)) {
        error =
            "GTA could not resolve loaded ground at the requested X/Y";
        return CameraStartProbeStatus::GroundNotFound;
    }
    const float z = ground_z + altitude_agl;
    const Ped player = PLAYER::PLAYER_PED_ID();
    const int handle = static_cast<int>(
        WORLDPROBE::_CAST_3D_RAY_POINT_TO_POINT(
            x,
            y,
            z - 0.25f,
            x,
            y,
            z + 0.25f,
            kCameraClearanceRadiusMeters,
            1,
            player,
            kRaycastOption));
    if (handle == 0) {
        error = "GTA did not create the camera-clearance shape test";
        return CameraStartProbeStatus::RaycastFailed;
    }

    for (int frame = 0; frame < kMaximumRaycastFrames; ++frame) {
        BOOL hit = FALSE;
        Vector3 hit_position{};
        Vector3 hit_normal{};
        Entity hit_entity = 0;
        const int state = WORLDPROBE::_GET_RAYCAST_RESULT(
            handle,
            &hit,
            &hit_position,
            &hit_normal,
            &hit_entity);
        if (state == 2) {
            if (hit == TRUE) {
                error =
                    "Camera-start clearance sphere intersects world "
                    "geometry";
                return CameraStartProbeStatus::SpaceBlocked;
            }
            output.position = {x, y, z};
            output.ground_z = ground_z;
            return CameraStartProbeStatus::Ok;
        }
        if (state != 1) {
            error =
                "GTA returned an invalid camera-clearance ray state";
            return CameraStartProbeStatus::RaycastFailed;
        }
        if (cancelled.load(std::memory_order_acquire)) {
            error = "Camera-start probe was cancelled";
            return CameraStartProbeStatus::Interrupted;
        }
        CameraController::instance()
            .suppress_player_controls_for_frame();
        WAIT(0);
    }
    error =
        "Camera-start clearance test did not complete within eight "
        "render frames";
    return CameraStartProbeStatus::RaycastFailed;
}
