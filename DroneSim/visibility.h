#pragma once

#include "scenario_types.h"

#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

enum class VisibilityTargetRole : std::uint32_t {
    FireSourceVehicle = 1,
    FireEnvelope = 2,
    FireTruck = 3,
    FleeingPedestrian = 4,
};

enum class VisibilityOperationStatus {
    Ok,
    InvalidRequest,
    ScenarioNotFound,
    ScenarioNotReady,
    LockstepNotActive,
    LockstepSessionMismatch,
    GeometryInvalid,
    RaycastFailed,
    Interrupted,
};

enum class CameraStartProbeStatus {
    Ok,
    InvalidRequest,
    GroundNotFound,
    SpaceBlocked,
    RaycastFailed,
    Interrupted,
};

struct CameraStartProbe {
    ScenarioVector3 position;
    float ground_z = 0.0f;
};

struct VisibilitySampleSnapshot {
    ScenarioVector3 position;
    bool clear_line_of_sight = false;
    std::int32_t hit_entity = 0;
};

struct VisibilityTargetSnapshot {
    std::uint64_t stable_id = 0;
    std::int32_t gta_handle = 0;
    VisibilityTargetRole role =
        VisibilityTargetRole::FireSourceVehicle;
    std::vector<VisibilitySampleSnapshot> samples;
};

struct VisibilitySnapshot {
    std::uint64_t scenario_id = 0;
    std::uint64_t lockstep_session_id = 0;
    std::uint64_t step_index = 0;
    std::uint32_t game_timer_ms = 0;
    std::uint32_t frame_count = 0;
    ScenarioVector3 camera_center;
    std::vector<VisibilityTargetSnapshot> targets;
};

struct GeometrySegment {
    ScenarioVector3 start;
    ScenarioVector3 end;
};

struct GeometryBatchSnapshot {
    std::uint64_t lockstep_session_id = 0;
    std::uint64_t step_index = 0;
    std::uint32_t game_timer_ms = 0;
    std::uint32_t frame_count = 0;
    std::vector<bool> point_clear;
    std::vector<bool> segment_clear;
};

struct TargetVisibilityCase {
    std::uint64_t stable_id = 0;
    ScenarioVector3 camera_center;
};

struct TargetVisibilityCaseSnapshot {
    std::uint64_t stable_id = 0;
    ScenarioVector3 camera_center;
    VisibilityTargetSnapshot target;
};

struct TargetVisibilityBatchSnapshot {
    std::uint64_t scenario_id = 0;
    std::uint64_t lockstep_session_id = 0;
    std::uint64_t step_index = 0;
    std::uint32_t game_timer_ms = 0;
    std::uint32_t frame_count = 0;
    std::vector<TargetVisibilityCaseSnapshot> cases;
};

class VisibilityEvaluator {
public:
    static VisibilityEvaluator& instance();

    VisibilityOperationStatus query(
        std::uint64_t scenario_id,
        std::uint64_t lockstep_session_id,
        const ScenarioVector3& camera_center,
        const std::atomic<bool>& cancelled,
        VisibilitySnapshot& output,
        std::string& error) const;
    VisibilityOperationStatus probe_camera_geometry_batch(
        std::uint64_t lockstep_session_id,
        const std::vector<ScenarioVector3>& points,
        const std::vector<GeometrySegment>& segments,
        const std::atomic<bool>& cancelled,
        GeometryBatchSnapshot& output,
        std::string& error) const;
    VisibilityOperationStatus query_target_batch(
        std::uint64_t scenario_id,
        std::uint64_t lockstep_session_id,
        const std::vector<TargetVisibilityCase>& cases,
        const std::atomic<bool>& cancelled,
        TargetVisibilityBatchSnapshot& output,
        std::string& error) const;
    CameraStartProbeStatus probe_camera_start(
        float x,
        float y,
        float altitude_agl,
        const std::atomic<bool>& cancelled,
        CameraStartProbe& output,
        std::string& error) const;

private:
    VisibilityEvaluator() = default;
};
