#pragma once

#include "scenario_types.h"
#include "types.h"

#include <cstdint>
#include <vector>

class EntityRegistry {
public:
    struct Entry {
        std::uint64_t stable_id = 0;
        Entity handle = 0;
        Hash model_hash = 0;
        ScenarioEntityKind kind = ScenarioEntityKind::Vehicle;
        ScenarioEntityRole role =
            ScenarioEntityRole::FireSourceVehicle;
        std::uint64_t event_id = 0;
        ScenarioTaskState task_state = ScenarioTaskState::None;
        std::uint32_t spawn_game_timer_ms = 0;
        std::uint32_t task_start_game_timer_ms = 0;
        std::uint32_t response_start_game_timer_ms = 0;
        ScenarioVector3 task_target;
        float last_progress_distance = 0.0f;
        std::uint32_t last_activity_game_timer_ms = 0;
        bool kinematics_frozen = false;
        bool frozen_exists = false;
        ScenarioVector3 frozen_position;
        ScenarioVector3 frozen_velocity;
        float frozen_speed = 0.0f;
        float frozen_heading = 0.0f;
    };

    std::uint64_t add(
        Entity handle,
        Hash model_hash,
        ScenarioEntityKind kind,
        ScenarioEntityRole role,
        std::uint64_t event_id,
        std::uint32_t spawn_game_timer_ms);
    Entry* find(std::uint64_t stable_id);
    const Entry* find(std::uint64_t stable_id) const;
    bool contains_handle(Entity handle) const;
    void start_task(
        std::uint64_t stable_id,
        const ScenarioVector3& target,
        std::uint32_t game_timer_ms,
        float initial_distance);
    void set_task_state(
        std::uint64_t stable_id,
        ScenarioTaskState state);
    void update_tasks(
        const ScenarioVector3& event_position,
        std::uint32_t game_timer_ms);
    void freeze_kinematics();
    void restore_frozen_velocities();
    void unfreeze_kinematics();
    std::vector<ScenarioEntitySnapshot> snapshots() const;
    void delete_all();
    void clear();
    bool empty() const;

private:
    std::vector<Entry> entries_;
    std::uint64_t next_stable_id_ = 1;
};
