#include "entity_registry.h"

#include "natives.h"

#include <algorithm>
#include <cmath>

namespace {

constexpr float kResponseSpeedThreshold = 0.5f;
constexpr float kProgressThresholdMeters = 0.5f;
constexpr float kFireTruckStopDistanceMeters = 10.0f;
constexpr float kPedestrianSuccessDistanceMeters = 60.0f;
constexpr std::uint32_t kStallTimeoutMilliseconds = 15000;

float distance_between(
    const ScenarioVector3& left,
    const ScenarioVector3& right) {
    const float dx = left.x - right.x;
    const float dy = left.y - right.y;
    const float dz = left.z - right.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

ScenarioVector3 entity_position(Entity entity) {
    const Vector3 value = ENTITY::GET_ENTITY_COORDS(entity, TRUE);
    return {value.x, value.y, value.z};
}

ScenarioVector3 entity_velocity(Entity entity) {
    const Vector3 value = ENTITY::GET_ENTITY_VELOCITY(entity);
    return {value.x, value.y, value.z};
}

}  // namespace

std::uint64_t EntityRegistry::add(
    Entity handle,
    Hash model_hash,
    ScenarioEntityKind kind,
    ScenarioEntityRole role,
    std::uint64_t event_id,
    std::uint32_t spawn_game_timer_ms) {
    Entry entry;
    entry.stable_id = next_stable_id_++;
    entry.handle = handle;
    entry.model_hash = model_hash;
    entry.kind = kind;
    entry.role = role;
    entry.event_id = event_id;
    entry.spawn_game_timer_ms = spawn_game_timer_ms;
    entries_.push_back(entry);
    return entry.stable_id;
}

EntityRegistry::Entry* EntityRegistry::find(
    std::uint64_t stable_id) {
    const auto iterator = std::find_if(
        entries_.begin(),
        entries_.end(),
        [stable_id](const Entry& entry) {
            return entry.stable_id == stable_id;
        });
    return iterator == entries_.end() ? nullptr : &*iterator;
}

const EntityRegistry::Entry* EntityRegistry::find(
    std::uint64_t stable_id) const {
    const auto iterator = std::find_if(
        entries_.begin(),
        entries_.end(),
        [stable_id](const Entry& entry) {
            return entry.stable_id == stable_id;
        });
    return iterator == entries_.end() ? nullptr : &*iterator;
}

bool EntityRegistry::contains_handle(Entity handle) const {
    return std::any_of(
        entries_.begin(),
        entries_.end(),
        [handle](const Entry& entry) {
            return entry.handle == handle;
        });
}

void EntityRegistry::start_task(
    std::uint64_t stable_id,
    const ScenarioVector3& target,
    std::uint32_t game_timer_ms,
    float initial_distance) {
    Entry* entry = find(stable_id);
    if (entry == nullptr) {
        return;
    }
    entry->task_state = ScenarioTaskState::Active;
    entry->task_target = target;
    entry->task_start_game_timer_ms = game_timer_ms;
    entry->last_activity_game_timer_ms = game_timer_ms;
    entry->last_progress_distance = initial_distance;
}

void EntityRegistry::set_task_state(
    std::uint64_t stable_id,
    ScenarioTaskState state) {
    Entry* entry = find(stable_id);
    if (entry != nullptr) {
        entry->task_state = state;
    }
}

void EntityRegistry::update_tasks(
    const ScenarioVector3& event_position,
    std::uint32_t game_timer_ms) {
    for (Entry& entry : entries_) {
        if (entry.task_state != ScenarioTaskState::Active) {
            continue;
        }
        if (entry.handle == 0 ||
            !ENTITY::DOES_ENTITY_EXIST(entry.handle)) {
            entry.task_state = ScenarioTaskState::Lost;
            continue;
        }

        const float speed = ENTITY::GET_ENTITY_SPEED(entry.handle);
        const bool moving =
            std::isfinite(speed) &&
            speed >= kResponseSpeedThreshold;
        if (entry.response_start_game_timer_ms == 0 &&
            moving) {
            entry.response_start_game_timer_ms = game_timer_ms;
        }

        const float distance =
            distance_between(entity_position(entry.handle), event_position);
        bool progressed = false;
        if (entry.role == ScenarioEntityRole::FireTruck) {
            if (distance <= kFireTruckStopDistanceMeters) {
                entry.task_state = ScenarioTaskState::Succeeded;
                continue;
            }
            progressed =
                distance <=
                entry.last_progress_distance - kProgressThresholdMeters;
        } else if (
            entry.role ==
            ScenarioEntityRole::FleeingPedestrian) {
            if (distance >= kPedestrianSuccessDistanceMeters) {
                entry.task_state = ScenarioTaskState::Succeeded;
                continue;
            }
            progressed =
                distance >=
                entry.last_progress_distance + kProgressThresholdMeters;
        } else {
            continue;
        }

        if (progressed) {
            entry.last_progress_distance = distance;
        }
        if (moving || progressed) {
            entry.last_activity_game_timer_ms = game_timer_ms;
        } else if (
            game_timer_ms - entry.last_activity_game_timer_ms >=
            kStallTimeoutMilliseconds) {
            entry.task_state = ScenarioTaskState::Failed;
        }
    }
}

std::vector<ScenarioEntitySnapshot>
EntityRegistry::snapshots() const {
    std::vector<ScenarioEntitySnapshot> output;
    output.reserve(entries_.size());
    for (const Entry& entry : entries_) {
        ScenarioEntitySnapshot snapshot;
        snapshot.stable_id = entry.stable_id;
        snapshot.gta_handle =
            static_cast<std::int32_t>(entry.handle);
        snapshot.model_hash =
            static_cast<std::uint32_t>(entry.model_hash);
        snapshot.kind = entry.kind;
        snapshot.role = entry.role;
        snapshot.event_id = entry.event_id;
        snapshot.task_state = entry.task_state;
        snapshot.spawn_game_timer_ms =
            entry.spawn_game_timer_ms;
        snapshot.task_start_game_timer_ms =
            entry.task_start_game_timer_ms;
        snapshot.response_start_game_timer_ms =
            entry.response_start_game_timer_ms;
        snapshot.task_target = entry.task_target;
        snapshot.exists =
            entry.handle != 0 &&
            ENTITY::DOES_ENTITY_EXIST(entry.handle);
        if (snapshot.exists) {
            snapshot.position = entity_position(entry.handle);
            snapshot.velocity = entity_velocity(entry.handle);
            snapshot.speed = ENTITY::GET_ENTITY_SPEED(entry.handle);
            snapshot.heading =
                ENTITY::GET_ENTITY_HEADING(entry.handle);
        }
        output.push_back(snapshot);
    }
    return output;
}

void EntityRegistry::delete_all() {
    for (Entry& entry : entries_) {
        if (entry.kind != ScenarioEntityKind::Pedestrian ||
            entry.handle == 0 ||
            !ENTITY::DOES_ENTITY_EXIST(entry.handle)) {
            continue;
        }
        AI::CLEAR_PED_TASKS_IMMEDIATELY(entry.handle);
        ENTITY::SET_ENTITY_AS_MISSION_ENTITY(
            entry.handle,
            TRUE,
            TRUE);
        Entity handle = entry.handle;
        ENTITY::DELETE_ENTITY(&handle);
        entry.handle = 0;
    }
    for (Entry& entry : entries_) {
        if (entry.kind != ScenarioEntityKind::Vehicle ||
            entry.handle == 0 ||
            !ENTITY::DOES_ENTITY_EXIST(entry.handle)) {
            continue;
        }
        ENTITY::SET_ENTITY_AS_MISSION_ENTITY(
            entry.handle,
            TRUE,
            TRUE);
        Entity handle = entry.handle;
        ENTITY::DELETE_ENTITY(&handle);
        entry.handle = 0;
    }
}

void EntityRegistry::clear() {
    entries_.clear();
    next_stable_id_ = 1;
}

bool EntityRegistry::empty() const {
    return entries_.empty();
}
