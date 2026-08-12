#include "fire_scenario.h"

#include "fire_visual_config.h"
#include "logging.h"
#include "main.h"
#include "natives.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <set>
#include <utility>

namespace {

constexpr float kMaximumAnchorSnapDistanceMeters = 30.0f;
constexpr float kMaximumPlayerDistanceMeters = 200.0f;
constexpr float kAmbientClearRadiusMeters = 120.0f;
constexpr std::uint32_t kAmbientMaintenanceIntervalMilliseconds = 100;
constexpr float kFireTruckMinimumDistanceMeters = 80.0f;
constexpr float kFireTruckMaximumDistanceMeters = 120.0f;
constexpr float kFireTruckMaximumRoadDistanceMeters = 350.0f;
constexpr float kFireTruckMinimumHeadingCosine = 0.0f;
constexpr std::array<std::array<float, 2>, 4>
    kPedestrianDistanceBandsMeters = {{
        {{8.0f, 20.0f}},
        {{20.0f, 35.0f}},
        {{35.0f, 50.0f}},
        {{50.0f, 65.0f}},
    }};
constexpr float kPedestrianMaximumVerticalOffsetMeters = 12.0f;
constexpr float kFireTruckSpeedMetersPerSecond = 12.0f;
constexpr float kFireTruckStopRangeMeters = 10.0f;
constexpr float kPedestrianFleeDistanceMeters = 120.0f;
constexpr int kDrivingStyle = 786603;
constexpr std::size_t kMaximumPlacementAttempts = 2048;
constexpr std::size_t kPlacementAttemptsPerFrame = 16;
constexpr std::size_t kPedestrianCandidateMultiplier = 3;
constexpr std::size_t kMaximumWorldEntities = 2048;
constexpr std::uint32_t kFireActivationTimeoutMilliseconds = 3000;
constexpr auto kModelLoadTimeout = std::chrono::seconds(10);
constexpr float kPi = 3.14159265358979323846f;
constexpr std::uint64_t kFiretruckRandomStreamTag =
    0x4649524554525543ULL;  // "FIRETRUC"
constexpr std::uint64_t kPedestrianPositionRandomStreamTag =
    0x5045445F504F5349ULL;  // "PED_POSI"
constexpr std::uint64_t kPedestrianModelRandomStreamTag =
    0x5045445F4D4F444CULL;  // "PED_MODL"
constexpr std::uint64_t kPedestrianActivationRandomStreamTag =
    0x5045445F41435449ULL;  // "PED_ACTI"
char kFirePtfxEffect[] = "ent_ray_ch2_farm_fire_dble";

const std::array<const char*, 4> kCivilianModels = {
    "a_m_y_business_01",
    "a_f_y_business_01",
    "a_m_y_hipster_01",
    "a_f_y_hipster_01",
};

std::uint64_t derive_random_stream_seed(
    std::uint64_t master_seed,
    std::uint64_t stream_tag) {
    // SplitMix64 finalization gives each named subsystem a stable random
    // stream without depending on implementation-defined std::hash values.
    std::uint64_t value =
        (master_seed ^ stream_tag) + 0x9E3779B97F4A7C15ULL;
    value =
        (value ^ (value >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    value =
        (value ^ (value >> 27U)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31U);
}

bool finite_vector(const ScenarioVector3& value) {
    return std::isfinite(value.x) &&
        std::isfinite(value.y) &&
        std::isfinite(value.z);
}

float distance_squared(
    const ScenarioVector3& left,
    const ScenarioVector3& right) {
    const float dx = left.x - right.x;
    const float dy = left.y - right.y;
    const float dz = left.z - right.z;
    return dx * dx + dy * dy + dz * dz;
}

float distance_between(
    const ScenarioVector3& left,
    const ScenarioVector3& right) {
    return std::sqrt(distance_squared(left, right));
}

float horizontal_distance_between(
    const ScenarioVector3& left,
    const ScenarioVector3& right) {
    const float dx = left.x - right.x;
    const float dy = left.y - right.y;
    return std::sqrt(dx * dx + dy * dy);
}

float event_angle(
    const ScenarioVector3& event_position,
    const ScenarioVector3& position) {
    return std::atan2(
        position.y - event_position.y,
        position.x - event_position.x);
}

float wrapped_angle_distance(float left, float right) {
    float difference = std::fmod(
        std::abs(left - right),
        2.0f * kPi);
    if (difference > kPi) {
        difference = 2.0f * kPi - difference;
    }
    return difference;
}

std::uint32_t activation_offset_ms(
    float event_distance,
    std::mt19937_64& random) {
    std::uniform_real_distribution<float> jitter(-0.5f, 0.5f);
    const float seconds = std::clamp(
        (std::max)(
            0.0f,
            (event_distance - 20.0f) / 10.0f +
                jitter(random)),
        0.0f,
        12.0f);
    return static_cast<std::uint32_t>(
        std::lround(seconds * 4.0f)) * 250U;
}

ScenarioVector3 to_scenario_vector(const Vector3& value) {
    return {value.x, value.y, value.z};
}

float heading_away_from(
    const ScenarioVector3& origin,
    const ScenarioVector3& position) {
    const float dx = position.x - origin.x;
    const float dy = position.y - origin.y;
    return std::atan2(-dx, dy) * 180.0f / kPi;
}

float heading_toward_cosine(
    float heading_degrees,
    const ScenarioVector3& position,
    const ScenarioVector3& target) {
    const float dx = target.x - position.x;
    const float dy = target.y - position.y;
    const float length = std::sqrt(dx * dx + dy * dy);
    if (length <= 1.0e-4f) {
        return 1.0f;
    }
    const float heading_radians =
        heading_degrees * kPi / 180.0f;
    const float forward_x = -std::sin(heading_radians);
    const float forward_y = std::cos(heading_radians);
    return
        forward_x * (dx / length) +
        forward_y * (dy / length);
}

}  // namespace

ScenarioOperationStatus FireScenario::prepare(
    std::uint64_t scenario_id,
    const FireScenarioConfig& config,
    std::string& error) {
    if (lifecycle_ != ScenarioLifecycle::Empty) {
        error = "A fire scenario is already active";
        return ScenarioOperationStatus::AlreadyActive;
    }
    if (scenario_id == 0) {
        error = "scenario_id must be non-zero";
        return ScenarioOperationStatus::InvalidConfig;
    }
    if (!finite_vector(config.anchor)) {
        error = "Fire scenario anchor must contain finite values";
        return ScenarioOperationStatus::InvalidConfig;
    }
    if (config.firetruck_count > 4) {
        error = "firetruck_count must be in [0, 4]";
        return ScenarioOperationStatus::InvalidConfig;
    }
    if (config.pedestrian_count > 32) {
        error = "pedestrian_count must be in [0, 32]";
        return ScenarioOperationStatus::InvalidConfig;
    }

    config_ = config;
    scenario_id_ = scenario_id;
    event_id_ = (scenario_id << 8U) | 1U;
    firetruck_random_.seed(derive_random_stream_seed(
        config.seed,
        kFiretruckRandomStreamTag));
    pedestrian_position_random_.seed(derive_random_stream_seed(
        config.seed,
        kPedestrianPositionRandomStreamTag));
    pedestrian_model_random_.seed(derive_random_stream_seed(
        config.seed,
        kPedestrianModelRandomStreamTag));
    pedestrian_activation_random_.seed(derive_random_stream_seed(
        config.seed,
        kPedestrianActivationRandomStreamTag));
    failure_.clear();
    removed_pedestrians_ = 0;
    removed_vehicles_ = 0;
    start_game_timer_ms_ = 0;
    start_frame_count_ = 0;

    const ScenarioOperationStatus validation_status =
        validate_area(error);
    if (validation_status != ScenarioOperationStatus::Ok) {
        reset();
        return validation_status;
    }
    if (config.blueprint_id != 0) {
        if (!reuse_blueprint(error)) {
            reset();
            return ScenarioOperationStatus::PrepareFailed;
        }
    } else {
        const ScenarioOperationStatus resolve_status =
            resolve_event(error);
        if (resolve_status != ScenarioOperationStatus::Ok) {
            reset();
            return resolve_status;
        }
        blueprint_id_ = scenario_id_;
        building_blueprint_ = true;
        firetruck_spawns_.clear();
        pedestrian_spawns_.clear();
        for (auto& candidates : pedestrian_candidate_spawns_) {
            candidates.clear();
        }
    }
    placement_attempts_ = 0;
    pedestrian_query_failures_ = 0;
    pedestrian_bounds_rejections_ = 0;
    pedestrian_duplicate_rejections_ = 0;

    lifecycle_ = ScenarioLifecycle::Preparing;
    prepare_stage_ = PrepareStage::CleanAmbient;
    LOGI(
        "scenario",
        "Preparing fire scenario " + std::to_string(scenario_id_) +
            " from blueprint " + std::to_string(blueprint_id_));
    return ScenarioOperationStatus::Ok;
}

ScenarioOperationStatus FireScenario::validate_area(
    std::string& error) {
    const Ped player = PLAYER::PLAYER_PED_ID();
    if (player == 0 || !ENTITY::DOES_ENTITY_EXIST(player)) {
        error = "Player entity is unavailable";
        return ScenarioOperationStatus::AreaNotReady;
    }
    const ScenarioVector3 player_position =
        to_scenario_vector(ENTITY::GET_ENTITY_COORDS(player, TRUE));
    if (distance_between(player_position, config_.anchor) >
        kMaximumPlayerDistanceMeters) {
        error =
            "Player must be within 200 meters of the scenario anchor "
            "before Prepare";
        return ScenarioOperationStatus::AreaNotReady;
    }
    if (!ENTITY::HAS_COLLISION_LOADED_AROUND_ENTITY(player)) {
        error =
            "Collision is not loaded around the player; preload the "
            "scenario area first";
        return ScenarioOperationStatus::AreaNotReady;
    }
    float ground_z = 0.0f;
    if (!GAMEPLAY::GET_GROUND_Z_FOR_3D_COORD(
            config_.anchor.x,
            config_.anchor.y,
            config_.anchor.z + 100.0f,
            &ground_z,
            FALSE) ||
        !std::isfinite(ground_z)) {
        error =
            "Collision around the scenario anchor is not ready";
        return ScenarioOperationStatus::AreaNotReady;
    }

    return ScenarioOperationStatus::Ok;
}

ScenarioOperationStatus FireScenario::resolve_event(
    std::string& error) {
    Vector3 event_node{};
    float event_heading = 0.0f;
    if (!PATHFIND::GET_CLOSEST_VEHICLE_NODE_WITH_HEADING(
            config_.anchor.x,
            config_.anchor.y,
            config_.anchor.z,
            &event_node,
            &event_heading,
            1,
            3.0f,
            0)) {
        error = "No vehicle node was found near the requested anchor";
        return ScenarioOperationStatus::PrepareFailed;
    }
    event_position_ = to_scenario_vector(event_node);
    event_heading_ = event_heading;
    if (horizontal_distance_between(event_position_, config_.anchor) >
        kMaximumAnchorSnapDistanceMeters) {
        error =
            "The closest vehicle node is more than 30 horizontal meters "
            "from the requested anchor";
        return ScenarioOperationStatus::PrepareFailed;
    }

    return ScenarioOperationStatus::Ok;
}

bool FireScenario::reuse_blueprint(std::string& error) {
    if (!cached_blueprint_.valid ||
        cached_blueprint_.id != config_.blueprint_id) {
        error =
            "Requested blueprint_id is not present in the single-slot "
            "blueprint cache";
        return false;
    }
    if (cached_blueprint_.seed != config_.seed) {
        error = "Requested blueprint seed does not match the cached blueprint";
        return false;
    }
    if (distance_between(
            cached_blueprint_.requested_anchor,
            config_.anchor) > 1.0e-3f) {
        error =
            "Requested blueprint anchor does not match the cached blueprint";
        return false;
    }
    if (config_.firetruck_count >
            cached_blueprint_.firetruck_spawns.size() ||
        config_.pedestrian_count >
            cached_blueprint_.pedestrian_spawns.size()) {
        error =
            "Requested actor counts exceed cached blueprint capacity; "
            "create a new superset blueprint first";
        return false;
    }

    blueprint_id_ = cached_blueprint_.id;
    building_blueprint_ = false;
    event_position_ = cached_blueprint_.event_position;
    event_heading_ = cached_blueprint_.event_heading;
    firetruck_spawns_.assign(
        cached_blueprint_.firetruck_spawns.begin(),
        cached_blueprint_.firetruck_spawns.begin() +
            config_.firetruck_count);
    pedestrian_spawns_.assign(
        cached_blueprint_.pedestrian_spawns.begin(),
        cached_blueprint_.pedestrian_spawns.begin() +
            config_.pedestrian_count);
    return true;
}

void FireScenario::commit_blueprint() {
    cached_blueprint_.valid = true;
    cached_blueprint_.id = blueprint_id_;
    cached_blueprint_.seed = config_.seed;
    cached_blueprint_.requested_anchor = config_.anchor;
    cached_blueprint_.event_position = event_position_;
    cached_blueprint_.event_heading = event_heading_;
    cached_blueprint_.firetruck_spawns = firetruck_spawns_;
    cached_blueprint_.pedestrian_spawns = pedestrian_spawns_;
    building_blueprint_ = false;
    LOGI(
        "scenario",
        "Committed immutable blueprint " +
            std::to_string(blueprint_id_) + " with capacity " +
            std::to_string(firetruck_spawns_.size()) +
            " firetrucks and " +
            std::to_string(pedestrian_spawns_.size()) +
            " pedestrians");
}

bool FireScenario::resolve_firetruck_spawns(
    std::string& error) {
    if (firetruck_spawns_.size() >= config_.firetruck_count) {
        placement_attempts_ = 0;
        return true;
    }
    std::uniform_real_distribution<float> angle_distribution(
        0.0f,
        2.0f * kPi);
    std::uniform_real_distribution<float> radius_distribution(
        kFireTruckMinimumDistanceMeters,
        kFireTruckMaximumDistanceMeters);
    for (std::size_t budget = 0;
         budget < kPlacementAttemptsPerFrame;
         ++budget) {
        if (placement_attempts_++ >= kMaximumPlacementAttempts) {
            error =
                "Could not resolve enough unique firetruck road nodes "
                "with a finite inbound route to the event";
            return false;
        }
        const float angle =
            angle_distribution(firetruck_random_);
        const float radius =
            radius_distribution(firetruck_random_);
        Vector3 node{};
        float heading = 0.0f;
        if (!PATHFIND::GET_CLOSEST_VEHICLE_NODE_WITH_HEADING(
                event_position_.x + std::cos(angle) * radius,
                event_position_.y + std::sin(angle) * radius,
                event_position_.z,
                &node,
                &heading,
                1,
                3.0f,
                0)) {
            continue;
        }
        const ScenarioVector3 position = to_scenario_vector(node);
        const float event_distance =
            distance_between(position, event_position_);
        const float road_distance =
            PATHFIND::CALCULATE_TRAVEL_DISTANCE_BETWEEN_POINTS(
                position.x,
                position.y,
                position.z,
                event_position_.x,
                event_position_.y,
                event_position_.z);
        const float heading_cosine =
            heading_toward_cosine(
                heading,
                position,
                event_position_);
        const bool duplicate = std::any_of(
            firetruck_spawns_.begin(),
            firetruck_spawns_.end(),
            [&](const SpawnPoint& point) {
                return distance_between(position, point.position) <
                    12.0f;
            });
        if (event_distance < kFireTruckMinimumDistanceMeters ||
            event_distance > kFireTruckMaximumDistanceMeters ||
            !std::isfinite(road_distance) ||
            road_distance <= 0.0f ||
            road_distance > kFireTruckMaximumRoadDistanceMeters ||
            heading_cosine < kFireTruckMinimumHeadingCosine ||
            duplicate) {
            continue;
        }
        firetruck_spawns_.push_back({position, heading, 0});
        placement_attempts_ = 0;
        if (firetruck_spawns_.size() >= config_.firetruck_count) {
            return true;
        }
    }
    return true;
}

bool FireScenario::resolve_pedestrian_spawns(
    std::string& error) {
    if (pedestrian_spawns_.size() >= config_.pedestrian_count) {
        placement_attempts_ = 0;
        return true;
    }
    if (config_.pedestrian_count == 0) {
        placement_attempts_ = 0;
        return true;
    }

    std::array<std::size_t, 4> required_per_band{};
    for (std::size_t index = 0;
         index < config_.pedestrian_count;
         ++index) {
        ++required_per_band[index % required_per_band.size()];
    }
    std::array<std::size_t, 4> candidate_targets{};
    bool candidates_complete = true;
    for (std::size_t band = 0;
         band < required_per_band.size();
         ++band) {
        candidate_targets[band] =
            required_per_band[band] *
            kPedestrianCandidateMultiplier;
        if (pedestrian_candidate_spawns_[band].size() <
            candidate_targets[band]) {
            candidates_complete = false;
        }
    }

    if (candidates_complete) {
        std::array<std::vector<SpawnPoint>, 4> selected;
        for (std::size_t band = 0;
             band < required_per_band.size();
             ++band) {
            const auto& candidates =
                pedestrian_candidate_spawns_[band];
            if (required_per_band[band] == 0) {
                continue;
            }
            std::vector<bool> used(candidates.size(), false);
            selected[band].reserve(required_per_band[band]);
            for (std::size_t selection = 0;
                 selection < required_per_band[band];
                 ++selection) {
                std::size_t best_index = candidates.size();
                float best_separation = -1.0f;
                for (std::size_t candidate_index = 0;
                     candidate_index < candidates.size();
                     ++candidate_index) {
                    if (used[candidate_index]) {
                        continue;
                    }
                    const float angle = event_angle(
                        event_position_,
                        candidates[candidate_index].position);
                    float minimum_separation = 2.0f * kPi;
                    for (const SpawnPoint& chosen :
                         selected[band]) {
                        minimum_separation = (std::min)(
                            minimum_separation,
                            wrapped_angle_distance(
                                angle,
                                event_angle(
                                    event_position_,
                                    chosen.position)));
                    }
                    if (selected[band].empty()) {
                        minimum_separation = 2.0f * kPi;
                    }
                    if (minimum_separation > best_separation) {
                        best_separation = minimum_separation;
                        best_index = candidate_index;
                    }
                }
                if (best_index == candidates.size()) {
                    error =
                        "Pedestrian angular selection exhausted a "
                        "distance-band candidate pool";
                    return false;
                }
                used[best_index] = true;
                SpawnPoint chosen = candidates[best_index];
                chosen.activation_offset_ms = activation_offset_ms(
                    horizontal_distance_between(
                        chosen.position,
                        event_position_),
                    pedestrian_activation_random_);
                selected[band].push_back(chosen);
            }
        }

        std::array<std::size_t, 4> next_in_band{};
        pedestrian_spawns_.clear();
        pedestrian_spawns_.reserve(config_.pedestrian_count);
        for (std::size_t index = 0;
             index < config_.pedestrian_count;
             ++index) {
            const std::size_t band =
                index % selected.size();
            pedestrian_spawns_.push_back(
                selected[band][next_in_band[band]++]);
        }
        placement_attempts_ = 0;
        return true;
    }

    std::size_t target_band = 0;
    bool found_target_band = false;
    for (std::size_t offset = 0;
         offset < required_per_band.size();
         ++offset) {
        const std::size_t band =
            (placement_attempts_ + offset) %
            required_per_band.size();
        if (pedestrian_candidate_spawns_[band].size() <
            candidate_targets[band]) {
            target_band = band;
            found_target_band = true;
            break;
        }
    }
    if (!found_target_band) {
        error =
            "Pedestrian candidate pool bookkeeping became inconsistent";
        return false;
    }

    std::uniform_real_distribution<float> angle_distribution(
        0.0f,
        2.0f * kPi);
    std::uniform_real_distribution<float> radius_distribution(
        kPedestrianDistanceBandsMeters[target_band][0],
        kPedestrianDistanceBandsMeters[target_band][1]);
    std::uniform_int_distribution<std::size_t> model_distribution(
        0,
        kCivilianModels.size() - 1);
    for (std::size_t budget = 0;
         budget < kPlacementAttemptsPerFrame;
         ++budget) {
        if (placement_attempts_++ >= kMaximumPlacementAttempts) {
            error =
                "Could not resolve enough unique pedestrian safe coordinates; "
                "candidate_counts=" +
                std::to_string(
                    pedestrian_candidate_spawns_[0].size()) +
                "," +
                std::to_string(
                    pedestrian_candidate_spawns_[1].size()) +
                "," +
                std::to_string(
                    pedestrian_candidate_spawns_[2].size()) +
                "," +
                std::to_string(
                    pedestrian_candidate_spawns_[3].size()) +
                ", native_query_failures=" +
                std::to_string(pedestrian_query_failures_) +
                ", bounds_rejections=" +
                std::to_string(pedestrian_bounds_rejections_) +
                ", duplicate_rejections=" +
                std::to_string(pedestrian_duplicate_rejections_);
            return false;
        }
        const float angle =
            angle_distribution(pedestrian_position_random_);
        const float radius =
            radius_distribution(pedestrian_position_random_);
        const float candidate_x =
            event_position_.x + std::cos(angle) * radius;
        const float candidate_y =
            event_position_.y + std::sin(angle) * radius;
        bool placed = false;
        for (const BOOL sidewalk_only : {TRUE, FALSE}) {
            Vector3 safe{};
            if (!PATHFIND::GET_SAFE_COORD_FOR_PED(
                    candidate_x,
                    candidate_y,
                    event_position_.z + 5.0f,
                    sidewalk_only,
                    &safe,
                    0)) {
                ++pedestrian_query_failures_;
                continue;
            }
            const ScenarioVector3 position =
                to_scenario_vector(safe);
            const float dx = position.x - event_position_.x;
            const float dy = position.y - event_position_.y;
            const float horizontal_distance =
                std::sqrt(dx * dx + dy * dy);
            const float vertical_offset =
                std::abs(position.z - event_position_.z);
            if (horizontal_distance <
                    kPedestrianDistanceBandsMeters[target_band][0] ||
                horizontal_distance >
                    kPedestrianDistanceBandsMeters[target_band][1] ||
                vertical_offset >
                    kPedestrianMaximumVerticalOffsetMeters) {
                ++pedestrian_bounds_rejections_;
                continue;
            }
            bool duplicate = false;
            for (const auto& candidates :
                 pedestrian_candidate_spawns_) {
                duplicate = duplicate ||
                    std::any_of(
                        candidates.begin(),
                        candidates.end(),
                        [&](const SpawnPoint& point) {
                            return distance_between(
                                       position,
                                       point.position) < 2.0f;
                        });
            }
            if (duplicate) {
                ++pedestrian_duplicate_rejections_;
                continue;
            }
            const Hash model = GAMEPLAY::GET_HASH_KEY(
                const_cast<char*>(
                    kCivilianModels[
                        model_distribution(
                            pedestrian_model_random_)]));
            pedestrian_candidate_spawns_[target_band].push_back(
                {
                    position,
                    heading_away_from(
                        event_position_,
                        position),
                    model,
                });
            placement_attempts_ = 0;
            placed = true;
            break;
        }
        if (placed &&
            pedestrian_candidate_spawns_[target_band].size() >=
                candidate_targets[target_band]) {
            return true;
        }
    }
    return true;
}

bool FireScenario::request_models(std::string& error) {
    const Hash source_model = GAMEPLAY::GET_HASH_KEY("blista");
    const Hash firetruck_model = GAMEPLAY::GET_HASH_KEY("firetruk");
    const Hash driver_model =
        GAMEPLAY::GET_HASH_KEY("s_m_y_fireman_01");
    if (!STREAMING::IS_MODEL_VALID(source_model) ||
        !STREAMING::IS_MODEL_IN_CDIMAGE(source_model) ||
        !STREAMING::IS_MODEL_A_VEHICLE(source_model)) {
        error = "The blista fire-source model is unavailable";
        return false;
    }
    if (config_.firetruck_count > 0 &&
        (!STREAMING::IS_MODEL_VALID(firetruck_model) ||
         !STREAMING::IS_MODEL_IN_CDIMAGE(firetruck_model) ||
         !STREAMING::IS_MODEL_A_VEHICLE(firetruck_model))) {
        error = "The firetruk model is unavailable";
        return false;
    }
    if (config_.firetruck_count > 0 &&
        (!STREAMING::IS_MODEL_VALID(driver_model) ||
         !STREAMING::IS_MODEL_IN_CDIMAGE(driver_model))) {
        error = "The firefighter driver model is unavailable";
        return false;
    }

    requested_models_.clear();
    requested_models_.push_back(source_model);
    if (config_.firetruck_count > 0) {
        requested_models_.push_back(firetruck_model);
        requested_models_.push_back(driver_model);
    }
    for (const SpawnPoint& spawn : pedestrian_spawns_) {
        if (!STREAMING::IS_MODEL_VALID(spawn.model_hash) ||
            !STREAMING::IS_MODEL_IN_CDIMAGE(spawn.model_hash)) {
            error = "A configured civilian model is unavailable";
            return false;
        }
        requested_models_.push_back(spawn.model_hash);
    }
    std::sort(requested_models_.begin(), requested_models_.end());
    requested_models_.erase(
        std::unique(
            requested_models_.begin(),
            requested_models_.end()),
        requested_models_.end());
    for (const Hash model : requested_models_) {
        STREAMING::REQUEST_MODEL(model);
    }
    STREAMING::REQUEST_NAMED_PTFX_ASSET(
        FireVisualConfig::kPtfxAsset);
    fire_ptfx_asset_requested_ = true;
    model_deadline_ =
        std::chrono::steady_clock::now() + kModelLoadTimeout;
    return true;
}

bool FireScenario::models_loaded() const {
    return
        std::all_of(
            requested_models_.begin(),
            requested_models_.end(),
            [](Hash model) {
                return STREAMING::HAS_MODEL_LOADED(model);
            }) &&
        STREAMING::HAS_NAMED_PTFX_ASSET_LOADED(
            FireVisualConfig::kPtfxAsset);
}

void FireScenario::tick() {
    if (ambient_suppression_active_) {
        suppress_ambient_for_frame();
        const std::uint32_t now =
            static_cast<std::uint32_t>(
                GAMEPLAY::GET_GAME_TIMER());
        if (now - last_ambient_maintenance_game_timer_ms_ >=
            kAmbientMaintenanceIntervalMilliseconds) {
            std::string maintenance_error;
            if (!clear_ambient(maintenance_error)) {
                fail(maintenance_error);
                return;
            }
        }
    }
    if (lifecycle_ == ScenarioLifecycle::Running) {
        update_running();
        return;
    }
    if (lifecycle_ != ScenarioLifecycle::Preparing) {
        return;
    }

    std::string error;
    switch (prepare_stage_) {
        case PrepareStage::CleanAmbient:
            if (!clear_ambient(error)) {
                fail(error);
                return;
            }
            ambient_suppression_active_ = true;
            prepare_stage_ =
                building_blueprint_
                    ? PrepareStage::ResolveFireTruckSpawns
                    : PrepareStage::RequestModels;
            break;
        case PrepareStage::ResolveFireTruckSpawns:
            if (!resolve_firetruck_spawns(error)) {
                fail(error);
                return;
            }
            if (firetruck_spawns_.size() >=
                config_.firetruck_count) {
                prepare_stage_ =
                    PrepareStage::ResolvePedestrianSpawns;
            }
            break;
        case PrepareStage::ResolvePedestrianSpawns:
            if (!resolve_pedestrian_spawns(error)) {
                fail(error);
                return;
            }
            if (pedestrian_spawns_.size() >=
                config_.pedestrian_count) {
                commit_blueprint();
                prepare_stage_ = PrepareStage::RequestModels;
            }
            break;
        case PrepareStage::RequestModels:
            if (!request_models(error)) {
                fail(error);
                return;
            }
            prepare_stage_ = PrepareStage::WaitModels;
            break;
        case PrepareStage::WaitModels:
            if (models_loaded()) {
                prepare_stage_ = PrepareStage::SpawnSource;
                break;
            }
            for (const Hash model : requested_models_) {
                STREAMING::REQUEST_MODEL(model);
            }
            STREAMING::REQUEST_NAMED_PTFX_ASSET(
                FireVisualConfig::kPtfxAsset);
            if (std::chrono::steady_clock::now() >= model_deadline_) {
                fail(
                    "Scenario models or fire particle asset did not load "
                    "within 10 seconds");
            }
            break;
        case PrepareStage::SpawnSource:
            if (!spawn_source(error)) {
                fail(error);
                return;
            }
            prepare_stage_ = PrepareStage::SpawnFireTrucks;
            break;
        case PrepareStage::SpawnFireTrucks:
            if (next_firetruck_ < firetruck_spawns_.size()) {
                if (!spawn_firetruck(next_firetruck_, error)) {
                    fail(error);
                    return;
                }
                ++next_firetruck_;
            } else {
                prepare_stage_ = PrepareStage::SpawnPedestrians;
            }
            break;
        case PrepareStage::SpawnPedestrians:
            if (next_pedestrian_ < pedestrian_spawns_.size()) {
                if (!spawn_pedestrian(next_pedestrian_, error)) {
                    fail(error);
                    return;
                }
                ++next_pedestrian_;
            } else {
                prepare_stage_ = PrepareStage::Complete;
            }
            break;
        case PrepareStage::Complete:
            complete_prepare();
            break;
        case PrepareStage::None:
        default:
            fail("Fire scenario entered an invalid preparation stage");
            break;
    }
}

bool FireScenario::clear_ambient(std::string& error) {
    std::array<int, kMaximumWorldEntities> ped_handles{};
    std::array<int, kMaximumWorldEntities> vehicle_handles{};
    const int ped_count = worldGetAllPeds(
        ped_handles.data(),
        static_cast<int>(ped_handles.size()));
    const int vehicle_count = worldGetAllVehicles(
        vehicle_handles.data(),
        static_cast<int>(vehicle_handles.size()));
    const Ped player = PLAYER::PLAYER_PED_ID();
    const Vehicle player_vehicle =
        PED::GET_VEHICLE_PED_IS_IN(player, FALSE);
    const float radius_squared =
        kAmbientClearRadiusMeters * kAmbientClearRadiusMeters;
    const std::size_t protected_pedestrian_count_before =
        protected_pedestrians_.size();
    const std::size_t protected_vehicle_count_before =
        protected_vehicles_.size();

    std::vector<Ped> pedestrians_to_delete;
    std::vector<Vehicle> vehicles_to_delete;
    for (int index = 0; index < ped_count; ++index) {
        const Ped ped = ped_handles[static_cast<std::size_t>(index)];
        if (ped == 0 ||
            !ENTITY::DOES_ENTITY_EXIST(ped) ||
            ped == player ||
            PED::IS_PED_A_PLAYER(ped) ||
            registry_.contains_handle(ped) ||
            is_protected_handle(ped)) {
            continue;
        }
        const ScenarioVector3 position =
            to_scenario_vector(ENTITY::GET_ENTITY_COORDS(ped, TRUE));
        if (distance_squared(position, event_position_) >
            radius_squared) {
            continue;
        }
        const Vehicle ped_vehicle =
            PED::GET_VEHICLE_PED_IS_IN(ped, FALSE);
        if (ENTITY::IS_ENTITY_A_MISSION_ENTITY(ped) ||
            (ped_vehicle != 0 &&
             ENTITY::DOES_ENTITY_EXIST(ped_vehicle) &&
             ENTITY::IS_ENTITY_A_MISSION_ENTITY(
                 ped_vehicle))) {
            add_protected_pedestrian(ped);
            if (ped_vehicle != 0) {
                add_protected_vehicle(ped_vehicle);
            }
            continue;
        }
        pedestrians_to_delete.push_back(ped);
    }
    for (int index = 0; index < vehicle_count; ++index) {
        const Vehicle vehicle =
            vehicle_handles[static_cast<std::size_t>(index)];
        if (vehicle == 0 ||
            !ENTITY::DOES_ENTITY_EXIST(vehicle) ||
            vehicle == player_vehicle ||
            registry_.contains_handle(vehicle) ||
            is_protected_handle(vehicle)) {
            continue;
        }
        const ScenarioVector3 position =
            to_scenario_vector(
                ENTITY::GET_ENTITY_COORDS(vehicle, TRUE));
        if (distance_squared(position, event_position_) >
            radius_squared) {
            continue;
        }
        if (ENTITY::IS_ENTITY_A_MISSION_ENTITY(vehicle)) {
            add_protected_vehicle(vehicle);
            continue;
        }
        vehicles_to_delete.push_back(vehicle);
    }
    const std::size_t added_protected_pedestrians =
        protected_pedestrians_.size() -
        protected_pedestrian_count_before;
    const std::size_t added_protected_vehicles =
        protected_vehicles_.size() -
        protected_vehicle_count_before;
    if (added_protected_pedestrians > 0 ||
        added_protected_vehicles > 0) {
        LOGW(
            "scenario",
            "Registered protected entities entering the controlled area: " +
                std::to_string(added_protected_pedestrians) +
                " pedestrians, " +
                std::to_string(added_protected_vehicles) +
                " vehicles");
    }

    for (Ped ped : pedestrians_to_delete) {
        ENTITY::SET_ENTITY_AS_MISSION_ENTITY(ped, TRUE, TRUE);
        Entity handle = ped;
        ENTITY::DELETE_ENTITY(&handle);
        ++removed_pedestrians_;
    }
    for (Vehicle vehicle : vehicles_to_delete) {
        ENTITY::SET_ENTITY_AS_MISSION_ENTITY(vehicle, TRUE, TRUE);
        Entity handle = vehicle;
        ENTITY::DELETE_ENTITY(&handle);
        ++removed_vehicles_;
    }
    last_ambient_maintenance_game_timer_ms_ =
        static_cast<std::uint32_t>(
            GAMEPLAY::GET_GAME_TIMER());
    return true;
}

void FireScenario::add_protected_pedestrian(Ped ped) {
    if (ped != 0 &&
        std::find(
            protected_pedestrians_.begin(),
            protected_pedestrians_.end(),
            ped) == protected_pedestrians_.end()) {
        protected_pedestrians_.push_back(ped);
    }
}

void FireScenario::add_protected_vehicle(Vehicle vehicle) {
    if (vehicle != 0 &&
        std::find(
            protected_vehicles_.begin(),
            protected_vehicles_.end(),
            vehicle) == protected_vehicles_.end()) {
        protected_vehicles_.push_back(vehicle);
    }
}

bool FireScenario::spawn_source(std::string& error) {
    const Hash model = GAMEPLAY::GET_HASH_KEY("blista");
    source_vehicle_ = VEHICLE::CREATE_VEHICLE(
        model,
        event_position_.x,
        event_position_.y,
        event_position_.z,
        event_heading_,
        TRUE,
        FALSE);
    if (source_vehicle_ == 0 ||
        !ENTITY::DOES_ENTITY_EXIST(source_vehicle_)) {
        error = "GTA failed to create the fire-source vehicle";
        return false;
    }
    ENTITY::SET_ENTITY_AS_MISSION_ENTITY(source_vehicle_, TRUE, TRUE);
    ENTITY::SET_ENTITY_HEADING(source_vehicle_, event_heading_);
    VEHICLE::SET_VEHICLE_ON_GROUND_PROPERLY(source_vehicle_);
    VEHICLE::SET_VEHICLE_ENGINE_ON(
        source_vehicle_,
        FALSE,
        TRUE,
        FALSE);
    VEHICLE::SET_VEHICLE_HANDBRAKE(source_vehicle_, TRUE);
    ENTITY::FREEZE_ENTITY_POSITION(source_vehicle_, TRUE);
    source_vehicle_id_ = registry_.add(
        source_vehicle_,
        model,
        ScenarioEntityKind::Vehicle,
        ScenarioEntityRole::FireSourceVehicle,
        event_id_,
        static_cast<std::uint32_t>(GAMEPLAY::GET_GAME_TIMER()));
    return true;
}

bool FireScenario::spawn_firetruck(
    std::size_t index,
    std::string& error) {
    const SpawnPoint& spawn = firetruck_spawns_.at(index);
    const Hash vehicle_model = GAMEPLAY::GET_HASH_KEY("firetruk");
    const Hash driver_model =
        GAMEPLAY::GET_HASH_KEY("s_m_y_fireman_01");
    FireTruckActor actor;
    actor.vehicle = VEHICLE::CREATE_VEHICLE(
        vehicle_model,
        spawn.position.x,
        spawn.position.y,
        spawn.position.z,
        spawn.heading,
        TRUE,
        FALSE);
    if (actor.vehicle == 0 ||
        !ENTITY::DOES_ENTITY_EXIST(actor.vehicle)) {
        error = "GTA failed to create a firetruck";
        return false;
    }
    ENTITY::SET_ENTITY_AS_MISSION_ENTITY(actor.vehicle, TRUE, TRUE);
    ENTITY::SET_ENTITY_HEADING(actor.vehicle, spawn.heading);
    VEHICLE::SET_VEHICLE_ON_GROUND_PROPERLY(actor.vehicle);
    VEHICLE::SET_VEHICLE_ENGINE_ON(
        actor.vehicle,
        FALSE,
        TRUE,
        FALSE);
    VEHICLE::SET_VEHICLE_HANDBRAKE(actor.vehicle, TRUE);
    VEHICLE::SET_VEHICLE_SIREN(actor.vehicle, FALSE);
    ENTITY::FREEZE_ENTITY_POSITION(actor.vehicle, TRUE);

    actor.driver = PED::CREATE_PED_INSIDE_VEHICLE(
        actor.vehicle,
        6,
        driver_model,
        -1,
        TRUE,
        FALSE);
    if (actor.driver == 0 ||
        !ENTITY::DOES_ENTITY_EXIST(actor.driver)) {
        Entity vehicle = actor.vehicle;
        ENTITY::DELETE_ENTITY(&vehicle);
        error = "GTA failed to create a firefighter driver";
        return false;
    }
    ENTITY::SET_ENTITY_AS_MISSION_ENTITY(actor.driver, TRUE, TRUE);
    PED::SET_BLOCKING_OF_NON_TEMPORARY_EVENTS(actor.driver, TRUE);
    PED::SET_PED_KEEP_TASK(actor.driver, TRUE);
    PED::SET_PED_CAN_BE_DRAGGED_OUT(actor.driver, FALSE);
    PED::SET_DRIVER_ABILITY(actor.driver, 1.0f);
    PED::SET_DRIVER_AGGRESSIVENESS(actor.driver, 0.2f);

    const std::uint32_t spawn_time =
        static_cast<std::uint32_t>(GAMEPLAY::GET_GAME_TIMER());
    actor.vehicle_id = registry_.add(
        actor.vehicle,
        vehicle_model,
        ScenarioEntityKind::Vehicle,
        ScenarioEntityRole::FireTruck,
        event_id_,
        spawn_time);
    actor.driver_id = registry_.add(
        actor.driver,
        driver_model,
        ScenarioEntityKind::Pedestrian,
        ScenarioEntityRole::FirefighterDriver,
        event_id_,
        spawn_time);
    registry_.schedule_task(actor.vehicle_id, 0);
    registry_.schedule_task(actor.driver_id, 0);
    firetrucks_.push_back(actor);
    return true;
}

bool FireScenario::spawn_pedestrian(
    std::size_t index,
    std::string& error) {
    const SpawnPoint& spawn = pedestrian_spawns_.at(index);
    const Ped pedestrian = PED::CREATE_PED(
        4,
        spawn.model_hash,
        spawn.position.x,
        spawn.position.y,
        spawn.position.z,
        spawn.heading,
        TRUE,
        FALSE);
    if (pedestrian == 0 ||
        !ENTITY::DOES_ENTITY_EXIST(pedestrian)) {
        error = "GTA failed to create a fleeing pedestrian";
        return false;
    }
    ENTITY::SET_ENTITY_AS_MISSION_ENTITY(pedestrian, TRUE, TRUE);
    ENTITY::SET_ENTITY_HEADING(pedestrian, spawn.heading);
    PED::SET_BLOCKING_OF_NON_TEMPORARY_EVENTS(pedestrian, TRUE);
    ENTITY::FREEZE_ENTITY_POSITION(pedestrian, TRUE);
    pedestrians_.push_back(pedestrian);
    const std::uint64_t pedestrian_id = registry_.add(
        pedestrian,
        spawn.model_hash,
        ScenarioEntityKind::Pedestrian,
        ScenarioEntityRole::FleeingPedestrian,
        event_id_,
        static_cast<std::uint32_t>(GAMEPLAY::GET_GAME_TIMER()));
    registry_.schedule_task(
        pedestrian_id,
        spawn.activation_offset_ms);
    pedestrian_ids_.push_back(pedestrian_id);
    return true;
}

bool FireScenario::pedestrian_active(std::size_t index) const {
    if (index >= pedestrian_ids_.size()) {
        return false;
    }
    const EntityRegistry::Entry* entry =
        registry_.find(pedestrian_ids_[index]);
    return entry != nullptr &&
        entry->task_state == ScenarioTaskState::Active;
}

bool FireScenario::activate_pedestrian(
    std::size_t index,
    std::uint32_t game_timer_ms,
    std::string& error) {
    if (index >= pedestrians_.size() ||
        index >= pedestrian_ids_.size() ||
        index >= pedestrian_spawns_.size()) {
        error = "Pedestrian activation index is out of range";
        return false;
    }
    const Ped pedestrian = pedestrians_[index];
    if (pedestrian == 0 ||
        !ENTITY::DOES_ENTITY_EXIST(pedestrian)) {
        error = "A scheduled fleeing pedestrian was lost";
        return false;
    }
    EntityRegistry::Entry* entry =
        registry_.find(pedestrian_ids_[index]);
    if (entry == nullptr) {
        error = "A scheduled pedestrian is absent from the registry";
        return false;
    }
    if (entry->task_state != ScenarioTaskState::Pending) {
        return entry->task_state == ScenarioTaskState::Active;
    }

    AI::TASK_SMART_FLEE_COORD(
        pedestrian,
        event_position_.x,
        event_position_.y,
        event_position_.z,
        kPedestrianFleeDistanceMeters,
        -1,
        FALSE,
        FALSE);
    PED::SET_PED_KEEP_TASK(pedestrian, TRUE);
    const ScenarioVector3 position = to_scenario_vector(
        ENTITY::GET_ENTITY_COORDS(pedestrian, TRUE));
    registry_.start_task(
        pedestrian_ids_[index],
        event_position_,
        game_timer_ms,
        distance_between(position, event_position_));
    if (!lockstep_frozen_) {
        ENTITY::FREEZE_ENTITY_POSITION(pedestrian, FALSE);
    }
    return true;
}

void FireScenario::complete_prepare() {
    release_models();
    prepare_stage_ = PrepareStage::None;
    lifecycle_ = ScenarioLifecycle::Ready;
    LOGI(
        "scenario",
        "Fire scenario " + std::to_string(scenario_id_) +
            " is READY");
}

ScenarioOperationStatus FireScenario::start(
    ScenarioStartInfo& info,
    std::string& error) {
    if (lifecycle_ != ScenarioLifecycle::Ready) {
        error = "Fire scenario must be READY before Start";
        return ScenarioOperationStatus::NotReady;
    }
    if (source_vehicle_ == 0 ||
        !ENTITY::DOES_ENTITY_EXIST(source_vehicle_)) {
        fail("Fire-source vehicle disappeared before Start");
        error = failure_;
        return ScenarioOperationStatus::StartFailed;
    }
    for (const FireTruckActor& actor : firetrucks_) {
        if (!ENTITY::DOES_ENTITY_EXIST(actor.vehicle) ||
            !ENTITY::DOES_ENTITY_EXIST(actor.driver)) {
            fail("A firetruck actor disappeared before Start");
            error = failure_;
            return ScenarioOperationStatus::StartFailed;
        }
    }
    for (Ped pedestrian : pedestrians_) {
        if (!ENTITY::DOES_ENTITY_EXIST(pedestrian)) {
            fail("A fleeing pedestrian disappeared before Start");
            error = failure_;
            return ScenarioOperationStatus::StartFailed;
        }
    }

    start_game_timer_ms_ =
        static_cast<std::uint32_t>(GAMEPLAY::GET_GAME_TIMER());
    start_frame_count_ =
        static_cast<std::uint32_t>(GAMEPLAY::GET_FRAME_COUNT());

    VEHICLE::SET_VEHICLE_ENGINE_ON(
        source_vehicle_,
        TRUE,
        TRUE,
        FALSE);
    VEHICLE::SET_VEHICLE_ENGINE_HEALTH(
        source_vehicle_,
        -1000.0f);
    VEHICLE::SET_VEHICLE_DAMAGE(
        source_vehicle_,
        0.0f,
        0.0f,
        0.0f,
        2000.0f,
        5.0f,
        TRUE);
    entity_fire_handle_ = FIRE::START_ENTITY_FIRE(source_vehicle_);
    script_fire_handle_ = FIRE::START_SCRIPT_FIRE(
        event_position_.x,
        event_position_.y,
        event_position_.z + 0.5f,
        25,
        TRUE);
    if (static_cast<std::int32_t>(script_fire_handle_) < 0) {
        fail("GTA failed to create the persistent script fire");
        error = failure_;
        return ScenarioOperationStatus::StartFailed;
    }
    script_fire_created_ = true;
    if (!STREAMING::HAS_NAMED_PTFX_ASSET_LOADED(
            FireVisualConfig::kPtfxAsset)) {
        fail("The fire particle asset was lost before Start");
        error = failure_;
        return ScenarioOperationStatus::StartFailed;
    }
    const auto register_visual_effect = [&](Any handle) {
        if (handle == 0 ||
            !GRAPHICS::DOES_PARTICLE_FX_LOOPED_EXIST(
                static_cast<int>(handle))) {
            return false;
        }
        GRAPHICS::_SET_PARTICLE_FX_LOOPED_RANGE(
            handle,
            FireVisualConfig::kMaximumRenderRangeMeters);
        visual_fire_handles_.push_back(handle);
        return true;
    };
    const auto start_fire_visual_effect = [&]() {
        GRAPHICS::_SET_PTFX_ASSET_NEXT_CALL(
            FireVisualConfig::kPtfxAsset);
        const Any handle =
            GRAPHICS::START_PARTICLE_FX_LOOPED_AT_COORD(
                kFirePtfxEffect,
                event_position_.x,
                event_position_.y,
                event_position_.z + 0.35f,
                0.0f,
                0.0f,
                event_heading_,
                2.5f,
                FALSE,
                FALSE,
                FALSE,
                FALSE);
        return register_visual_effect(handle);
    };
    const auto start_smoke_visual_effect = [&]() {
        GRAPHICS::_SET_PTFX_ASSET_NEXT_CALL(
            FireVisualConfig::kPtfxAsset);
        const Any handle =
            GRAPHICS::START_PARTICLE_FX_LOOPED_AT_COORD(
                FireVisualConfig::kSmokePtfxEffect,
                event_position_.x,
                event_position_.y,
                event_position_.z +
                    FireVisualConfig::kSmokeEmitterZOffsetMeters,
                0.0f,
                0.0f,
                event_heading_,
                FireVisualConfig::kSmokeScale,
                FALSE,
                FALSE,
                FALSE,
                FALSE);
        if (!register_visual_effect(handle)) {
            return false;
        }
        GRAPHICS::SET_PARTICLE_FX_LOOPED_COLOUR(
            handle,
            0.12f,
            0.12f,
            0.12f,
            FALSE);
        GRAPHICS::SET_PARTICLE_FX_LOOPED_ALPHA(
            handle,
            1.0f);
        return true;
    };
    if (!start_fire_visual_effect() ||
        !start_smoke_visual_effect()) {
        fail(
            "GTA failed to create the required fire and smoke "
            "effects");
        error = failure_;
        return ScenarioOperationStatus::StartFailed;
    }

    for (FireTruckActor& actor : firetrucks_) {
        ENTITY::FREEZE_ENTITY_POSITION(
            actor.vehicle,
            lockstep_frozen_ ? TRUE : FALSE);
        VEHICLE::SET_VEHICLE_HANDBRAKE(actor.vehicle, FALSE);
        VEHICLE::SET_VEHICLE_UNDRIVEABLE(actor.vehicle, FALSE);
        VEHICLE::SET_VEHICLE_ENGINE_ON(
            actor.vehicle,
            TRUE,
            TRUE,
            FALSE);
        VEHICLE::SET_VEHICLE_SIREN(actor.vehicle, TRUE);
        if (!PED::IS_PED_IN_VEHICLE(
                actor.driver,
                actor.vehicle,
                FALSE)) {
            fail("A firefighter driver left its firetruck before task start");
            error = failure_;
            return ScenarioOperationStatus::StartFailed;
        }
        AI::TASK_VEHICLE_DRIVE_TO_COORD_LONGRANGE(
            actor.driver,
            actor.vehicle,
            event_position_.x,
            event_position_.y,
            event_position_.z,
            kFireTruckSpeedMetersPerSecond,
            kDrivingStyle,
            kFireTruckStopRangeMeters);
        const ScenarioVector3 position = to_scenario_vector(
            ENTITY::GET_ENTITY_COORDS(actor.vehicle, TRUE));
        const float distance =
            distance_between(position, event_position_);
        registry_.start_task(
            actor.vehicle_id,
            event_position_,
            start_game_timer_ms_,
            distance);
        registry_.start_task(
            actor.driver_id,
            event_position_,
            start_game_timer_ms_,
            distance);
    }
    for (std::size_t index = 0;
         index < pedestrians_.size();
         ++index) {
        const SpawnPoint& spawn = pedestrian_spawns_[index];
        if (spawn.activation_offset_ms == 0 &&
            !activate_pedestrian(
                index,
                start_game_timer_ms_,
                error)) {
            fail(error);
            return ScenarioOperationStatus::StartFailed;
        }
        if (!pedestrian_active(index)) {
            ENTITY::FREEZE_ENTITY_POSITION(
                pedestrians_[index],
                TRUE);
        }
    }

    lifecycle_ = ScenarioLifecycle::Running;
    if (lockstep_frozen_) {
        registry_.freeze_kinematics();
    }
    info.scenario_id = scenario_id_;
    info.game_timer_ms = start_game_timer_ms_;
    info.frame_count = start_frame_count_;
    LOGI(
        "scenario",
        "Started fire scenario " + std::to_string(scenario_id_));
    return ScenarioOperationStatus::Ok;
}

void FireScenario::set_lockstep_frozen(bool frozen) {
    lockstep_frozen_ = frozen;
    if (lifecycle_ != ScenarioLifecycle::Running) {
        return;
    }
    if (frozen) {
        registry_.freeze_kinematics();
    }
    for (const FireTruckActor& actor : firetrucks_) {
        if (actor.vehicle != 0 &&
            ENTITY::DOES_ENTITY_EXIST(actor.vehicle)) {
            ENTITY::FREEZE_ENTITY_POSITION(
                actor.vehicle,
                frozen ? TRUE : FALSE);
        }
    }
    for (std::size_t index = 0;
         index < pedestrians_.size();
         ++index) {
        const Ped pedestrian = pedestrians_[index];
        if (pedestrian != 0 &&
            ENTITY::DOES_ENTITY_EXIST(pedestrian)) {
            ENTITY::FREEZE_ENTITY_POSITION(
                pedestrian,
                frozen || !pedestrian_active(index)
                    ? TRUE
                    : FALSE);
        }
    }
    if (!frozen) {
        registry_.restore_frozen_velocities();
        registry_.unfreeze_kinematics();
    }
}

void FireScenario::update_running() {
    if (!visual_fire_effects_alive()) {
        fail(
            "A required visual fire particle effect stopped while "
            "the scenario was running");
        return;
    }
    if (lockstep_frozen_) {
        return;
    }
    if (source_vehicle_ == 0 ||
        !ENTITY::DOES_ENTITY_EXIST(source_vehicle_)) {
        fail("Fire-source vehicle was lost while the scenario was running");
        return;
    }
    const std::uint32_t now =
        static_cast<std::uint32_t>(GAMEPLAY::GET_GAME_TIMER());
    const std::uint32_t elapsed =
        now - start_game_timer_ms_;
    for (std::size_t index = 0;
         index < pedestrian_ids_.size();
         ++index) {
        const EntityRegistry::Entry* entry =
            registry_.find(pedestrian_ids_[index]);
        if (entry != nullptr &&
            entry->task_state == ScenarioTaskState::Pending &&
            elapsed >= entry->planned_activation_offset_ms) {
            std::string activation_error;
            if (!activate_pedestrian(
                    index,
                    now,
                    activation_error)) {
                fail(activation_error);
                return;
            }
        }
    }
    if (now - start_game_timer_ms_ >=
            kFireActivationTimeoutMilliseconds &&
        !FIRE::IS_ENTITY_ON_FIRE(source_vehicle_) &&
        FIRE::GET_NUMBER_OF_FIRES_IN_RANGE(
            event_position_.x,
            event_position_.y,
            event_position_.z,
            5.0f) <= 0) {
        fail("Fire did not become active within three seconds");
        return;
    }
    registry_.update_tasks(event_position_, now);
    for (const FireTruckActor& actor : firetrucks_) {
        const EntityRegistry::Entry* vehicle =
            registry_.find(actor.vehicle_id);
        const EntityRegistry::Entry* driver =
            registry_.find(actor.driver_id);
        if (vehicle != nullptr &&
            driver != nullptr &&
            driver->task_state != ScenarioTaskState::Lost) {
            registry_.set_task_state(
                actor.driver_id,
                vehicle->task_state);
        }
    }
}

bool FireScenario::visual_fire_effects_alive() const {
    if (visual_fire_handles_.size() != 2) {
        return false;
    }
    return std::all_of(
        visual_fire_handles_.begin(),
        visual_fire_handles_.end(),
        [](Any handle) {
            return
                handle != 0 &&
                GRAPHICS::DOES_PARTICLE_FX_LOOPED_EXIST(
                    static_cast<int>(handle));
        });
}

ScenarioOperationStatus FireScenario::snapshot(
    ScenarioSnapshot& output,
    std::string& error) const {
    if (lifecycle_ == ScenarioLifecycle::Empty) {
        error = "No fire scenario exists";
        return ScenarioOperationStatus::IdMismatch;
    }
    output = {};
    output.scenario_id = scenario_id_;
    output.blueprint_id = blueprint_id_;
    output.seed = config_.seed;
    output.lifecycle = lifecycle_;
    output.game_timer_ms =
        static_cast<std::uint32_t>(GAMEPLAY::GET_GAME_TIMER());
    output.frame_count =
        static_cast<std::uint32_t>(GAMEPLAY::GET_FRAME_COUNT());
    output.start_game_timer_ms = start_game_timer_ms_;
    output.start_frame_count = start_frame_count_;
    output.requested_anchor = config_.anchor;
    output.event_position = event_position_;
    output.event_active =
        lifecycle_ == ScenarioLifecycle::Running &&
        source_vehicle_ != 0 &&
        ENTITY::DOES_ENTITY_EXIST(source_vehicle_) &&
        (FIRE::IS_ENTITY_ON_FIRE(source_vehicle_) ||
         FIRE::GET_NUMBER_OF_FIRES_IN_RANGE(
             event_position_.x,
             event_position_.y,
             event_position_.z,
             5.0f) > 0);
    output.removed_pedestrians = removed_pedestrians_;
    output.removed_vehicles = removed_vehicles_;
    output.failure_message = failure_;
    output.protected_entities =
        protected_entity_snapshots();
    count_ambient(
        output.ambient_pedestrians,
        output.ambient_vehicles);
    output.entities = registry_.snapshots();
    return ScenarioOperationStatus::Ok;
}

void FireScenario::suppress_ambient_for_frame() const {
    PED::SET_PED_DENSITY_MULTIPLIER_THIS_FRAME(0.0f);
    PED::SET_SCENARIO_PED_DENSITY_MULTIPLIER_THIS_FRAME(
        0.0f,
        0.0f);
    VEHICLE::SET_VEHICLE_DENSITY_MULTIPLIER_THIS_FRAME(0.0f);
    VEHICLE::SET_RANDOM_VEHICLE_DENSITY_MULTIPLIER_THIS_FRAME(0.0f);
    VEHICLE::SET_PARKED_VEHICLE_DENSITY_MULTIPLIER_THIS_FRAME(0.0f);
}

void FireScenario::count_ambient(
    std::uint32_t& pedestrians,
    std::uint32_t& vehicles) const {
    pedestrians = 0;
    vehicles = 0;
    std::array<int, kMaximumWorldEntities> ped_handles{};
    std::array<int, kMaximumWorldEntities> vehicle_handles{};
    const int ped_count = worldGetAllPeds(
        ped_handles.data(),
        static_cast<int>(ped_handles.size()));
    const int vehicle_count = worldGetAllVehicles(
        vehicle_handles.data(),
        static_cast<int>(vehicle_handles.size()));
    const Ped player = PLAYER::PLAYER_PED_ID();
    const Vehicle player_vehicle =
        PED::GET_VEHICLE_PED_IS_IN(player, FALSE);
    const float radius_squared =
        kAmbientClearRadiusMeters * kAmbientClearRadiusMeters;
    for (int index = 0; index < ped_count; ++index) {
        const Ped ped = ped_handles[static_cast<std::size_t>(index)];
        if (ped == 0 ||
            ped == player ||
            PED::IS_PED_A_PLAYER(ped) ||
            registry_.contains_handle(ped) ||
            is_protected_handle(ped) ||
            !ENTITY::DOES_ENTITY_EXIST(ped)) {
            continue;
        }
        const ScenarioVector3 position =
            to_scenario_vector(ENTITY::GET_ENTITY_COORDS(ped, TRUE));
        if (distance_squared(position, event_position_) <=
            radius_squared) {
            ++pedestrians;
        }
    }
    for (int index = 0; index < vehicle_count; ++index) {
        const Vehicle vehicle =
            vehicle_handles[static_cast<std::size_t>(index)];
        if (vehicle == 0 ||
            vehicle == player_vehicle ||
            registry_.contains_handle(vehicle) ||
            is_protected_handle(vehicle) ||
            !ENTITY::DOES_ENTITY_EXIST(vehicle)) {
            continue;
        }
        const ScenarioVector3 position =
            to_scenario_vector(
                ENTITY::GET_ENTITY_COORDS(vehicle, TRUE));
        if (distance_squared(position, event_position_) <=
            radius_squared) {
            ++vehicles;
        }
    }
}

bool FireScenario::is_protected_handle(Entity handle) const {
    return std::find(
               protected_pedestrians_.begin(),
               protected_pedestrians_.end(),
               static_cast<Ped>(handle)) !=
            protected_pedestrians_.end() ||
        std::find(
               protected_vehicles_.begin(),
               protected_vehicles_.end(),
               static_cast<Vehicle>(handle)) !=
            protected_vehicles_.end();
}

std::vector<ScenarioProtectedEntitySnapshot>
FireScenario::protected_entity_snapshots() const {
    std::vector<ScenarioProtectedEntitySnapshot> output;
    output.reserve(
        protected_pedestrians_.size() +
        protected_vehicles_.size());
    const auto append = [&output](
                            Entity handle,
                            ScenarioEntityKind kind) {
        ScenarioProtectedEntitySnapshot snapshot;
        snapshot.gta_handle =
            static_cast<std::int32_t>(handle);
        snapshot.kind = kind;
        snapshot.exists =
            handle != 0 &&
            ENTITY::DOES_ENTITY_EXIST(handle);
        if (snapshot.exists) {
            snapshot.model_hash = static_cast<std::uint32_t>(
                ENTITY::GET_ENTITY_MODEL(handle));
            snapshot.position = to_scenario_vector(
                ENTITY::GET_ENTITY_COORDS(handle, TRUE));
        }
        output.push_back(snapshot);
    };
    for (Ped ped : protected_pedestrians_) {
        append(ped, ScenarioEntityKind::Pedestrian);
    }
    for (Vehicle vehicle : protected_vehicles_) {
        append(vehicle, ScenarioEntityKind::Vehicle);
    }
    return output;
}

void FireScenario::release_models() {
    for (const Hash model : requested_models_) {
        STREAMING::SET_MODEL_AS_NO_LONGER_NEEDED(model);
    }
    requested_models_.clear();
}

void FireScenario::cleanup_owned_resources() {
    for (const Any handle : visual_fire_handles_) {
        if (handle != 0 &&
            GRAPHICS::DOES_PARTICLE_FX_LOOPED_EXIST(
                static_cast<int>(handle))) {
            GRAPHICS::STOP_PARTICLE_FX_LOOPED(
                handle,
                FALSE);
        }
    }
    visual_fire_handles_.clear();
    if (source_vehicle_ != 0 &&
        ENTITY::DOES_ENTITY_EXIST(source_vehicle_)) {
        FIRE::STOP_ENTITY_FIRE(source_vehicle_);
    }
    if (script_fire_created_) {
        FIRE::REMOVE_SCRIPT_FIRE(script_fire_handle_);
    }
    script_fire_created_ = false;
    script_fire_handle_ = 0;
    entity_fire_handle_ = 0;
    registry_.delete_all();
    registry_.clear();
    source_vehicle_ = 0;
    source_vehicle_id_ = 0;
    firetrucks_.clear();
    pedestrians_.clear();
    pedestrian_ids_.clear();
    protected_pedestrians_.clear();
    protected_vehicles_.clear();
    next_firetruck_ = 0;
    next_pedestrian_ = 0;
    placement_attempts_ = 0;
    pedestrian_query_failures_ = 0;
    pedestrian_bounds_rejections_ = 0;
    pedestrian_duplicate_rejections_ = 0;
    release_models();
    if (fire_ptfx_asset_requested_) {
        STREAMING::_REMOVE_NAMED_PTFX_ASSET(
            FireVisualConfig::kPtfxAsset);
        fire_ptfx_asset_requested_ = false;
    }
    ambient_suppression_active_ = false;
    last_ambient_maintenance_game_timer_ms_ = 0;
}

void FireScenario::fail(std::string message) {
    LOGE(
        "scenario",
        "Fire scenario " + std::to_string(scenario_id_) +
            " failed: " + message);
    cleanup_owned_resources();
    prepare_stage_ = PrepareStage::None;
    lifecycle_ = ScenarioLifecycle::Failed;
    failure_ = std::move(message);
}

void FireScenario::reset() {
    cleanup_owned_resources();
    lifecycle_ = ScenarioLifecycle::Empty;
    prepare_stage_ = PrepareStage::None;
    config_ = {};
    scenario_id_ = 0;
    event_id_ = 0;
    blueprint_id_ = 0;
    building_blueprint_ = false;
    event_position_ = {};
    event_heading_ = 0.0f;
    firetruck_spawns_.clear();
    pedestrian_spawns_.clear();
    for (auto& candidates : pedestrian_candidate_spawns_) {
        candidates.clear();
    }
    removed_pedestrians_ = 0;
    removed_vehicles_ = 0;
    start_game_timer_ms_ = 0;
    start_frame_count_ = 0;
    failure_.clear();
}

ScenarioLifecycle FireScenario::lifecycle() const {
    return lifecycle_;
}

std::uint64_t FireScenario::scenario_id() const {
    return scenario_id_;
}

const std::string& FireScenario::failure() const {
    return failure_;
}
