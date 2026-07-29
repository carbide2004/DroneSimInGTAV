#pragma once

#include "entity_registry.h"
#include "scenario_types.h"
#include "types.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

class FireScenario {
public:
    ScenarioOperationStatus prepare(
        std::uint64_t scenario_id,
        const FireScenarioConfig& config,
        std::string& error);
    void tick();
    ScenarioOperationStatus start(
        ScenarioStartInfo& info,
        std::string& error);
    ScenarioOperationStatus snapshot(
        ScenarioSnapshot& output,
        std::string& error) const;
    void reset();

    ScenarioLifecycle lifecycle() const;
    std::uint64_t scenario_id() const;
    const std::string& failure() const;

private:
    enum class PrepareStage {
        None,
        CleanAmbient,
        ResolveFireTruckSpawns,
        ResolvePedestrianSpawns,
        RequestModels,
        WaitModels,
        SpawnSource,
        SpawnFireTrucks,
        SpawnPedestrians,
        Complete,
    };

    struct SpawnPoint {
        ScenarioVector3 position;
        float heading = 0.0f;
        Hash model_hash = 0;
    };

    struct Blueprint {
        bool valid = false;
        std::uint64_t id = 0;
        std::uint64_t seed = 0;
        ScenarioVector3 requested_anchor;
        ScenarioVector3 event_position;
        float event_heading = 0.0f;
        std::vector<SpawnPoint> firetruck_spawns;
        std::vector<SpawnPoint> pedestrian_spawns;
    };

    struct FireTruckActor {
        Vehicle vehicle = 0;
        Ped driver = 0;
        std::uint64_t vehicle_id = 0;
        std::uint64_t driver_id = 0;
    };

    ScenarioOperationStatus validate_area(
        std::string& error);
    ScenarioOperationStatus resolve_event(
        std::string& error);
    bool reuse_blueprint(std::string& error);
    void commit_blueprint();
    bool resolve_firetruck_spawns(std::string& error);
    bool resolve_pedestrian_spawns(std::string& error);
    bool request_models(std::string& error);
    bool models_loaded() const;
    bool clear_ambient(std::string& error);
    void add_protected_pedestrian(Ped ped);
    void add_protected_vehicle(Vehicle vehicle);
    bool spawn_source(std::string& error);
    bool spawn_firetruck(std::size_t index, std::string& error);
    bool spawn_pedestrian(std::size_t index, std::string& error);
    void complete_prepare();
    void update_running();
    void suppress_ambient_for_frame() const;
    void count_ambient(
        std::uint32_t& pedestrians,
        std::uint32_t& vehicles) const;
    bool is_protected_handle(Entity handle) const;
    std::vector<ScenarioProtectedEntitySnapshot>
    protected_entity_snapshots() const;
    void release_models();
    void cleanup_owned_resources();
    void fail(std::string message);

    ScenarioLifecycle lifecycle_ = ScenarioLifecycle::Empty;
    PrepareStage prepare_stage_ = PrepareStage::None;
    FireScenarioConfig config_;
    std::uint64_t scenario_id_ = 0;
    std::uint64_t event_id_ = 0;
    std::uint64_t blueprint_id_ = 0;
    bool building_blueprint_ = false;
    Blueprint cached_blueprint_;
    std::mt19937_64 firetruck_random_;
    std::mt19937_64 pedestrian_position_random_;
    std::mt19937_64 pedestrian_model_random_;
    ScenarioVector3 event_position_;
    float event_heading_ = 0.0f;
    std::vector<SpawnPoint> firetruck_spawns_;
    std::vector<SpawnPoint> pedestrian_spawns_;
    std::vector<Hash> requested_models_;
    std::chrono::steady_clock::time_point model_deadline_;
    bool ambient_suppression_active_ = false;
    std::uint32_t last_ambient_maintenance_game_timer_ms_ = 0;
    std::uint32_t removed_pedestrians_ = 0;
    std::uint32_t removed_vehicles_ = 0;
    EntityRegistry registry_;
    Vehicle source_vehicle_ = 0;
    std::uint64_t source_vehicle_id_ = 0;
    Any entity_fire_handle_ = 0;
    Any script_fire_handle_ = 0;
    bool script_fire_created_ = false;
    std::vector<Any> visual_fire_handles_;
    bool fire_ptfx_asset_requested_ = false;
    std::vector<FireTruckActor> firetrucks_;
    std::vector<Ped> pedestrians_;
    std::vector<std::uint64_t> pedestrian_ids_;
    std::vector<Ped> protected_pedestrians_;
    std::vector<Vehicle> protected_vehicles_;
    std::size_t next_firetruck_ = 0;
    std::size_t next_pedestrian_ = 0;
    std::size_t placement_attempts_ = 0;
    std::size_t pedestrian_query_failures_ = 0;
    std::size_t pedestrian_bounds_rejections_ = 0;
    std::size_t pedestrian_duplicate_rejections_ = 0;
    std::uint32_t start_game_timer_ms_ = 0;
    std::uint32_t start_frame_count_ = 0;
    std::string failure_;
};
