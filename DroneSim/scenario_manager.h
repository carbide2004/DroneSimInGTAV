#pragma once

#include "fire_scenario.h"
#include "scenario_types.h"

#include <cstdint>
#include <string>

class ScenarioManager {
public:
    static ScenarioManager& instance();

    ScenarioOperationStatus prepare_fire(
        const FireScenarioConfig& config,
        std::uint64_t& scenario_id,
        std::string& error);
    ScenarioOperationStatus start(
        std::uint64_t scenario_id,
        ScenarioStartInfo& info,
        std::string& error);
    ScenarioOperationStatus snapshot(
        std::uint64_t scenario_id,
        ScenarioSnapshot& output,
        std::string& error) const;
    ScenarioOperationStatus reset(
        std::uint64_t scenario_id,
        std::string& error);
    void force_reset();
    void tick();
    void set_lockstep_frozen(bool frozen);

    ScenarioLifecycle lifecycle() const;

private:
    ScenarioManager() = default;

    FireScenario fire_;
    std::uint64_t next_scenario_id_ = 1;
    bool lockstep_frozen_ = false;
};
