#include "scenario_manager.h"

ScenarioManager& ScenarioManager::instance() {
    static ScenarioManager manager;
    return manager;
}

ScenarioOperationStatus ScenarioManager::prepare_fire(
    const FireScenarioConfig& config,
    std::uint64_t& scenario_id,
    std::string& error) {
    if (fire_.lifecycle() != ScenarioLifecycle::Empty) {
        error = "A scenario is already active; Reset it first";
        return ScenarioOperationStatus::AlreadyActive;
    }
    scenario_id = next_scenario_id_++;
    if (scenario_id == 0) {
        scenario_id = next_scenario_id_++;
    }
    const ScenarioOperationStatus status =
        fire_.prepare(scenario_id, config, error);
    fire_.set_lockstep_frozen(lockstep_frozen_);
    if (status != ScenarioOperationStatus::Ok) {
        scenario_id = 0;
        return status;
    }
    return ScenarioOperationStatus::Ok;
}

ScenarioOperationStatus ScenarioManager::start(
    std::uint64_t scenario_id,
    ScenarioStartInfo& info,
    std::string& error) {
    if (scenario_id == 0 ||
        fire_.scenario_id() != scenario_id) {
        error = "scenario_id does not match the active scenario";
        return ScenarioOperationStatus::IdMismatch;
    }
    return fire_.start(info, error);
}

ScenarioOperationStatus ScenarioManager::snapshot(
    std::uint64_t scenario_id,
    ScenarioSnapshot& output,
    std::string& error) const {
    if (scenario_id == 0 ||
        fire_.scenario_id() != scenario_id) {
        error = "scenario_id does not match the active scenario";
        return ScenarioOperationStatus::IdMismatch;
    }
    return fire_.snapshot(output, error);
}

ScenarioOperationStatus ScenarioManager::reset(
    std::uint64_t scenario_id,
    std::string& error) {
    if (scenario_id == 0 ||
        fire_.scenario_id() != scenario_id) {
        error = "scenario_id does not match the active scenario";
        return ScenarioOperationStatus::IdMismatch;
    }
    fire_.reset();
    return ScenarioOperationStatus::Ok;
}

void ScenarioManager::force_reset() {
    if (fire_.lifecycle() != ScenarioLifecycle::Empty) {
        fire_.reset();
    }
}

void ScenarioManager::tick() {
    fire_.tick();
}

void ScenarioManager::set_lockstep_frozen(bool frozen) {
    lockstep_frozen_ = frozen;
    fire_.set_lockstep_frozen(frozen);
}

ScenarioLifecycle ScenarioManager::lifecycle() const {
    return fire_.lifecycle();
}
