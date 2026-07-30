#include "simulation_clock.h"

#include "camera.h"
#include "keyboard.h"
#include "logging.h"
#include "main.h"
#include "natives.h"
#include "scenario_manager.h"

#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <utility>

namespace {

constexpr std::uint64_t kStepMilliseconds = 250;
constexpr auto kAdvanceWallTimeout = std::chrono::seconds(5);

std::uint32_t game_timer_ms() {
    return static_cast<std::uint32_t>(
        GAMEPLAY::GET_GAME_TIMER());
}

std::uint32_t frame_count() {
    return static_cast<std::uint32_t>(
        GAMEPLAY::GET_FRAME_COUNT());
}

std::uint32_t elapsed_ms(
    std::uint32_t epoch,
    std::uint32_t current) {
    return current - epoch;
}

class ScopeExit {
public:
    explicit ScopeExit(std::function<void()> callback)
        : callback_(std::move(callback)) {}

    ~ScopeExit() {
        if (active_) {
            callback_();
        }
    }

    void dismiss() {
        active_ = false;
    }

private:
    std::function<void()> callback_;
    bool active_ = true;
};

}  // namespace

SimulationClock& SimulationClock::instance() {
    static SimulationClock clock;
    return clock;
}

void SimulationClock::freeze_world() {
    GAMEPLAY::SET_TIME_SCALE(0.0f);
    TIME::PAUSE_CLOCK(TRUE);
    ScenarioManager::instance().set_lockstep_frozen(true);
    phase_ = LockstepPhase::Frozen;
}

void SimulationClock::restore_realtime() {
    ScenarioManager::instance().set_lockstep_frozen(false);
    GAMEPLAY::SET_TIME_SCALE(1.0f);
    TIME::PAUSE_CLOCK(FALSE);
    phase_ = LockstepPhase::Inactive;
}

void SimulationClock::clear_session() {
    session_id_ = 0;
    step_index_ = 0;
    epoch_game_timer_ms_ = 0;
    actual_elapsed_ms_ = 0;
    last_advance_ms_ = 0;
    render_frames_ = 0;
    max_frame_time_ms_ = 0.0f;
}

void SimulationClock::fill_snapshot(
    LockstepSnapshot& output) const {
    output.session_id = session_id_;
    output.step_index = step_index_;
    output.epoch_game_timer_ms = epoch_game_timer_ms_;
    output.game_timer_ms = game_timer_ms();
    output.frame_count = frame_count();
    output.target_elapsed_ms =
        step_index_ * kStepMilliseconds;
    output.actual_elapsed_ms = actual_elapsed_ms_;
    output.last_advance_ms = last_advance_ms_;
    output.render_frames = render_frames_;
    output.max_frame_time_ms = max_frame_time_ms_;
}

LockstepOperationStatus SimulationClock::validate_session(
    std::uint64_t session_id,
    std::string& error) const {
    if (phase_ == LockstepPhase::Inactive) {
        error = "No lockstep session is active";
        return LockstepOperationStatus::NotActive;
    }
    if (session_id == 0 || session_id != session_id_) {
        error = "session_id does not own the active lockstep session";
        return LockstepOperationStatus::SessionMismatch;
    }
    return LockstepOperationStatus::Ok;
}

LockstepOperationStatus SimulationClock::enter(
    LockstepSnapshot& output,
    std::string& error) {
    if (phase_ != LockstepPhase::Inactive) {
        error = "A lockstep session is already active";
        return LockstepOperationStatus::AlreadyActive;
    }
    if (!CameraController::instance().is_active()) {
        error = "Lockstep requires an active scripted camera";
        return LockstepOperationStatus::InvariantFailed;
    }
    const ScenarioLifecycle lifecycle =
        ScenarioManager::instance().lifecycle();
    if (lifecycle != ScenarioLifecycle::Empty &&
        lifecycle != ScenarioLifecycle::Ready) {
        error =
            "Lockstep Enter requires the scenario lifecycle to be "
            "EMPTY or READY";
        return LockstepOperationStatus::InvariantFailed;
    }

    const std::uint32_t before = game_timer_ms();
    freeze_world();
    CameraController::instance()
        .suppress_player_controls_for_frame();
    WAIT(0);
    const std::uint32_t after = game_timer_ms();
    if (after != before) {
        restore_realtime();
        clear_session();
        error =
            "GET_GAME_TIMER advanced while time scale was zero";
        return LockstepOperationStatus::InvariantFailed;
    }
    if (F11.consume_press()) {
        restore_realtime();
        clear_session();
        emergency_recovery_requested_ = true;
        error = "F11 interrupted lockstep Enter";
        return LockstepOperationStatus::Interrupted;
    }
    if (!CameraController::instance().is_active()) {
        restore_realtime();
        clear_session();
        error =
            "The scripted camera became inactive during lockstep Enter";
        return LockstepOperationStatus::InvariantFailed;
    }

    session_id_ = next_session_id_++;
    if (session_id_ == 0) {
        session_id_ = next_session_id_++;
    }
    step_index_ = 0;
    epoch_game_timer_ms_ = after;
    actual_elapsed_ms_ = 0;
    last_advance_ms_ = 0;
    render_frames_ = 0;
    max_frame_time_ms_ = 0.0f;
    fill_snapshot(output);
    LOGI(
        "lockstep",
        "Entered lockstep session " +
            std::to_string(session_id_));
    return LockstepOperationStatus::Ok;
}

LockstepOperationStatus SimulationClock::snapshot(
    std::uint64_t session_id,
    LockstepSnapshot& output,
    std::string& error) const {
    const LockstepOperationStatus status =
        validate_session(session_id, error);
    if (status != LockstepOperationStatus::Ok) {
        return status;
    }
    if (phase_ != LockstepPhase::Frozen) {
        error = "Lockstep state can only be queried while frozen";
        return LockstepOperationStatus::InvariantFailed;
    }
    const std::uint64_t observed_elapsed =
        elapsed_ms(epoch_game_timer_ms_, game_timer_ms());
    if (observed_elapsed != actual_elapsed_ms_) {
        error =
            "GET_GAME_TIMER changed while lockstep was frozen";
        return LockstepOperationStatus::InvariantFailed;
    }
    fill_snapshot(output);
    return LockstepOperationStatus::Ok;
}

LockstepOperationStatus SimulationClock::advance(
    std::uint64_t session_id,
    const std::atomic<bool>& cancelled,
    LockstepSnapshot& output,
    std::string& error) {
    const LockstepOperationStatus validation =
        validate_session(session_id, error);
    if (validation != LockstepOperationStatus::Ok) {
        return validation;
    }
    if (phase_ != LockstepPhase::Frozen) {
        error = "Lockstep Advance requires the FROZEN state";
        return LockstepOperationStatus::InvariantFailed;
    }
    const std::uint64_t observed_before =
        elapsed_ms(epoch_game_timer_ms_, game_timer_ms());
    if (observed_before != actual_elapsed_ms_) {
        error =
            "GET_GAME_TIMER changed before lockstep Advance";
        return LockstepOperationStatus::InvariantFailed;
    }
    if (step_index_ >=
        (std::numeric_limits<std::uint32_t>::max)() /
            kStepMilliseconds) {
        error = "Lockstep elapsed time exceeds the uint32 timer window";
        return LockstepOperationStatus::InvariantFailed;
    }

    const std::uint64_t target_elapsed =
        (step_index_ + 1) * kStepMilliseconds;
    const std::uint64_t previous_elapsed = actual_elapsed_ms_;
    const std::uint32_t start_frame = frame_count();
    float maximum_frame_time_ms = 0.0f;
    phase_ = LockstepPhase::Advancing;
    ScenarioManager::instance().set_lockstep_frozen(false);
    GAMEPLAY::SET_TIME_SCALE(1.0f);
    TIME::PAUSE_CLOCK(TRUE);

    ScopeExit guard([this]() { freeze_world(); });

    const auto deadline =
        std::chrono::steady_clock::now() +
        kAdvanceWallTimeout;
    std::uint64_t observed_elapsed = previous_elapsed;
    while (observed_elapsed < target_elapsed) {
        if (F11.consume_press()) {
            emergency_recovery_requested_ = true;
            error = "F11 interrupted lockstep Advance";
            return LockstepOperationStatus::Interrupted;
        }
        if (cancelled.load(std::memory_order_acquire) ||
            std::chrono::steady_clock::now() >= deadline) {
            error =
                "Lockstep Advance did not reach its 250ms target "
                "within five wall-clock seconds";
            return LockstepOperationStatus::AdvanceTimeout;
        }

        CameraController::instance()
            .suppress_player_controls_for_frame();
        ScenarioManager::instance().tick();
        WAIT(0);

        const float frame_time_ms =
            GAMEPLAY::GET_FRAME_TIME() * 1000.0f;
        if (!std::isfinite(frame_time_ms) ||
            frame_time_ms < 0.0f) {
            error =
                "GTA returned an invalid frame time during Advance";
            return LockstepOperationStatus::InvariantFailed;
        }
        if (frame_time_ms > maximum_frame_time_ms) {
            maximum_frame_time_ms = frame_time_ms;
        }
        observed_elapsed =
            elapsed_ms(epoch_game_timer_ms_, game_timer_ms());
    }

    freeze_world();
    guard.dismiss();
    actual_elapsed_ms_ = observed_elapsed;
    ++step_index_;
    const std::uint64_t last_advance =
        actual_elapsed_ms_ - previous_elapsed;
    if (last_advance >
        (std::numeric_limits<std::uint32_t>::max)()) {
        error = "Lockstep Advance duration overflowed uint32";
        return LockstepOperationStatus::InvariantFailed;
    }
    last_advance_ms_ =
        static_cast<std::uint32_t>(last_advance);
    render_frames_ = frame_count() - start_frame;
    max_frame_time_ms_ = maximum_frame_time_ms;
    fill_snapshot(output);
    return LockstepOperationStatus::Ok;
}

LockstepOperationStatus SimulationClock::exit(
    std::uint64_t session_id,
    std::string& error) {
    const LockstepOperationStatus validation =
        validate_session(session_id, error);
    if (validation != LockstepOperationStatus::Ok) {
        return validation;
    }
    if (phase_ != LockstepPhase::Frozen) {
        error = "Lockstep Exit requires the FROZEN state";
        return LockstepOperationStatus::InvariantFailed;
    }
    if (ScenarioManager::instance().lifecycle() !=
        ScenarioLifecycle::Empty) {
        error =
            "Reset the active scenario while frozen before "
            "exiting lockstep";
        return LockstepOperationStatus::InvariantFailed;
    }
    LOGI(
        "lockstep",
        "Exited lockstep session " +
            std::to_string(session_id_));
    restore_realtime();
    clear_session();
    return LockstepOperationStatus::Ok;
}

bool SimulationClock::is_active() const {
    return phase_ != LockstepPhase::Inactive;
}

void SimulationClock::request_emergency_recovery() {
    emergency_recovery_requested_ = true;
}

bool SimulationClock::emergency_recovery_requested() const {
    return emergency_recovery_requested_;
}

bool SimulationClock::take_emergency_recovery_request() {
    const bool requested = emergency_recovery_requested_;
    emergency_recovery_requested_ = false;
    return requested;
}

void SimulationClock::force_exit() {
    restore_realtime();
    clear_session();
}
