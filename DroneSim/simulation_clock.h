#pragma once

#include <atomic>
#include <cstdint>
#include <string>

enum class LockstepPhase : std::uint32_t {
    Inactive = 0,
    Frozen = 1,
    Advancing = 2,
};

enum class LockstepOperationStatus {
    Ok,
    AlreadyActive,
    NotActive,
    SessionMismatch,
    AdvanceTimeout,
    Interrupted,
    InvariantFailed,
};

struct LockstepSnapshot {
    std::uint64_t session_id = 0;
    std::uint64_t step_index = 0;
    std::uint32_t epoch_game_timer_ms = 0;
    std::uint32_t game_timer_ms = 0;
    std::uint32_t frame_count = 0;
    std::uint64_t target_elapsed_ms = 0;
    std::uint64_t actual_elapsed_ms = 0;
    std::uint32_t last_advance_ms = 0;
    std::uint32_t render_frames = 0;
    float max_frame_time_ms = 0.0f;
};

class SimulationClock {
public:
    static SimulationClock& instance();

    LockstepOperationStatus enter(
        LockstepSnapshot& output,
        std::string& error);
    LockstepOperationStatus snapshot(
        std::uint64_t session_id,
        LockstepSnapshot& output,
        std::string& error) const;
    LockstepOperationStatus advance(
        std::uint64_t session_id,
        const std::atomic<bool>& cancelled,
        LockstepSnapshot& output,
        std::string& error);
    LockstepOperationStatus exit(
        std::uint64_t session_id,
        std::string& error);

    bool is_active() const;
    bool take_emergency_recovery_request();
    void force_exit();

private:
    SimulationClock() = default;

    void freeze_world();
    void restore_realtime();
    void clear_session();
    void fill_snapshot(LockstepSnapshot& output) const;
    LockstepOperationStatus validate_session(
        std::uint64_t session_id,
        std::string& error) const;

    LockstepPhase phase_ = LockstepPhase::Inactive;
    std::uint64_t session_id_ = 0;
    std::uint64_t next_session_id_ = 1;
    std::uint64_t step_index_ = 0;
    std::uint32_t epoch_game_timer_ms_ = 0;
    std::uint64_t actual_elapsed_ms_ = 0;
    std::uint32_t last_advance_ms_ = 0;
    std::uint32_t render_frames_ = 0;
    float max_frame_time_ms_ = 0.0f;
    bool emergency_recovery_requested_ = false;
};
