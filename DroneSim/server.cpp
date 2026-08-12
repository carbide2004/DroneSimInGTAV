#include "server.h"

#include "command_queue.h"
#include "logging.h"
#include "rgbd_capture.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <thread>
#include <utility>

namespace {

using boost::asio::ip::tcp;

constexpr std::size_t kWireHeaderSize = 20;
constexpr std::uint32_t kMaximumRequestPayload = 64 * 1024;
constexpr auto kCommandTimeout = std::chrono::milliseconds(7000);
constexpr auto kCaptureSubmitTimeout = std::chrono::milliseconds(2000);

std::unique_ptr<Server> g_server;
boost::asio::io_context g_io;
bool g_winsock_initialized = false;

template <typename T>
void append_scalar(
    std::vector<unsigned char>& output,
    const T& value) {
    const auto* bytes =
        reinterpret_cast<const unsigned char*>(&value);
    output.insert(output.end(), bytes, bytes + sizeof(T));
}

template <typename T>
bool read_scalar(
    const std::vector<unsigned char>& input,
    std::size_t offset,
    T& value) {
    if (offset > input.size() ||
        input.size() - offset < sizeof(T)) {
        return false;
    }
    std::memcpy(&value, input.data() + offset, sizeof(T));
    return true;
}

std::vector<unsigned char> make_response(
    std::uint8_t type,
    std::uint64_t request_id,
    const std::vector<unsigned char>& payload) {
    if (payload.size() >
        std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error(
            "Response payload exceeds the V3 wire limit");
    }

    MsgHeader header{};
    std::memcpy(header.magic, "DSV3", 4);
    header.version = 3;
    header.type = type;
    header.request_id = request_id;
    header.length =
        static_cast<std::uint32_t>(payload.size());

    std::vector<unsigned char> response(
        kWireHeaderSize + payload.size());
    std::memcpy(response.data(), header.magic, 4);
    std::memcpy(response.data() + 4, &header.version, 1);
    std::memcpy(response.data() + 5, &header.type, 1);
    std::memcpy(response.data() + 6, &header.flags, 1);
    std::memcpy(response.data() + 7, &header.reserved, 1);
    std::memcpy(
        response.data() + 8,
        &header.request_id,
        sizeof(header.request_id));
    std::memcpy(
        response.data() + 16,
        &header.length,
        sizeof(header.length));
    if (!payload.empty()) {
        std::memcpy(
            response.data() + kWireHeaderSize,
            payload.data(),
            payload.size());
    }
    return response;
}

std::vector<unsigned char> make_command_payload(
    const RuntimeCommandResult& result,
    const std::vector<unsigned char>& data = {}) {
    std::vector<unsigned char> payload;
    const auto status =
        static_cast<std::uint32_t>(result.status);
    const auto message_size =
        static_cast<std::uint32_t>(result.message.size());
    payload.reserve(
        sizeof(status) +
        sizeof(message_size) +
        result.message.size() +
        data.size());
    append_scalar(payload, status);
    append_scalar(payload, message_size);
    payload.insert(
        payload.end(),
        result.message.begin(),
        result.message.end());
    payload.insert(payload.end(), data.begin(), data.end());
    return payload;
}

std::vector<unsigned char> make_capture_error_payload(
    CaptureStatus status,
    const std::string& error) {
    std::vector<unsigned char> payload;
    const auto wire_status =
        static_cast<std::uint32_t>(status);
    const auto error_size =
        static_cast<std::uint32_t>(error.size());
    append_scalar(payload, wire_status);
    append_scalar(payload, error_size);
    payload.insert(payload.end(), error.begin(), error.end());
    return payload;
}

RuntimeCommandResult invalid_request(std::string message) {
    RuntimeCommandResult result;
    result.status = RuntimeCommandStatus::InvalidRequest;
    result.message = std::move(message);
    return result;
}

RuntimeCommandResult run_command(
    const RuntimeCommandPtr& command,
    std::chrono::milliseconds timeout = kCommandTimeout) {
    enqueue_command(command);
    RuntimeCommandResult result;
    if (!wait_for_command(command, timeout, result)) {
        result.status = RuntimeCommandStatus::CommandTimeout;
        result.message =
            "GTA script thread did not complete request " +
            std::to_string(command->request_id) +
            " before timeout";
    }
    return result;
}

std::vector<unsigned char> pose_bytes(const RuntimePose& pose) {
    std::vector<unsigned char> data;
    data.reserve(sizeof(float) * 6);
    append_scalar(data, pose.x);
    append_scalar(data, pose.y);
    append_scalar(data, pose.z);
    append_scalar(data, pose.pitch);
    append_scalar(data, pose.roll);
    append_scalar(data, pose.yaw);
    return data;
}

void append_vector3(
    std::vector<unsigned char>& data,
    const ScenarioVector3& value) {
    append_scalar(data, value.x);
    append_scalar(data, value.y);
    append_scalar(data, value.z);
}

std::vector<unsigned char> scenario_start_bytes(
    const ScenarioStartInfo& info) {
    std::vector<unsigned char> data;
    data.reserve(16);
    append_scalar(data, info.scenario_id);
    append_scalar(data, info.game_timer_ms);
    append_scalar(data, info.frame_count);
    return data;
}

std::vector<unsigned char> lockstep_snapshot_bytes(
    const LockstepSnapshot& snapshot) {
    std::vector<unsigned char> data;
    data.reserve(56);
    append_scalar(data, snapshot.session_id);
    append_scalar(data, snapshot.step_index);
    append_scalar(data, snapshot.epoch_game_timer_ms);
    append_scalar(data, snapshot.game_timer_ms);
    append_scalar(data, snapshot.frame_count);
    append_scalar(data, snapshot.target_elapsed_ms);
    append_scalar(data, snapshot.actual_elapsed_ms);
    append_scalar(data, snapshot.last_advance_ms);
    append_scalar(data, snapshot.render_frames);
    append_scalar(data, snapshot.max_frame_time_ms);
    return data;
}

std::vector<unsigned char> scenario_snapshot_bytes(
    const ScenarioSnapshot& snapshot) {
    std::vector<unsigned char> data;
    data.reserve(
        97 +
        snapshot.protected_entities.size() * 25 +
        snapshot.entities.size() * 101);
    append_scalar(data, snapshot.scenario_id);
    append_scalar(data, snapshot.blueprint_id);
    append_scalar(data, snapshot.seed);
    append_scalar(
        data,
        static_cast<std::uint32_t>(snapshot.lifecycle));
    append_scalar(data, snapshot.game_timer_ms);
    append_scalar(data, snapshot.frame_count);
    append_scalar(data, snapshot.start_game_timer_ms);
    append_scalar(data, snapshot.start_frame_count);
    append_vector3(data, snapshot.requested_anchor);
    append_vector3(data, snapshot.event_position);
    data.push_back(snapshot.event_active ? 1 : 0);
    append_scalar(data, snapshot.removed_pedestrians);
    append_scalar(data, snapshot.removed_vehicles);
    append_scalar(data, snapshot.ambient_pedestrians);
    append_scalar(data, snapshot.ambient_vehicles);
    append_scalar(
        data,
        static_cast<std::uint32_t>(
            snapshot.failure_message.size()));
    data.insert(
        data.end(),
        snapshot.failure_message.begin(),
        snapshot.failure_message.end());
    append_scalar(
        data,
        static_cast<std::uint32_t>(
            snapshot.protected_entities.size()));
    for (const ScenarioProtectedEntitySnapshot& entity :
         snapshot.protected_entities) {
        append_scalar(data, entity.gta_handle);
        append_scalar(data, entity.model_hash);
        append_scalar(
            data,
            static_cast<std::uint32_t>(entity.kind));
        data.push_back(entity.exists ? 1 : 0);
        append_vector3(data, entity.position);
    }
    append_scalar(
        data,
        static_cast<std::uint32_t>(snapshot.entities.size()));

    for (const ScenarioEntitySnapshot& entity : snapshot.entities) {
        append_scalar(data, entity.stable_id);
        append_scalar(data, entity.gta_handle);
        append_scalar(data, entity.model_hash);
        append_scalar(
            data,
            static_cast<std::uint32_t>(entity.kind));
        append_scalar(
            data,
            static_cast<std::uint32_t>(entity.role));
        append_scalar(data, entity.event_id);
        append_scalar(
            data,
            static_cast<std::uint32_t>(entity.task_state));
        data.push_back(entity.exists ? 1 : 0);
        append_vector3(data, entity.position);
        append_vector3(data, entity.velocity);
        append_scalar(data, entity.speed);
        append_scalar(data, entity.heading);
        append_scalar(data, entity.spawn_game_timer_ms);
        append_scalar(
            data,
            entity.planned_activation_offset_ms);
        append_scalar(
            data,
            entity.activation_game_timer_ms);
        append_scalar(data, entity.task_start_game_timer_ms);
        append_scalar(data, entity.response_start_game_timer_ms);
        append_vector3(data, entity.task_target);
    }
    return data;
}

std::vector<unsigned char> visibility_snapshot_bytes(
    const VisibilitySnapshot& snapshot) {
    std::vector<unsigned char> data;
    std::size_t sample_count = 0;
    for (const VisibilityTargetSnapshot& target : snapshot.targets) {
        sample_count += target.samples.size();
    }
    data.reserve(
        48 +
        snapshot.targets.size() * 24 +
        sample_count * 17);
    append_scalar(data, snapshot.scenario_id);
    append_scalar(data, snapshot.lockstep_session_id);
    append_scalar(data, snapshot.step_index);
    append_scalar(data, snapshot.game_timer_ms);
    append_scalar(data, snapshot.frame_count);
    append_vector3(data, snapshot.camera_center);
    append_scalar(
        data,
        static_cast<std::uint32_t>(snapshot.targets.size()));
    for (const VisibilityTargetSnapshot& target : snapshot.targets) {
        append_scalar(data, target.stable_id);
        append_scalar(data, target.gta_handle);
        append_scalar(
            data,
            static_cast<std::uint32_t>(target.role));
        append_scalar(
            data,
            static_cast<std::uint32_t>(target.samples.size()));
        for (const VisibilitySampleSnapshot& sample : target.samples) {
            append_vector3(data, sample.position);
            data.push_back(sample.clear_line_of_sight ? 1 : 0);
            append_scalar(data, sample.hit_entity);
        }
    }
    return data;
}

std::vector<unsigned char> geometry_batch_snapshot_bytes(
    const GeometryBatchSnapshot& snapshot) {
    std::vector<unsigned char> data;
    data.reserve(
        36 +
        snapshot.point_clear.size() +
        snapshot.segment_clear.size());
    append_scalar(data, snapshot.lockstep_session_id);
    append_scalar(data, snapshot.step_index);
    append_scalar(data, snapshot.game_timer_ms);
    append_scalar(data, snapshot.frame_count);
    append_scalar(
        data,
        static_cast<std::uint32_t>(snapshot.point_clear.size()));
    append_scalar(
        data,
        static_cast<std::uint32_t>(snapshot.segment_clear.size()));
    for (bool clear : snapshot.point_clear) {
        data.push_back(clear ? 1 : 0);
    }
    for (bool clear : snapshot.segment_clear) {
        data.push_back(clear ? 1 : 0);
    }
    return data;
}

std::vector<unsigned char> target_visibility_batch_snapshot_bytes(
    const TargetVisibilityBatchSnapshot& snapshot) {
    std::vector<unsigned char> data;
    std::size_t sample_count = 0;
    for (const TargetVisibilityCaseSnapshot& item : snapshot.cases) {
        sample_count += item.target.samples.size();
    }
    data.reserve(
        36 +
        snapshot.cases.size() * 36 +
        sample_count * 17);
    append_scalar(data, snapshot.scenario_id);
    append_scalar(data, snapshot.lockstep_session_id);
    append_scalar(data, snapshot.step_index);
    append_scalar(data, snapshot.game_timer_ms);
    append_scalar(data, snapshot.frame_count);
    append_scalar(
        data,
        static_cast<std::uint32_t>(snapshot.cases.size()));
    for (const TargetVisibilityCaseSnapshot& item : snapshot.cases) {
        append_scalar(data, item.stable_id);
        append_vector3(data, item.camera_center);
        append_scalar(data, item.target.gta_handle);
        append_scalar(
            data,
            static_cast<std::uint32_t>(item.target.role));
        append_scalar(
            data,
            static_cast<std::uint32_t>(
                item.target.samples.size()));
        for (const VisibilitySampleSnapshot& sample :
             item.target.samples) {
            append_vector3(data, sample.position);
            data.push_back(
                sample.clear_line_of_sight ? 1 : 0);
            append_scalar(data, sample.hit_entity);
        }
    }
    return data;
}

bool empty_payload(
    const std::vector<unsigned char>& payload,
    RuntimeCommandResult& error) {
    if (payload.empty()) {
        return true;
    }
    error = invalid_request("This command has no request payload");
    return false;
}

bool valid_weather_name(const std::string& name) {
    if (name.empty() || name.size() > 32) {
        return false;
    }
    for (const unsigned char value : name) {
        const bool valid =
            (value >= 'A' && value <= 'Z') ||
            (value >= '0' && value <= '9') ||
            value == '_';
        if (!valid) {
            return false;
        }
    }
    return true;
}

}  // namespace

Server::Server(
    boost::asio::io_context& io,
    unsigned short port)
    : acceptor_(io, tcp::endpoint(tcp::v4(), port)),
      socket_(io) {
    start_accept();
}

void Server::start_accept() {
    acceptor_.async_accept(
        socket_,
        [this](const boost::system::error_code& error) {
            if (error) {
                LOGE(
                    "server",
                    "Accept failed: " + error.message());
                start_accept();
                return;
            }
            handle_client();
        });
}

bool Server::read_exact(
    tcp::socket& socket,
    void* buffer,
    std::size_t length) {
    auto* bytes = static_cast<unsigned char*>(buffer);
    std::size_t total = 0;
    while (total < length) {
        boost::system::error_code error;
        const std::size_t read = socket.read_some(
            boost::asio::buffer(
                bytes + total,
                length - total),
            error);
        if (error) {
            return false;
        }
        total += read;
    }
    return true;
}

void Server::handle_client() {
    try {
        MsgHeader header{};
        if (!read_exact(socket_, header.magic, 4) ||
            !read_exact(socket_, &header.version, 1) ||
            !read_exact(socket_, &header.type, 1) ||
            !read_exact(socket_, &header.flags, 1) ||
            !read_exact(socket_, &header.reserved, 1) ||
            !read_exact(
                socket_,
                &header.request_id,
                sizeof(header.request_id)) ||
            !read_exact(
                socket_,
                &header.length,
                sizeof(header.length))) {
            throw std::runtime_error(
                "Client disconnected before the request header completed");
        }
        if (std::memcmp(header.magic, "DSV3", 4) != 0 ||
            header.version != 3) {
            throw std::runtime_error(
                "Expected DSV3 protocol version 3");
        }
        if (header.request_id == 0) {
            throw std::runtime_error(
                "request_id must be non-zero");
        }
        if (header.length > kMaximumRequestPayload) {
            const RuntimeCommandResult result = invalid_request(
                "Request payload exceeds 64 KiB");
            write_response(make_response(
                header.type,
                header.request_id,
                make_command_payload(result)));
            return;
        }

        std::vector<unsigned char> payload(header.length);
        if (!payload.empty() &&
            !read_exact(
                socket_,
                payload.data(),
                payload.size())) {
            throw std::runtime_error(
                "Client disconnected before the request payload completed");
        }

        RuntimeCommandResult result;
        std::vector<unsigned char> data;
        switch (header.type) {
            case MSG_CREATE_CAMERA: {
                if (empty_payload(payload, result)) {
                    result = run_command(make_runtime_command(
                        RuntimeCommandType::CreateCamera,
                        header.request_id));
                    if (result.status == RuntimeCommandStatus::Ok) {
                        append_scalar(data, result.value);
                    }
                }
                break;
            }
            case MSG_STOP_CAMERA: {
                if (empty_payload(payload, result)) {
                    result = run_command(make_runtime_command(
                        RuntimeCommandType::StopCamera,
                        header.request_id));
                }
                break;
            }
            case MSG_GET_CAMERA_STATE: {
                if (empty_payload(payload, result)) {
                    result = run_command(make_runtime_command(
                        RuntimeCommandType::GetCameraState,
                        header.request_id));
                    if (result.status == RuntimeCommandStatus::Ok) {
                        data.push_back(
                            result.bool_value ? 1 : 0);
                    }
                }
                break;
            }
            case MSG_GET_POSE: {
                if (empty_payload(payload, result)) {
                    result = run_command(make_runtime_command(
                        RuntimeCommandType::GetCameraPose,
                        header.request_id));
                    if (result.status == RuntimeCommandStatus::Ok) {
                        data = pose_bytes(result.pose);
                    }
                }
                break;
            }
            case MSG_SET_CAMERA_POSE: {
                if (payload.size() != sizeof(float) * 4 + 1) {
                    result = invalid_request(
                        "SET_CAMERA_POSE expects four float32 values "
                        "and one collision-check byte");
                    break;
                }
                auto command = make_runtime_command(
                    RuntimeCommandType::SetCameraPose,
                    header.request_id);
                for (std::size_t index = 0; index < 4; ++index) {
                    read_scalar(
                        payload,
                        index * sizeof(float),
                        command->floats[index]);
                }
                const unsigned char collision =
                    payload[sizeof(float) * 4];
                if (collision > 1) {
                    result = invalid_request(
                        "collision_check must be 0 or 1");
                    break;
                }
                command->flag = collision == 1;
                result = run_command(command);
                if (result.status == RuntimeCommandStatus::Ok) {
                    data = pose_bytes(result.pose);
                }
                break;
            }
            case MSG_SET_CAMERA_PITCH: {
                if (payload.size() != sizeof(float)) {
                    result = invalid_request(
                        "SET_CAMERA_PITCH expects one float32 value");
                    break;
                }
                auto command = make_runtime_command(
                    RuntimeCommandType::SetCameraPitch,
                    header.request_id);
                read_scalar(payload, 0, command->floats[0]);
                result = run_command(command);
                if (result.status == RuntimeCommandStatus::Ok) {
                    data = pose_bytes(result.pose);
                }
                break;
            }
            case MSG_SET_FOV: {
                if (payload.size() != sizeof(float)) {
                    result = invalid_request(
                        "SET_FOV expects one float32 value");
                    break;
                }
                auto command = make_runtime_command(
                    RuntimeCommandType::SetFov,
                    header.request_id);
                read_scalar(payload, 0, command->floats[0]);
                result = run_command(command);
                break;
            }
            case MSG_SET_TIME: {
                if (payload.size() != sizeof(std::int32_t) * 3) {
                    result = invalid_request(
                        "SET_TIME expects hour, minute, and second int32 values");
                    break;
                }
                auto command = make_runtime_command(
                    RuntimeCommandType::SetTime,
                    header.request_id);
                for (std::size_t index = 0; index < 3; ++index) {
                    read_scalar(
                        payload,
                        index * sizeof(std::int32_t),
                        command->integers[index]);
                }
                if (command->integers[0] < 0 ||
                    command->integers[0] > 23 ||
                    command->integers[1] < 0 ||
                    command->integers[1] > 59 ||
                    command->integers[2] < 0 ||
                    command->integers[2] > 59) {
                    result = invalid_request(
                        "Time must satisfy hour 0..23 and minute/second 0..59");
                    break;
                }
                result = run_command(command);
                break;
            }
            case MSG_SET_WEATHER: {
                const std::string name(
                    payload.begin(),
                    payload.end());
                if (!valid_weather_name(name)) {
                    result = invalid_request(
                        "Weather must be 1..32 uppercase ASCII letters, "
                        "digits, or underscores");
                    break;
                }
                auto command = make_runtime_command(
                    RuntimeCommandType::SetWeather,
                    header.request_id);
                command->text = name;
                result = run_command(command);
                break;
            }
            case MSG_TELEPORT_PLAYER: {
                if (payload.size() != sizeof(float) * 3) {
                    result = invalid_request(
                        "TELEPORT_PLAYER expects three float32 values");
                    break;
                }
                auto command = make_runtime_command(
                    RuntimeCommandType::TeleportPlayer,
                    header.request_id);
                for (std::size_t index = 0; index < 3; ++index) {
                    read_scalar(
                        payload,
                        index * sizeof(float),
                        command->floats[index]);
                }
                result = run_command(command);
                break;
            }
            case MSG_RESTORE_PLAYER: {
                if (empty_payload(payload, result)) {
                    result = run_command(make_runtime_command(
                        RuntimeCommandType::RestorePlayer,
                        header.request_id));
                }
                break;
            }
            case MSG_PREPARE_FIRE_SCENARIO: {
                constexpr std::size_t expected_size =
                    sizeof(float) * 3 +
                    sizeof(std::uint64_t) * 2 +
                    sizeof(std::uint16_t) * 2;
                if (payload.size() != expected_size) {
                    result = invalid_request(
                        "PREPARE_FIRE_SCENARIO expects anchor float32[3], "
                        "seed uint64, firetruck_count uint16, and "
                        "pedestrian_count uint16, and blueprint_id uint64");
                    break;
                }
                auto command = make_runtime_command(
                    RuntimeCommandType::PrepareFireScenario,
                    header.request_id);
                read_scalar(
                    payload,
                    0,
                    command->fire_scenario_config.anchor.x);
                read_scalar(
                    payload,
                    sizeof(float),
                    command->fire_scenario_config.anchor.y);
                read_scalar(
                    payload,
                    sizeof(float) * 2,
                    command->fire_scenario_config.anchor.z);
                read_scalar(
                    payload,
                    sizeof(float) * 3,
                    command->fire_scenario_config.seed);
                read_scalar(
                    payload,
                    sizeof(float) * 3 + sizeof(std::uint64_t),
                    command->fire_scenario_config.firetruck_count);
                read_scalar(
                    payload,
                    sizeof(float) * 3 + sizeof(std::uint64_t) +
                        sizeof(std::uint16_t),
                    command->fire_scenario_config.pedestrian_count);
                read_scalar(
                    payload,
                    sizeof(float) * 3 + sizeof(std::uint64_t) +
                        sizeof(std::uint16_t) * 2,
                    command->fire_scenario_config.blueprint_id);
                result = run_command(command);
                if (result.status == RuntimeCommandStatus::Ok) {
                    append_scalar(data, result.value);
                }
                break;
            }
            case MSG_GET_SCENARIO_STATE:
            case MSG_START_SCENARIO:
            case MSG_RESET_SCENARIO: {
                if (payload.size() != sizeof(std::uint64_t)) {
                    result = invalid_request(
                        "Scenario command expects one uint64 scenario_id");
                    break;
                }
                std::uint64_t scenario_id = 0;
                read_scalar(payload, 0, scenario_id);
                RuntimeCommandType command_type =
                    RuntimeCommandType::GetScenarioState;
                if (header.type == MSG_START_SCENARIO) {
                    command_type = RuntimeCommandType::StartScenario;
                } else if (header.type == MSG_RESET_SCENARIO) {
                    command_type = RuntimeCommandType::ResetScenario;
                }
                auto command = make_runtime_command(
                    command_type,
                    header.request_id);
                command->scenario_id = scenario_id;
                result = run_command(command);
                if (result.status == RuntimeCommandStatus::Ok) {
                    if (header.type == MSG_GET_SCENARIO_STATE) {
                        data = scenario_snapshot_bytes(
                            result.scenario_snapshot);
                    } else if (header.type == MSG_START_SCENARIO) {
                        data = scenario_start_bytes(
                            result.scenario_start);
                    }
                }
                break;
            }
            case MSG_ENTER_LOCKSTEP: {
                if (empty_payload(payload, result)) {
                    result = run_command(make_runtime_command(
                        RuntimeCommandType::EnterLockstep,
                        header.request_id));
                    if (result.status == RuntimeCommandStatus::Ok) {
                        data = lockstep_snapshot_bytes(
                            result.lockstep_snapshot);
                    }
                }
                break;
            }
            case MSG_GET_LOCKSTEP_STATE:
            case MSG_ADVANCE_LOCKSTEP:
            case MSG_EXIT_LOCKSTEP: {
                if (payload.size() != sizeof(std::uint64_t)) {
                    result = invalid_request(
                        "Lockstep command expects one uint64 session_id");
                    break;
                }
                std::uint64_t session_id = 0;
                read_scalar(payload, 0, session_id);
                RuntimeCommandType command_type =
                    RuntimeCommandType::GetLockstepState;
                if (header.type == MSG_ADVANCE_LOCKSTEP) {
                    command_type =
                        RuntimeCommandType::AdvanceLockstep;
                } else if (header.type == MSG_EXIT_LOCKSTEP) {
                    command_type =
                        RuntimeCommandType::ExitLockstep;
                }
                auto command = make_runtime_command(
                    command_type,
                    header.request_id);
                command->lockstep_session_id = session_id;
                result = run_command(command);
                if (result.status == RuntimeCommandStatus::Ok &&
                    header.type != MSG_EXIT_LOCKSTEP) {
                    data = lockstep_snapshot_bytes(
                        result.lockstep_snapshot);
                }
                break;
            }
            case MSG_QUERY_VISIBILITY: {
                constexpr std::size_t expected_size =
                    sizeof(std::uint64_t) * 2 +
                    sizeof(float) * 3;
                if (payload.size() != expected_size) {
                    result = invalid_request(
                        "QUERY_VISIBILITY expects scenario_id uint64, "
                        "session_id uint64, and camera_center float32[3]");
                    break;
                }
                auto command = make_runtime_command(
                    RuntimeCommandType::QueryVisibility,
                    header.request_id);
                read_scalar(payload, 0, command->scenario_id);
                read_scalar(
                    payload,
                    sizeof(std::uint64_t),
                    command->lockstep_session_id);
                read_scalar(
                    payload,
                    sizeof(std::uint64_t) * 2,
                    command->visibility_camera_center.x);
                read_scalar(
                    payload,
                    sizeof(std::uint64_t) * 2 + sizeof(float),
                    command->visibility_camera_center.y);
                read_scalar(
                    payload,
                    sizeof(std::uint64_t) * 2 + sizeof(float) * 2,
                    command->visibility_camera_center.z);
                result = run_command(
                    command,
                    std::chrono::milliseconds(30000));
                if (result.status == RuntimeCommandStatus::Ok) {
                    data = visibility_snapshot_bytes(
                        result.visibility_snapshot);
                }
                break;
            }
            case MSG_PROBE_CAMERA_START: {
                if (payload.size() != sizeof(float) * 3) {
                    result = invalid_request(
                        "PROBE_CAMERA_START expects X, Y, and altitude "
                        "AGL as float32");
                    break;
                }
                auto command = make_runtime_command(
                    RuntimeCommandType::ProbeCameraStart,
                    header.request_id);
                read_scalar(payload, 0, command->floats[0]);
                read_scalar(
                    payload,
                    sizeof(float),
                    command->floats[1]);
                read_scalar(
                    payload,
                    sizeof(float) * 2,
                    command->floats[2]);
                result = run_command(command);
                if (result.status == RuntimeCommandStatus::Ok) {
                    append_vector3(
                        data,
                        result.camera_start_probe.position);
                    append_scalar(
                        data,
                        result.camera_start_probe.ground_z);
                }
                break;
            }
            case MSG_PROBE_CAMERA_GEOMETRY_BATCH: {
                constexpr std::size_t fixed_size =
                    sizeof(std::uint64_t) +
                    sizeof(std::uint32_t) * 2;
                if (payload.size() < fixed_size) {
                    result = invalid_request(
                        "PROBE_CAMERA_GEOMETRY_BATCH payload is "
                        "shorter than its header");
                    break;
                }
                std::uint64_t session_id = 0;
                std::uint32_t point_count = 0;
                std::uint32_t segment_count = 0;
                read_scalar(payload, 0, session_id);
                read_scalar(
                    payload,
                    sizeof(std::uint64_t),
                    point_count);
                read_scalar(
                    payload,
                    sizeof(std::uint64_t) +
                        sizeof(std::uint32_t),
                    segment_count);
                const std::uint64_t total =
                    static_cast<std::uint64_t>(point_count) +
                    static_cast<std::uint64_t>(segment_count);
                const std::uint64_t expected_size =
                    fixed_size +
                    static_cast<std::uint64_t>(point_count) *
                        sizeof(float) * 3 +
                    static_cast<std::uint64_t>(segment_count) *
                        sizeof(float) * 6;
                if (total == 0 ||
                    total > 256 ||
                    expected_size != payload.size()) {
                    result = invalid_request(
                        "PROBE_CAMERA_GEOMETRY_BATCH expects 1..256 "
                        "declared points and segments");
                    break;
                }
                auto command = make_runtime_command(
                    RuntimeCommandType::ProbeCameraGeometryBatch,
                    header.request_id);
                command->lockstep_session_id = session_id;
                command->geometry_points.reserve(point_count);
                command->geometry_segments.reserve(segment_count);
                std::size_t offset = fixed_size;
                for (std::uint32_t index = 0;
                     index < point_count;
                     ++index) {
                    ScenarioVector3 point;
                    read_scalar(payload, offset, point.x);
                    read_scalar(
                        payload,
                        offset + sizeof(float),
                        point.y);
                    read_scalar(
                        payload,
                        offset + sizeof(float) * 2,
                        point.z);
                    offset += sizeof(float) * 3;
                    command->geometry_points.push_back(point);
                }
                for (std::uint32_t index = 0;
                     index < segment_count;
                     ++index) {
                    GeometrySegment segment;
                    read_scalar(payload, offset, segment.start.x);
                    read_scalar(
                        payload,
                        offset + sizeof(float),
                        segment.start.y);
                    read_scalar(
                        payload,
                        offset + sizeof(float) * 2,
                        segment.start.z);
                    read_scalar(
                        payload,
                        offset + sizeof(float) * 3,
                        segment.end.x);
                    read_scalar(
                        payload,
                        offset + sizeof(float) * 4,
                        segment.end.y);
                    read_scalar(
                        payload,
                        offset + sizeof(float) * 5,
                        segment.end.z);
                    offset += sizeof(float) * 6;
                    command->geometry_segments.push_back(segment);
                }
                result = run_command(
                    command,
                    std::chrono::milliseconds(30000));
                if (result.status == RuntimeCommandStatus::Ok) {
                    data = geometry_batch_snapshot_bytes(
                        result.geometry_batch_snapshot);
                }
                break;
            }
            case MSG_QUERY_TARGET_VISIBILITY_BATCH: {
                constexpr std::size_t fixed_size =
                    sizeof(std::uint64_t) * 2 +
                    sizeof(std::uint32_t);
                constexpr std::size_t case_size =
                    sizeof(std::uint64_t) +
                    sizeof(float) * 3;
                if (payload.size() < fixed_size) {
                    result = invalid_request(
                        "QUERY_TARGET_VISIBILITY_BATCH payload is "
                        "shorter than its header");
                    break;
                }
                std::uint64_t scenario_id = 0;
                std::uint64_t session_id = 0;
                std::uint32_t case_count = 0;
                read_scalar(payload, 0, scenario_id);
                read_scalar(
                    payload,
                    sizeof(std::uint64_t),
                    session_id);
                read_scalar(
                    payload,
                    sizeof(std::uint64_t) * 2,
                    case_count);
                const std::uint64_t expected_size =
                    fixed_size +
                    static_cast<std::uint64_t>(case_count) *
                        case_size;
                if (case_count == 0 ||
                    case_count > 64 ||
                    expected_size != payload.size()) {
                    result = invalid_request(
                        "QUERY_TARGET_VISIBILITY_BATCH expects 1..64 "
                        "declared target-pose cases");
                    break;
                }
                auto command = make_runtime_command(
                    RuntimeCommandType::QueryTargetVisibilityBatch,
                    header.request_id);
                command->scenario_id = scenario_id;
                command->lockstep_session_id = session_id;
                command->target_visibility_cases.reserve(case_count);
                std::size_t offset = fixed_size;
                for (std::uint32_t index = 0;
                     index < case_count;
                     ++index) {
                    TargetVisibilityCase item;
                    read_scalar(payload, offset, item.stable_id);
                    read_scalar(
                        payload,
                        offset + sizeof(std::uint64_t),
                        item.camera_center.x);
                    read_scalar(
                        payload,
                        offset + sizeof(std::uint64_t) +
                            sizeof(float),
                        item.camera_center.y);
                    read_scalar(
                        payload,
                        offset + sizeof(std::uint64_t) +
                            sizeof(float) * 2,
                        item.camera_center.z);
                    offset += case_size;
                    command->target_visibility_cases.push_back(item);
                }
                result = run_command(
                    command,
                    std::chrono::milliseconds(30000));
                if (result.status == RuntimeCommandStatus::Ok) {
                    data = target_visibility_batch_snapshot_bytes(
                        result.target_visibility_batch_snapshot);
                }
                break;
            }
            case MSG_PING: {
                result.status = RuntimeCommandStatus::Ok;
                data = payload;
                break;
            }
            case MSG_CAPTURE: {
                std::uint32_t timeout_ms = 5000;
                if (payload.size() == sizeof(timeout_ms)) {
                    read_scalar(payload, 0, timeout_ms);
                } else if (!payload.empty()) {
                    write_response(make_response(
                        MSG_CAPTURE,
                        header.request_id,
                        make_capture_error_payload(
                            CaptureStatus::InternalError,
                            "Capture payload must contain one uint32 timeout_ms")));
                    return;
                }
                if (timeout_ms == 0 || timeout_ms > 60000) {
                    write_response(make_response(
                        MSG_CAPTURE,
                        header.request_id,
                        make_capture_error_payload(
                            CaptureStatus::InternalError,
                            "timeout_ms must be in [1, 60000]")));
                    return;
                }

                CaptureStatus begin_status = CaptureStatus::Ok;
                std::string begin_error;
                if (!RgbdCapture::instance().begin_request(
                        header.request_id,
                        timeout_ms,
                        begin_status,
                        begin_error)) {
                    write_response(make_response(
                        MSG_CAPTURE,
                        header.request_id,
                        make_capture_error_payload(
                            begin_status,
                            begin_error)));
                    return;
                }

                const RuntimeCommandResult submit_result =
                    run_command(
                        make_runtime_command(
                            RuntimeCommandType::Capture,
                            header.request_id),
                        kCaptureSubmitTimeout);
                if (submit_result.status != RuntimeCommandStatus::Ok) {
                    RgbdCapture::instance().cancel_request(
                        header.request_id);
                    write_response(make_response(
                        MSG_CAPTURE,
                        header.request_id,
                        make_capture_error_payload(
                            submit_result.status ==
                                    RuntimeCommandStatus::CommandTimeout
                                ? CaptureStatus::CaptureTimeout
                                : CaptureStatus::InvalidCameraParameters,
                            submit_result.message)));
                    return;
                }

                CaptureResult capture;
                if (!RgbdCapture::instance().wait_result(
                        header.request_id,
                        std::chrono::milliseconds(timeout_ms) +
                            std::chrono::milliseconds(250),
                        capture)) {
                    RgbdCapture::instance().cancel_request(
                        header.request_id);
                    write_response(make_response(
                        MSG_CAPTURE,
                        header.request_id,
                        make_capture_error_payload(
                            CaptureStatus::CaptureTimeout,
                            "No fresh RGB-D frame completed before timeout")));
                    return;
                }
                if (capture.status != CaptureStatus::Ok) {
                    write_response(make_response(
                        MSG_CAPTURE,
                        header.request_id,
                        make_capture_error_payload(
                            capture.status,
                            capture.error)));
                    return;
                }

                const auto rgb_size =
                    static_cast<std::uint32_t>(capture.rgb.size());
                const auto depth_size =
                    static_cast<std::uint32_t>(
                        capture.depth_meters.size() * sizeof(float));
                std::vector<unsigned char> capture_payload;
                capture_payload.reserve(
                    168ULL + rgb_size + depth_size);
                append_scalar(
                    capture_payload,
                    static_cast<std::uint32_t>(CaptureStatus::Ok));
                append_scalar(capture_payload, capture.frame_id);
                append_scalar(capture_payload, capture.width);
                append_scalar(capture_payload, capture.height);
                append_scalar(capture_payload, rgb_size);
                append_scalar(capture_payload, depth_size);
                append_scalar(
                    capture_payload,
                    capture.camera.fov_degrees);
                append_scalar(
                    capture_payload,
                    capture.camera.near_clip);
                append_scalar(
                    capture_payload,
                    capture.camera.far_clip);
                for (const float value :
                     capture.camera.projection) {
                    append_scalar(capture_payload, value);
                }
                for (const float value : capture.camera.view) {
                    append_scalar(capture_payload, value);
                }
                capture_payload.insert(
                    capture_payload.end(),
                    capture.rgb.begin(),
                    capture.rgb.end());
                const auto* depth_bytes =
                    reinterpret_cast<const unsigned char*>(
                        capture.depth_meters.data());
                capture_payload.insert(
                    capture_payload.end(),
                    depth_bytes,
                    depth_bytes + depth_size);
                write_response(make_response(
                    MSG_CAPTURE,
                    header.request_id,
                    capture_payload));
                return;
            }
            default:
                result = invalid_request(
                    "Unknown message type " +
                    std::to_string(header.type));
                break;
        }

        write_response(make_response(
            header.type,
            header.request_id,
            make_command_payload(result, data)));
    } catch (const std::exception& exception) {
        LOGE(
            "server",
            "Request failed: " + std::string(exception.what()));
        boost::system::error_code ignored;
        socket_.close(ignored);
        start_accept();
    }
}

void Server::write_response(
    const std::vector<unsigned char>& data) {
    boost::system::error_code error;
    boost::asio::write(
        socket_,
        boost::asio::buffer(data),
        error);
    if (error) {
        LOGE(
            "server",
            "Response write failed: " + error.message());
    }
    boost::system::error_code ignored;
    socket_.shutdown(tcp::socket::shutdown_both, ignored);
    socket_.close(ignored);
    start_accept();
}

void InitializeServer() {
    if (!g_winsock_initialized) {
        WSADATA data{};
        const int status =
            WSAStartup(MAKEWORD(2, 2), &data);
        if (status != 0) {
            LOGE(
                "server",
                "WSAStartup failed with code " +
                    std::to_string(status));
            return;
        }
        g_winsock_initialized = true;
    }
    if (g_server) {
        return;
    }

    try {
        g_io.restart();
        g_server =
            std::make_unique<Server>(g_io, 23456);
        std::thread([]() { g_io.run(); }).detach();
        LOGI(
            "server",
            "DSV3 server listening on 0.0.0.0:23456");
    } catch (const std::exception& exception) {
        LOGE(
            "server",
            "Server initialization failed: " +
                std::string(exception.what()));
        g_server.reset();
    }
}

void ShutdownServer() {
    g_io.stop();
    g_server.reset();
    if (g_winsock_initialized) {
        WSACleanup();
        g_winsock_initialized = false;
    }
}
