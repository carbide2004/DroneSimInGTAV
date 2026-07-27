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
constexpr auto kCommandTimeout = std::chrono::milliseconds(5000);
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
