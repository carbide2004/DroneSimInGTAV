#pragma once

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <d3d11.h>
#include <dxgi.h>
#include <wrl/client.h>

enum class CaptureStatus : std::uint32_t {
    Ok = 0,
    Busy = 1,
    CaptureTimeout = 2,
    RgbFormatMismatch = 3,
    DepthFormatMismatch = 4,
    DepthTargetNotFound = 5,
    DepthTargetAmbiguous = 6,
    ResourceGenerationChanged = 7,
    InvalidCameraParameters = 8,
    DepthConversionFailed = 9,
    GpuReadbackFailed = 10,
    InternalError = 11,
};

const char* capture_status_name(CaptureStatus status);

struct CaptureCamera {
    float fov_degrees = 0.0f;
    float near_clip = 0.0f;
    float far_clip = 0.0f;
    // Row-major matrices. projection maps GTA camera/view coordinates to NDC.
    // view maps GTA world coordinates (X, Y, Z; Z up) to a RH camera frame
    // whose forward direction is -Z.
    std::array<float, 16> projection{};
    std::array<float, 16> view{};
};

struct CaptureResult {
    CaptureStatus status = CaptureStatus::InternalError;
    std::string error;
    std::uint64_t request_id = 0;
    std::uint64_t frame_id = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    CaptureCamera camera;
    std::vector<std::uint8_t> rgb;
    std::vector<float> depth_meters;
};

CaptureStatus build_capture_camera(
    float fov_degrees,
    float near_clip,
    float far_clip,
    float position_x,
    float position_y,
    float position_z,
    float rotation_x_degrees,
    float rotation_y_degrees,
    float rotation_z_degrees,
    CaptureCamera& output,
    std::string& error);

class RgbdCapture {
public:
    static RgbdCapture& instance();

    bool begin_request(
        std::uint64_t request_id,
        std::uint32_t timeout_ms,
        CaptureStatus& status,
        std::string& error);
    void submit_camera(std::uint64_t request_id, const CaptureCamera& camera);
    bool wait_result(
        std::uint64_t request_id,
        std::chrono::milliseconds timeout,
        CaptureResult& result);
    bool try_take_result(std::uint64_t request_id, CaptureResult& result);
    void cancel_request(std::uint64_t request_id);

    // Render-thread entry points.
    void observe_depth_target(ID3D11DepthStencilView* dsv);
    void on_present(IDXGISwapChain* swap_chain);

    std::uint64_t current_frame_id() const;

private:
    RgbdCapture();
    ~RgbdCapture() = default;
    RgbdCapture(const RgbdCapture&) = delete;
    RgbdCapture& operator=(const RgbdCapture&) = delete;

    struct Request;
    struct Slot;
    struct RawFrame;

    void poll_gpu(ID3D11Device* device, ID3D11DeviceContext* context);
    void schedule_capture(
        ID3D11Device* device,
        ID3D11DeviceContext* context,
        ID3D11Texture2D* back_buffer,
        const D3D11_TEXTURE2D_DESC& back_desc,
        std::vector<Microsoft::WRL::ComPtr<ID3D11Texture2D>> depth_candidates);
    void worker_main();
    void complete(CaptureResult result);
    void fail(std::uint64_t request_id, CaptureStatus status, const std::string& error);

    mutable std::mutex mutex_;
    std::condition_variable result_cv_;
    Request* request_;
    CaptureResult* completed_;
    std::uint64_t frame_id_;

    std::mutex render_mutex_;
    std::vector<Microsoft::WRL::ComPtr<ID3D11Texture2D>> depth_candidates_;
    std::array<Slot*, 3> slots_;
    std::size_t next_slot_;

    std::mutex worker_mutex_;
    std::condition_variable worker_cv_;
    std::deque<RawFrame*> worker_queue_;
    std::thread worker_;
};
