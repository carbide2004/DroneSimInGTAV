#include "rgbd_capture.h"

#include "logging.h"

#include <Eigen/Core>
#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>

using Microsoft::WRL::ComPtr;

namespace {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kDepthTolerance = 1.0e-5f;
constexpr float kHomogeneousEpsilon = 1.0e-8f;

enum class RequestStage {
    WaitingCamera,
    WaitingFrame,
    GpuPending,
    WorkerPending,
};

enum class SlotStage {
    Free,
    GpuPending,
};

std::string hresult_text(const char* operation, HRESULT hr) {
    std::ostringstream stream;
    stream << operation << " failed (HRESULT=0x" << std::hex
           << static_cast<unsigned long>(hr) << ")";
    return stream.str();
}

bool finite_matrix(const std::array<float, 16>& matrix) {
    return std::all_of(matrix.begin(), matrix.end(), [](float value) {
        return std::isfinite(value);
    });
}

void create_staging_texture(
    ID3D11Device* device,
    const D3D11_TEXTURE2D_DESC& source,
    ComPtr<ID3D11Texture2D>& output) {
    D3D11_TEXTURE2D_DESC desc = source;
    desc.BindFlags = 0;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    desc.MiscFlags = 0;
    desc.Usage = D3D11_USAGE_STAGING;
    desc.SampleDesc.Count = 1;
    desc.SampleDesc.Quality = 0;

    HRESULT hr = device->CreateTexture2D(&desc, nullptr, output.ReleaseAndGetAddressOf());
    if (FAILED(hr) || output == nullptr) {
        throw std::runtime_error(hresult_text("CreateTexture2D", hr));
    }
}

void create_event_query(ID3D11Device* device, ComPtr<ID3D11Query>& output) {
    D3D11_QUERY_DESC desc{};
    desc.Query = D3D11_QUERY_EVENT;
    HRESULT hr = device->CreateQuery(&desc, output.ReleaseAndGetAddressOf());
    if (FAILED(hr) || output == nullptr) {
        throw std::runtime_error(hresult_text("CreateQuery", hr));
    }
}

struct MappedTexture {
    ID3D11DeviceContext* context = nullptr;
    ID3D11Resource* resource = nullptr;
    D3D11_MAPPED_SUBRESOURCE mapped{};

    MappedTexture(ID3D11DeviceContext* ctx, ID3D11Resource* res)
        : context(ctx), resource(res) {
        HRESULT hr = context->Map(
            resource,
            0,
            D3D11_MAP_READ,
            D3D11_MAP_FLAG_DO_NOT_WAIT,
            &mapped);
        if (FAILED(hr)) {
            context = nullptr;
            resource = nullptr;
            throw std::runtime_error(hresult_text("Map", hr));
        }
    }

    ~MappedTexture() {
        if (context != nullptr && resource != nullptr) {
            context->Unmap(resource, 0);
        }
    }

    MappedTexture(const MappedTexture&) = delete;
    MappedTexture& operator=(const MappedTexture&) = delete;
};

template <typename T>
void copy_rows(
    const D3D11_MAPPED_SUBRESOURCE& source,
    std::uint32_t height,
    std::size_t bytes_per_row,
    std::vector<T>& destination) {
    const std::size_t total_bytes = bytes_per_row * static_cast<std::size_t>(height);
    if (total_bytes % sizeof(T) != 0) {
        throw std::runtime_error("Mapped texture byte count is not element-aligned");
    }
    destination.resize(total_bytes / sizeof(T));
    auto* destination_bytes = reinterpret_cast<std::uint8_t*>(destination.data());
    const auto* source_bytes = static_cast<const std::uint8_t*>(source.pData);
    for (std::uint32_t row = 0; row < height; ++row) {
        std::memcpy(
            destination_bytes + static_cast<std::size_t>(row) * bytes_per_row,
            source_bytes + static_cast<std::size_t>(row) * source.RowPitch,
            bytes_per_row);
    }
}

}  // namespace

struct RgbdCapture::Request {
    std::uint64_t id = 0;
    std::uint64_t minimum_frame = 0;
    std::chrono::steady_clock::time_point deadline;
    RequestStage stage = RequestStage::WaitingCamera;
    CaptureCamera camera;
};

struct RgbdCapture::Slot {
    SlotStage stage = SlotStage::Free;
    std::uint64_t request_id = 0;
    std::uint64_t frame_id = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    CaptureCamera camera;
    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11Texture2D> color;
    ComPtr<ID3D11Texture2D> depth;
    ComPtr<ID3D11Query> completion;

    void release_frame() {
        stage = SlotStage::Free;
        request_id = 0;
        frame_id = 0;
        width = 0;
        height = 0;
        camera = {};
    }

    void reset_resources() {
        release_frame();
        color.Reset();
        depth.Reset();
        completion.Reset();
        device.Reset();
    }
};

struct RgbdCapture::RawFrame {
    std::uint64_t request_id = 0;
    std::uint64_t frame_id = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    CaptureCamera camera;
    std::vector<std::uint8_t> bgra;
    std::vector<std::uint8_t> packed_depth;
};

const char* capture_status_name(CaptureStatus status) {
    switch (status) {
        case CaptureStatus::Ok: return "OK";
        case CaptureStatus::Busy: return "CAPTURE_BUSY";
        case CaptureStatus::CaptureTimeout: return "CAPTURE_TIMEOUT";
        case CaptureStatus::RgbFormatMismatch: return "RGB_FORMAT_MISMATCH";
        case CaptureStatus::DepthFormatMismatch: return "DEPTH_FORMAT_MISMATCH";
        case CaptureStatus::DepthTargetNotFound: return "DEPTH_TARGET_NOT_FOUND";
        case CaptureStatus::DepthTargetAmbiguous: return "DEPTH_TARGET_AMBIGUOUS";
        case CaptureStatus::ResourceGenerationChanged: return "RESOURCE_GENERATION_CHANGED";
        case CaptureStatus::InvalidCameraParameters: return "INVALID_CAMERA_PARAMETERS";
        case CaptureStatus::DepthConversionFailed: return "DEPTH_CONVERSION_FAILED";
        case CaptureStatus::GpuReadbackFailed: return "GPU_READBACK_FAILED";
        case CaptureStatus::InternalError: return "INTERNAL_ERROR";
        default: return "UNKNOWN_CAPTURE_STATUS";
    }
}

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
    std::string& error) {
    const float values[] = {
        fov_degrees,
        near_clip,
        far_clip,
        position_x,
        position_y,
        position_z,
        rotation_x_degrees,
        rotation_y_degrees,
        rotation_z_degrees,
    };
    for (float value : values) {
        if (!std::isfinite(value)) {
            error = "Camera parameters contain a non-finite value";
            return CaptureStatus::InvalidCameraParameters;
        }
    }
    if (fov_degrees <= 0.0f || fov_degrees >= 180.0f) {
        error = "Camera FOV must be in the open interval (0, 180)";
        return CaptureStatus::InvalidCameraParameters;
    }
    if (near_clip <= 0.0f || far_clip <= near_clip) {
        error = "Camera clip planes must satisfy 0 < near < far";
        return CaptureStatus::InvalidCameraParameters;
    }

    output = {};
    output.fov_degrees = fov_degrees;
    output.near_clip = near_clip;
    output.far_clip = far_clip;

    const float pitch = rotation_x_degrees * kPi / 180.0f;
    const float roll = rotation_y_degrees * kPi / 180.0f;
    const float yaw = rotation_z_degrees * kPi / 180.0f;

    const float cos_pitch = std::cos(pitch);
    const std::array<float, 3> forward{
        -std::sin(yaw) * std::fabs(cos_pitch),
        std::cos(yaw) * std::fabs(cos_pitch),
        std::sin(pitch),
    };
    const std::array<float, 3> base_right{
        std::cos(yaw),
        std::sin(yaw),
        0.0f,
    };
    const std::array<float, 3> base_up{
        base_right[1] * forward[2],
        -base_right[0] * forward[2],
        base_right[0] * forward[1] - base_right[1] * forward[0],
    };

    const float cos_roll = std::cos(roll);
    const float sin_roll = std::sin(roll);
    std::array<float, 3> right{};
    std::array<float, 3> up{};
    for (int i = 0; i < 3; ++i) {
        right[i] = base_right[i] * cos_roll + base_up[i] * sin_roll;
        up[i] = -base_right[i] * sin_roll + base_up[i] * cos_roll;
    }
    const std::array<float, 3> backward{
        -forward[0],
        -forward[1],
        -forward[2],
    };
    const std::array<float, 3> position{position_x, position_y, position_z};

    output.view = {
        right[0], right[1], right[2],
        -(right[0] * position[0] + right[1] * position[1] + right[2] * position[2]),
        up[0], up[1], up[2],
        -(up[0] * position[0] + up[1] * position[1] + up[2] * position[2]),
        backward[0], backward[1], backward[2],
        -(backward[0] * position[0] + backward[1] * position[1] + backward[2] * position[2]),
        0.0f, 0.0f, 0.0f, 1.0f,
    };
    if (!finite_matrix(output.view)) {
        error = "Constructed view matrix is not finite";
        return CaptureStatus::InvalidCameraParameters;
    }
    return CaptureStatus::Ok;
}

RgbdCapture& RgbdCapture::instance() {
    // Intentionally process-lifetime: joining worker threads from DLL detach runs
    // under the loader lock and is less safe than allowing process teardown.
    static RgbdCapture* capture = new RgbdCapture();
    return *capture;
}

RgbdCapture::RgbdCapture()
    : request_(nullptr),
      completed_(nullptr),
      frame_id_(0),
      slots_{new Slot(), new Slot(), new Slot()},
      next_slot_(0),
      worker_(&RgbdCapture::worker_main, this) {
    worker_.detach();
}

bool RgbdCapture::begin_request(
    std::uint64_t request_id,
    std::uint32_t timeout_ms,
    CaptureStatus& status,
    std::string& error) {
    if (request_id == 0 || timeout_ms == 0) {
        status = CaptureStatus::InternalError;
        error = "request_id and timeout_ms must be non-zero";
        return false;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (request_ != nullptr || completed_ != nullptr) {
        status = CaptureStatus::Busy;
        error = "Another RGB-D request is active";
        return false;
    }
    request_ = new Request();
    request_->id = request_id;
    request_->minimum_frame = frame_id_ + 1;
    request_->deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    status = CaptureStatus::Ok;
    error.clear();
    return true;
}

void RgbdCapture::submit_camera(
    std::uint64_t request_id,
    const CaptureCamera& camera) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (request_ == nullptr || request_->id != request_id ||
        request_->stage != RequestStage::WaitingCamera) {
        return;
    }
    if (!std::isfinite(camera.fov_degrees) ||
        !std::isfinite(camera.near_clip) ||
        !std::isfinite(camera.far_clip) ||
        camera.fov_degrees <= 0.0f ||
        camera.fov_degrees >= 180.0f ||
        camera.near_clip <= 0.0f ||
        camera.far_clip <= camera.near_clip ||
        !finite_matrix(camera.view)) {
        const std::uint64_t id = request_->id;
        delete request_;
        request_ = nullptr;
        completed_ = new CaptureResult();
        completed_->status = CaptureStatus::InvalidCameraParameters;
        completed_->error = "Camera metadata failed validation";
        completed_->request_id = id;
        result_cv_.notify_all();
        return;
    }
    request_->camera = camera;
    request_->stage = RequestStage::WaitingFrame;
}

bool RgbdCapture::wait_result(
    std::uint64_t request_id,
    std::chrono::milliseconds timeout,
    CaptureResult& result) {
    std::unique_lock<std::mutex> lock(mutex_);
    const bool ready = result_cv_.wait_for(lock, timeout, [&]() {
        return completed_ != nullptr && completed_->request_id == request_id;
    });
    if (!ready) {
        return false;
    }
    result = std::move(*completed_);
    delete completed_;
    completed_ = nullptr;
    return true;
}

bool RgbdCapture::try_take_result(
    std::uint64_t request_id,
    CaptureResult& result) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (completed_ == nullptr || completed_->request_id != request_id) {
        return false;
    }
    result = std::move(*completed_);
    delete completed_;
    completed_ = nullptr;
    return true;
}

void RgbdCapture::cancel_request(std::uint64_t request_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (request_ != nullptr && request_->id == request_id) {
        delete request_;
        request_ = nullptr;
    }
    if (completed_ != nullptr && completed_->request_id == request_id) {
        delete completed_;
        completed_ = nullptr;
    }
}

void RgbdCapture::observe_depth_target(ID3D11DepthStencilView* dsv) {
    if (dsv == nullptr) {
        return;
    }
    ComPtr<ID3D11Resource> resource;
    dsv->GetResource(resource.GetAddressOf());
    ComPtr<ID3D11Texture2D> texture;
    if (resource == nullptr || FAILED(resource.As(&texture)) || texture == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(render_mutex_);
    const auto duplicate = std::find_if(
        depth_candidates_.begin(),
        depth_candidates_.end(),
        [&](const ComPtr<ID3D11Texture2D>& existing) {
            return existing.Get() == texture.Get();
        });
    if (duplicate == depth_candidates_.end()) {
        depth_candidates_.push_back(texture);
    }
}

void RgbdCapture::on_present(IDXGISwapChain* swap_chain) {
    if (swap_chain == nullptr) {
        return;
    }

    ComPtr<ID3D11Device> device;
    HRESULT hr = swap_chain->GetDevice(
        __uuidof(ID3D11Device),
        reinterpret_cast<void**>(device.GetAddressOf()));
    if (FAILED(hr) || device == nullptr) {
        return;
    }
    ComPtr<ID3D11DeviceContext> context;
    device->GetImmediateContext(context.GetAddressOf());
    if (context == nullptr) {
        return;
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        ++frame_id_;
    }

    poll_gpu(device.Get(), context.Get());

    ComPtr<ID3D11Texture2D> back_buffer;
    hr = swap_chain->GetBuffer(
        0,
        __uuidof(ID3D11Texture2D),
        reinterpret_cast<void**>(back_buffer.GetAddressOf()));
    if (FAILED(hr) || back_buffer == nullptr) {
        return;
    }
    D3D11_TEXTURE2D_DESC back_desc{};
    back_buffer->GetDesc(&back_desc);

    std::vector<ComPtr<ID3D11Texture2D>> candidates;
    {
        std::lock_guard<std::mutex> lock(render_mutex_);
        candidates.swap(depth_candidates_);
    }
    schedule_capture(
        device.Get(),
        context.Get(),
        back_buffer.Get(),
        back_desc,
        std::move(candidates));
}

std::uint64_t RgbdCapture::current_frame_id() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return frame_id_;
}

void RgbdCapture::poll_gpu(
    ID3D11Device* device,
    ID3D11DeviceContext* context) {
    for (Slot* slot : slots_) {
        if (slot->stage != SlotStage::GpuPending) {
            continue;
        }
        if (slot->device.Get() != device) {
            const std::uint64_t request_id = slot->request_id;
            slot->reset_resources();
            fail(
                request_id,
                CaptureStatus::ResourceGenerationChanged,
                "D3D11 device changed while RGB-D readback was pending");
            continue;
        }
        const HRESULT query_hr = context->GetData(
            slot->completion.Get(),
            nullptr,
            0,
            D3D11_ASYNC_GETDATA_DONOTFLUSH);
        if (query_hr == S_FALSE) {
            continue;
        }
        if (FAILED(query_hr)) {
            const std::uint64_t request_id = slot->request_id;
            const std::string error = hresult_text("GetData", query_hr);
            slot->reset_resources();
            fail(request_id, CaptureStatus::GpuReadbackFailed, error);
            continue;
        }

        try {
            auto raw = std::make_unique<RawFrame>();
            raw->request_id = slot->request_id;
            raw->frame_id = slot->frame_id;
            raw->width = slot->width;
            raw->height = slot->height;
            raw->camera = slot->camera;
            {
                MappedTexture color_map(context, slot->color.Get());
                copy_rows(
                    color_map.mapped,
                    slot->height,
                    static_cast<std::size_t>(slot->width) * 4,
                    raw->bgra);
            }
            {
                MappedTexture depth_map(context, slot->depth.Get());
                copy_rows(
                    depth_map.mapped,
                    slot->height,
                    static_cast<std::size_t>(slot->width) * 8,
                    raw->packed_depth);
            }
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (request_ != nullptr && request_->id == raw->request_id) {
                    request_->stage = RequestStage::WorkerPending;
                }
            }
            {
                std::lock_guard<std::mutex> lock(worker_mutex_);
                worker_queue_.push_back(raw.release());
            }
            worker_cv_.notify_one();
            slot->release_frame();
        } catch (const std::exception& exception) {
            const std::uint64_t request_id = slot->request_id;
            slot->reset_resources();
            fail(
                request_id,
                CaptureStatus::GpuReadbackFailed,
                exception.what());
        }
    }
}

void RgbdCapture::schedule_capture(
    ID3D11Device* device,
    ID3D11DeviceContext* context,
    ID3D11Texture2D* back_buffer,
    const D3D11_TEXTURE2D_DESC& back_desc,
    std::vector<ComPtr<ID3D11Texture2D>> depth_candidates) {
    std::uint64_t request_id = 0;
    std::uint64_t frame_id = 0;
    CaptureCamera camera;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (request_ == nullptr || request_->stage != RequestStage::WaitingFrame) {
            return;
        }
        if (std::chrono::steady_clock::now() > request_->deadline) {
            return;
        }
        if (frame_id_ < request_->minimum_frame) {
            return;
        }
        request_id = request_->id;
        frame_id = frame_id_;
        camera = request_->camera;
    }

    if (back_desc.Format != DXGI_FORMAT_B8G8R8A8_UNORM) {
        fail(
            request_id,
            CaptureStatus::RgbFormatMismatch,
            "Backbuffer format is not DXGI_FORMAT_B8G8R8A8_UNORM");
        return;
    }
    if (back_desc.SampleDesc.Count != 1) {
        fail(
            request_id,
            CaptureStatus::RgbFormatMismatch,
            "MSAA backbuffers are not supported by deterministic RGB-D capture");
        return;
    }

    std::vector<ComPtr<ID3D11Texture2D>> full_size;
    for (const auto& candidate : depth_candidates) {
        D3D11_TEXTURE2D_DESC depth_desc{};
        candidate->GetDesc(&depth_desc);
        if (depth_desc.Width == back_desc.Width &&
            depth_desc.Height == back_desc.Height) {
            full_size.push_back(candidate);
        }
    }
    if (full_size.empty()) {
        // GTA does not necessarily clear the main depth target in every render
        // cycle. This Present cannot form a same-frame RGB-D pair, so leave the
        // request pending for the next eligible cycle. No previous DSV is reused.
        return;
    }
    std::vector<ComPtr<ID3D11Texture2D>> matching;
    for (const auto& candidate : full_size) {
        D3D11_TEXTURE2D_DESC depth_desc{};
        candidate->GetDesc(&depth_desc);
        if (depth_desc.Format == DXGI_FORMAT_R32G8X24_TYPELESS &&
            depth_desc.SampleDesc.Count == 1) {
            matching.push_back(candidate);
        }
    }
    if (matching.empty()) {
        fail(
            request_id,
            CaptureStatus::DepthFormatMismatch,
            "Full-size depth targets exist, but none use non-MSAA "
            "DXGI_FORMAT_R32G8X24_TYPELESS");
        return;
    }
    if (matching.size() != 1) {
        fail(
            request_id,
            CaptureStatus::DepthTargetAmbiguous,
            "More than one R32G8X24 depth target matched the current backbuffer");
        return;
    }

    Slot* free_slot = nullptr;
    for (std::size_t offset = 0; offset < slots_.size(); ++offset) {
        Slot* slot = slots_[(next_slot_ + offset) % slots_.size()];
        if (slot->stage == SlotStage::Free) {
            free_slot = slot;
            next_slot_ = (next_slot_ + offset + 1) % slots_.size();
            break;
        }
    }
    if (free_slot == nullptr) {
        fail(
            request_id,
            CaptureStatus::InternalError,
            "All RGB-D staging slots are busy");
        return;
    }

    try {
        D3D11_TEXTURE2D_DESC depth_desc{};
        matching.front()->GetDesc(&depth_desc);
        if (depth_desc.Format != DXGI_FORMAT_R32G8X24_TYPELESS) {
            fail(
                request_id,
                CaptureStatus::DepthFormatMismatch,
                "Depth target format changed before scheduling");
            return;
        }

        bool recreate = free_slot->device.Get() != device ||
            free_slot->color == nullptr ||
            free_slot->depth == nullptr ||
            free_slot->completion == nullptr;
        if (!recreate) {
            D3D11_TEXTURE2D_DESC existing_color{};
            D3D11_TEXTURE2D_DESC existing_depth{};
            free_slot->color->GetDesc(&existing_color);
            free_slot->depth->GetDesc(&existing_depth);
            recreate =
                existing_color.Width != back_desc.Width ||
                existing_color.Height != back_desc.Height ||
                existing_color.Format != back_desc.Format ||
                existing_depth.Width != depth_desc.Width ||
                existing_depth.Height != depth_desc.Height ||
                existing_depth.Format != depth_desc.Format;
        }
        if (recreate) {
            free_slot->reset_resources();
            create_staging_texture(device, back_desc, free_slot->color);
            create_staging_texture(device, depth_desc, free_slot->depth);
            create_event_query(device, free_slot->completion);
            free_slot->device = device;
        }
        free_slot->request_id = request_id;
        free_slot->frame_id = frame_id;
        free_slot->width = back_desc.Width;
        free_slot->height = back_desc.Height;
        free_slot->camera = camera;

        context->CopyResource(free_slot->color.Get(), back_buffer);
        context->CopyResource(free_slot->depth.Get(), matching.front().Get());
        context->End(free_slot->completion.Get());
        free_slot->stage = SlotStage::GpuPending;

        std::lock_guard<std::mutex> lock(mutex_);
        if (request_ != nullptr && request_->id == request_id) {
            request_->stage = RequestStage::GpuPending;
        }
    } catch (const std::exception& exception) {
        free_slot->reset_resources();
        fail(
            request_id,
            CaptureStatus::GpuReadbackFailed,
            exception.what());
    }
}

void RgbdCapture::worker_main() {
    for (;;) {
        RawFrame* raw = nullptr;
        {
            std::unique_lock<std::mutex> lock(worker_mutex_);
            worker_cv_.wait(lock, [&]() { return !worker_queue_.empty(); });
            raw = worker_queue_.front();
            worker_queue_.pop_front();
        }

        CaptureResult result;
        result.status = CaptureStatus::Ok;
        result.request_id = raw->request_id;
        result.frame_id = raw->frame_id;
        result.width = raw->width;
        result.height = raw->height;
        result.camera = raw->camera;

        try {
            const float aspect =
                static_cast<float>(raw->height) / static_cast<float>(raw->width);
            const float fov_radians =
                raw->camera.fov_degrees * kPi / 180.0f;
            const float tangent = std::tan(fov_radians * 0.5f);
            const float near_clip = raw->camera.near_clip;
            const float far_clip = raw->camera.far_clip;
            const float near_minus_far = near_clip - far_clip;

            // Same reversed-Z projection used by GTA5Event's
            // postprocessing/gta_math.py::construct_proj_matrix.
            Eigen::Matrix4f projection = Eigen::Matrix4f::Zero();
            projection(0, 0) = aspect / tangent;
            projection(1, 1) = 1.0f / tangent;
            projection(2, 2) = -near_clip / near_minus_far;
            projection(2, 3) = (-near_clip * far_clip) / near_minus_far;
            projection(3, 2) = -1.0f;
            if (!projection.allFinite() ||
                std::fabs(projection.determinant()) < kHomogeneousEpsilon) {
                throw std::runtime_error("Constructed projection matrix is invalid");
            }
            const Eigen::Matrix4f inverse_projection = projection.inverse();
            if (!inverse_projection.allFinite()) {
                throw std::runtime_error("Projection matrix inverse is not finite");
            }
            for (int row = 0; row < 4; ++row) {
                for (int column = 0; column < 4; ++column) {
                    result.camera.projection[static_cast<std::size_t>(row) * 4 + column] =
                        projection(row, column);
                }
            }

            const std::size_t pixel_count =
                static_cast<std::size_t>(raw->width) * raw->height;
            if (raw->bgra.size() != pixel_count * 4 ||
                raw->packed_depth.size() != pixel_count * 8) {
                throw std::runtime_error("Mapped RGB-D byte counts do not match dimensions");
            }

            result.rgb.resize(pixel_count * 3);
            result.depth_meters.resize(pixel_count);
            for (std::size_t index = 0; index < pixel_count; ++index) {
                result.rgb[index * 3 + 0] = raw->bgra[index * 4 + 2];
                result.rgb[index * 3 + 1] = raw->bgra[index * 4 + 1];
                result.rgb[index * 3 + 2] = raw->bgra[index * 4 + 0];

                float ndc_depth = 0.0f;
                std::memcpy(
                    &ndc_depth,
                    raw->packed_depth.data() + index * 8,
                    sizeof(float));
                if (!std::isfinite(ndc_depth) ||
                    ndc_depth < -kDepthTolerance ||
                    ndc_depth > 1.0f + kDepthTolerance) {
                    throw std::runtime_error("Depth buffer contains a value outside [0, 1]");
                }
                const Eigen::Vector4f ndc(0.0f, 0.0f, ndc_depth, 1.0f);
                const Eigen::Vector4f view_homogeneous = inverse_projection * ndc;
                if (!view_homogeneous.allFinite() ||
                    std::fabs(view_homogeneous.w()) < kHomogeneousEpsilon) {
                    throw std::runtime_error("Depth inverse projection produced invalid homogeneous coordinates");
                }
                const float depth_meters =
                    -view_homogeneous.z() / view_homogeneous.w();
                if (!std::isfinite(depth_meters) || depth_meters < 0.0f) {
                    throw std::runtime_error("Depth inverse projection produced an invalid metric value");
                }
                result.depth_meters[index] = depth_meters;
            }
        } catch (const std::exception& exception) {
            result.status = CaptureStatus::DepthConversionFailed;
            result.error = exception.what();
            result.rgb.clear();
            result.depth_meters.clear();
        }

        delete raw;
        complete(std::move(result));
    }
}

void RgbdCapture::complete(CaptureResult result) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (request_ == nullptr || request_->id != result.request_id) {
        return;
    }
    delete request_;
    request_ = nullptr;
    if (completed_ != nullptr) {
        delete completed_;
    }
    completed_ = new CaptureResult(std::move(result));
    result_cv_.notify_all();
}

void RgbdCapture::fail(
    std::uint64_t request_id,
    CaptureStatus status,
    const std::string& error) {
    CaptureResult result;
    result.status = status;
    result.error = error;
    result.request_id = request_id;
    LOGE(
        "rgbd_capture",
        std::string(capture_status_name(status)) + ": " + error);
    complete(std::move(result));
}
