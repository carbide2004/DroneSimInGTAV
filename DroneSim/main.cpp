#include "main.h"

#include "keyboard.h"
#include "logging.h"
#include "rgbd_capture.h"
#include "script.h"

#include <MinHook.h>
#include <d3d11.h>
#include <dxgi.h>

#include <atomic>
#include <exception>
#include <mutex>
#include <string>

namespace {

constexpr std::size_t kClearDepthStencilViewOffset = 53;

template <std::size_t Offset, typename Interface>
void* g_original = nullptr;

template <std::size_t Offset, typename Interface>
void* g_target = nullptr;

std::once_flag g_minhook_once;
std::atomic<bool> g_minhook_ready{false};

bool ensure_minhook() {
    std::call_once(g_minhook_once, []() {
        const MH_STATUS status = MH_Initialize();
        if (status == MH_OK || status == MH_ERROR_ALREADY_INITIALIZED) {
            g_minhook_ready.store(true, std::memory_order_release);
        } else {
            LOGE(
                "main",
                std::string("MH_Initialize failed: ") +
                    std::to_string(static_cast<int>(status)));
        }
    });
    return g_minhook_ready.load(std::memory_order_acquire);
}
template <std::size_t Offset, typename Interface>
bool install_hook(Interface* instance, void* hook) {
    if (instance == nullptr || hook == nullptr || !ensure_minhook()) {
        return false;
    }
    void** vtable = *reinterpret_cast<void***>(instance);
    if (vtable == nullptr || vtable[Offset] == nullptr) {
        return false;
    }
    void* target = vtable[Offset];
    if (g_target<Offset, Interface> == target &&
        g_original<Offset, Interface> != nullptr) {
        return true;
    }
    if (g_target<Offset, Interface> != nullptr) {
        MH_DisableHook(g_target<Offset, Interface>);
        MH_RemoveHook(g_target<Offset, Interface>);
        g_target<Offset, Interface> = nullptr;
        g_original<Offset, Interface> = nullptr;
    }

    MH_STATUS status = MH_CreateHook(
        target,
        hook,
        &g_original<Offset, Interface>);
    if (status != MH_OK) {
        LOGE(
            "main",
            std::string("MH_CreateHook failed: ") +
                std::to_string(static_cast<int>(status)));
        g_original<Offset, Interface> = nullptr;
        return false;
    }
    status = MH_EnableHook(target);
    if (status != MH_OK && status != MH_ERROR_ENABLED) {
        LOGE(
            "main",
            std::string("MH_EnableHook failed: ") +
                std::to_string(static_cast<int>(status)));
        MH_RemoveHook(target);
        g_original<Offset, Interface> = nullptr;
        return false;
    }
    g_target<Offset, Interface> = target;
    LOGI("main", "Installed D3D11 ClearDepthStencilView hook");
    return true;
}

void remove_depth_hook() {
    void* target =
        g_target<kClearDepthStencilViewOffset, ID3D11DeviceContext>;
    if (target != nullptr) {
        MH_DisableHook(target);
        MH_RemoveHook(target);
        g_target<kClearDepthStencilViewOffset, ID3D11DeviceContext> = nullptr;
        g_original<kClearDepthStencilViewOffset, ID3D11DeviceContext> = nullptr;
    }
    if (g_minhook_ready.load(std::memory_order_acquire)) {
        MH_Uninitialize();
        g_minhook_ready.store(false, std::memory_order_release);
    }
}

void clear_depth_stencil_view_hook(
    ID3D11DeviceContext* context,
    ID3D11DepthStencilView* depth_stencil_view,
    UINT clear_flags,
    float depth,
    UINT8 stencil) {
    using Hook = void (*)(
        ID3D11DeviceContext*,
        ID3D11DepthStencilView*,
        UINT,
        float,
        UINT8);
    auto original = reinterpret_cast<Hook>(
        g_original<kClearDepthStencilViewOffset, ID3D11DeviceContext>);
    if (original == nullptr) {
        return;
    }

    // Observation only: no Map, CPU conversion, file I/O, or network work is
    // allowed in this hook.
    RgbdCapture::instance().observe_depth_target(depth_stencil_view);
    original(
        context,
        depth_stencil_view,
        clear_flags,
        depth,
        stencil);
}

void present_callback(void* chain) {
    if (chain == nullptr) {
        LOGE("main", "Present callback received a null swap chain");
        return;
    }
    try {
        auto* swap_chain = static_cast<IDXGISwapChain*>(chain);
        Microsoft::WRL::ComPtr<ID3D11Device> device;
        HRESULT hr = swap_chain->GetDevice(
            __uuidof(ID3D11Device),
            reinterpret_cast<void**>(device.GetAddressOf()));
        if (FAILED(hr) || device == nullptr) {
            LOGE("main", "IDXGISwapChain::GetDevice failed");
            return;
        }
        Microsoft::WRL::ComPtr<ID3D11DeviceContext> context;
        device->GetImmediateContext(context.GetAddressOf());
        if (context == nullptr) {
            LOGE("main", "D3D11 immediate context is null");
            return;
        }
        if (!install_hook<kClearDepthStencilViewOffset>(
                context.Get(),
                reinterpret_cast<void*>(&clear_depth_stencil_view_hook))) {
            LOGE("main", "Could not install the depth-target observation hook");
            return;
        }
        RgbdCapture::instance().on_present(swap_chain);
    } catch (const std::exception& exception) {
        LOGE(
            "main",
            std::string("Present callback failed: ") + exception.what());
    }
}

}  // namespace

BOOL APIENTRY DllMain(HMODULE module, DWORD reason, LPVOID) {
    switch (reason) {
        case DLL_PROCESS_ATTACH:
            DisableThreadLibraryCalls(module);
            Logger::init();
            presentCallbackRegister(present_callback);
            keyboardHandlerRegister(OnKeyboardMessage);
            scriptRegister(module, scriptMain);
            break;
        case DLL_PROCESS_DETACH:
            presentCallbackUnregister(present_callback);
            keyboardHandlerUnregister(OnKeyboardMessage);
            remove_depth_hook();
            Logger::shutdown();
            break;
        default:
            break;
    }
    return TRUE;
}
