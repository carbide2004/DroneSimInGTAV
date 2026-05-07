#include "main.h"
#include <dxgi.h>
#include <d3d11.h>
#include <d3d11_4.h>
#include <wrl.h>
#include <ShlObj.h>
#include <system_error>
#include <string>
#include <filesystem>
#include <wincodec.h>
#include <cstdio>
#include <MinHook.h>
#include <cassert>
#include <chrono>
#include "export.h"
#include "script.h"
#include "logging.h"
#include <d3d11shader.h>
#include <queue>
#include <d3dcompiler.h>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <atlimage.h>
#include <fstream>
#include <ScreenGrab.h>
#include "keyboard.h"
#include <cstring>
#include <exception>
#include <atomic>
using Microsoft::WRL::ComPtr;
using namespace std::experimental::filesystem;
using namespace std::string_literals;
using std::chrono::milliseconds;
using std::chrono::time_point;
using std::chrono::system_clock;
using std::vector;
using Eigen::Matrix4f;
using Eigen::Vector3f;
using Eigen::Vector4f;
typedef void(*draw_indexed_hook_t)(ID3D11DeviceContext*, UINT, UINT, INT);
void presentCallback(void* chain);

// void scriptMain();

//void draw_indexed_hook(ID3D11DeviceContext3* self, UINT IndexStart, UINT StartIndexLocation, INT BaseVertexLocation);
static time_point<system_clock> last_capture_color;
static time_point<system_clock> last_capture_depth;
 
//--------
//offsets
//--------
const size_t drawIndexedOffset = 12;
const size_t clearDepthStencilViewOffset = 53;
//-------------------------
//interesting D3D resources
//-------------------------
static ComPtr<ID3D11DepthStencilView> lastDsv;
static ComPtr<ID3D11RenderTargetView> lastRtv;
static ComPtr<ID3D11Buffer> lastConstants;

static bool saveNextFrame = false;
static bool hooked = false;

//-------------------------
//global control variables
//-------------------------

static int draw_indexed_count = 0;

const size_t fileLength = 256;
std::atomic<catchState> cmdToCatch{catchStop};
static WCHAR imgPath[fileLength] = L"data\\screen.bmp";
static char rawPath[fileLength] = "data\\stencil.raw";
static char depthPath[fileLength] = "data\\depth.raw";
static char matrixPath[fileLength] = "data\\matrix.txt";
static bool onlyScreen = false, forceSave = false;
std::string g_rgbCapturedFilePath;
std::string g_depthCapturedFilePath;
std::string g_stencilCapturedFilePath;
std::string g_matrixCapturedFilePath = matrixPath;
 
static std::vector<unsigned char> g_lastRgbBytes;
static std::vector<unsigned char> g_lastDepthBytes;


inline void makeCmdStart()
{
	cmdToCatch.store(catchStart, std::memory_order_release);
}
inline void makeCmdStop()
{
	cmdToCatch.store(catchStop, std::memory_order_release);
}
inline void cmdCatchScreen()
{
	cmdToCatch.store(catchScreen, std::memory_order_release);
}

void catchCurveAndScreen(WCHAR *_imgPath, char *_rawPath, bool _forceSave, bool _onlyScreen)
{
	wcscpy(imgPath, _imgPath);
	onlyScreen = _onlyScreen;
	forceSave = _forceSave;
	if(!onlyScreen) strcpy(rawPath, _rawPath);
	makeCmdStart();
}

static void (_stdcall ID3D11DeviceContext::* origDrawInstanced)(UINT, UINT, INT) = nullptr;
int __stdcall DllMain(HMODULE hinstance, DWORD reason, LPVOID lpReserved)
{
    MH_STATUS res;
	switch(reason)
	{
case DLL_PROCESS_ATTACH:
        Logger::init();
		res = MH_Initialize();
        if (res != MH_OK) LOGE("main", "Could not init Minihook");
		presentCallbackRegister(presentCallback);
		keyboardHandlerRegister(OnKeyboardMessage);
		scriptRegister(hinstance, scriptMain);
		break;
case DLL_PROCESS_DETACH:
        Logger::shutdown();
		res = MH_Uninitialize();
        if (res != MH_OK) LOGE("main", "Could not deinit MiniHook");
		presentCallbackUnregister(presentCallback);
		keyboardHandlerUnregister(OnKeyboardMessage);
		//scriptUnregister(hinstance);

		break;
	}
    return TRUE;
}

template<int offset, typename T>
static void* orig;
template<int offset, typename T>
static void* targets;
template<int offset, typename T>
void hook_function(T* inst, void* hook, bool unhook = false)
{
	//__debugbreak();
	if (inst == nullptr) {
		LOGE("main", std::string("hook_function: null instance at offset ") + std::to_string(offset));
		return;
	}
	void** vtbl = *reinterpret_cast<void***>(inst);
	if (vtbl == nullptr) {
		LOGE("main", std::string("hook_function: null vtable at offset ") + std::to_string(offset));
		return;
	}
    
	//fprintf(f, "Hooking %p at offset %d\n", inst, offset);
	MH_STATUS res = MH_OK;
	DWORD oldProt = 0;
	vtbl += offset;
	if (*vtbl == nullptr) {
		LOGE("main", std::string("hook_function: null target at offset ") + std::to_string(offset));
		return;
	}
	//VirtualProtect(vtbl, 8, PAGE_READWRITE, &oldProt);
	if (unhook)
	{
		res = MH_DisableHook(vtbl);
            if(res != MH_OK) LOGE("main", std::string("error ") + std::to_string(res) + " disabling hook at offset " + std::to_string(offset));
		orig<offset, T> = nullptr;
	}
	else { // 执行 Hook 操作
        // 1. 检查是否检测到目标改变，并清除旧状态
        if (targets<offset, T> != nullptr && targets<offset, T> != *vtbl)
        {
            LOGT("main", "detected target change, someone else is screwing with our functions. Re-hooking.");
            
            // 尝试禁用和移除 (如果失败，就忽略，MinHook 状态优先)
            MH_DisableHook(targets<offset, T>);
            MH_RemoveHook(targets<offset, T>);
            
            // 无论 MinHook 操作结果如何，都清除本地状态，强制进入创建流程
            targets<offset, T> = nullptr;
            orig<offset, T> = nullptr;
        }

        // 2. 检查是否需要创建挂钩
        // 只有当 orig<offset, T> 为 nullptr 时才创建
        if (orig<offset, T> == nullptr) {
            // MH_CreateHook 会检查 *vtbl 是否已经被挂钩 (如果被挂钩，会返回 error 3)
            res = MH_CreateHook(*vtbl, hook, &(orig<offset, T>));
            if (res != MH_OK && res != MH_ERROR_ALREADY_CREATED) {
                 LOGE("main", std::string("error ") + std::to_string(res) + " creating hook at offset " + std::to_string(offset));
                 // 如果创建失败，则重置 orig，避免下次再次尝试创建
                 orig<offset, T> = nullptr; 
                 targets<offset, T> = nullptr; // 同时清除 targets
                 return; // 挂钩失败，提前返回
            }
        }
        
        // 3. 检查是否需要启用挂钩
        // 只有当 targets<offset, T> 与当前 *vtbl 不一致时，才尝试启用并更新 targets
        if (targets<offset, T> != *vtbl) {
            // MH_EnableHook 会检查是否已经被启用 (如果已启用，会返回 error 5)
            res = MH_EnableHook(*vtbl);
            if (res != MH_OK && res != MH_ERROR_ENABLED) {
                LOGE("main", std::string("error ") + std::to_string(res) + " enabling hook at offset " + std::to_string(offset));
                // 如果启用失败，清除本地状态
                targets<offset, T> = nullptr; 
                orig<offset, T> = nullptr; 
                return; // 启用失败，提前返回
            }
            
            // 如果成功 (或已经是启用状态 MH_ERROR_ENABLED)，则更新本地状态
            targets<offset, T> = *vtbl;
        }
    }
	//VirtualProtect(vtbl, 8, oldProt, nullptr);
	//fprintf(f, "clear_hook: %p\n", hook);
	//fprintf(f, "clearFn: %p\n", (void*)(*(*reinterpret_cast<long long**>(inst) + 50)));
    
}

template<int offset, typename T>
void unhook_function(T* inst)
{
	hook_function<offset>(inst, nullptr, true);
}
void draw_hook_impl()
{
    LOGD("main", "Draw Call");
}
void draw_indexed_hook(ID3D11DeviceContext* self, UINT indexCount, UINT startLoc, UINT baseLoc) {
	auto origMethod = reinterpret_cast<decltype(draw_indexed_hook)*>(orig<drawIndexedOffset, ID3D11DeviceContext>);
	if (origMethod == nullptr) {
		LOGE("main", "draw_indexed_hook: original method is null");
		return;
	}
	if (self == nullptr) {
		LOGE("main", "draw_indexed_hook: device context is null");
		return;
	}
	HRESULT hr;
	ComPtr<ID3D11VertexShader> vs;
	self->VSGetShader(&vs, nullptr, nullptr);
	ComPtr<ID3D11Buffer> buf;
	ComPtr<ID3D11Device> dev;
	self->GetDevice(&dev);
	self->VSGetConstantBuffers(1, 1, &buf);
	// auto f = fopen(logFilePath, "a");
	// fprintf(f, "Draw Indexed Call count: %d\n", draw_indexed_count);
	// fclose(f);
    if (buf != nullptr && draw_indexed_count == 1000) {
        lastConstants = buf;
		try {
			ExtractConstantBuffer(dev.Get(), self, buf.Get());
		} catch (const std::exception& e) {
			LOGE("main", std::string("draw_indexed_hook: ExtractConstantBuffer failed: ") + e.what());
		} catch (...) {
			LOGE("main", "draw_indexed_hook: ExtractConstantBuffer failed with unknown exception");
		}
    }

	draw_indexed_count += 1;
	origMethod(self, indexCount, startLoc, baseLoc);
}
void clear_render_target_view_hook(ID3D11DeviceContext* self, ID3D11RenderTargetView* rtv, float color[4])
{
	auto origMethod = reinterpret_cast<void (*)(ID3D11DeviceContext*, ID3D11RenderTargetView*, float[4])>(orig<50, ID3D11DeviceContext>);
	if (origMethod == nullptr) {
		LOGE("main", "clear_render_target_view_hook: original method is null");
		return;
	}
	if (self == nullptr) {
		LOGE("main", "clear_render_target_view_hook: device context is null");
		return;
	}
	
	ComPtr<ID3D11RenderTargetView> curRTV;
	self->OMGetRenderTargets(1, &curRTV, nullptr);
	if (curRTV != nullptr)
	{
		D3D11_TEXTURE2D_DESC desc;
		ComPtr<ID3D11Resource> res;
		ComPtr<ID3D11Texture2D> tex;
		curRTV->GetResource(&res);
		if (res == nullptr) {
			origMethod(self, rtv, color);
			return;
		}
		HRESULT hr = S_OK;
		hr = res.As(&tex);
		if (hr != S_OK) return;
		tex->GetDesc(&desc);
		if (desc.Format == DXGI_FORMAT_B8G8R8A8_UNORM && desc.Width > 600 && desc.Height > 600) {
			lastRtv = curRTV;
		}
	}
	origMethod(self, rtv, color);
}

auto screenShot = []() {
	int screenCapResult = export_get_screen_buffer(imgPath);
	std::chrono::milliseconds ms = std::chrono::duration_cast< std::chrono::milliseconds >(
			std::chrono::system_clock::now().time_since_epoch()
			);
	char currentImgPathNarrow[fileLength];
	sprintf(currentImgPathNarrow, "data\\screen.bmp");
	g_rgbCapturedFilePath = currentImgPathNarrow;
	if (screenCapResult != 1) {
		LOGE("main", "export screen failed, screenCapResult=" + std::to_string(screenCapResult));
	}
	else {
		LOGI("main", "export screen success");
	}
};

void clear_depth_stencil_view_hook(ID3D11DeviceContext* self, ID3D11DepthStencilView* dsv, UINT8 flags, float depth, UINT8 stencil)
{
	auto origMethod = reinterpret_cast<decltype(&clear_depth_stencil_view_hook)>(orig<53, ID3D11DeviceContext>);
	if (origMethod == nullptr) {
		LOGE("main", "clear_depth_stencil_view_hook: original method is null");
		return;
	}
	if (self == nullptr) {
		LOGE("main", "clear_depth_stencil_view_hook: device context is null");
		return;
	}

	try {
		ComPtr<ID3D11DepthStencilView> curDSV;
		self->OMGetRenderTargets(1, nullptr, &curDSV);
		ComPtr<ID3D11Device> dev;
		self->GetDevice(&dev);
		if (curDSV != nullptr && dev != nullptr) {
			D3D11_TEXTURE2D_DESC desc;
			ComPtr<ID3D11Resource> res;
			ComPtr<ID3D11Texture2D> tex;
			curDSV->GetResource(&res);
			if (res == nullptr) {
				origMethod(self, dsv, flags, depth, stencil);
				return;
			}
			HRESULT hr = S_OK;
			hr = res.As(&tex);
			if (hr == S_OK && tex != nullptr) {
				tex->GetDesc(&desc);
				
				if (lastDsv == nullptr && desc.Width > 600 && desc.Height > 600 && desc.Format == DXGI_FORMAT_R32G8X24_TYPELESS) {
					lastDsv = curDSV;
					std::chrono::milliseconds ms = std::chrono::duration_cast< std::chrono::milliseconds >(
						std::chrono::system_clock::now().time_since_epoch()
						);
		            
					//go = true;
					//fprintf(f, "[%I64d] : trans stencil info over, cmdToCatch = %d.\n", ms.count(), cmdToCatch);
					
					ExtractDepthBuffer(dev.Get(), self, res.Get());
					last_capture_depth = system_clock::now();

					if (cmdToCatch.load(std::memory_order_acquire) == catchStart) {
						void* rgb_buf = nullptr;
						void* depth_buf = nullptr;
						void* stencil_buf = nullptr;
						int sizeRgb = export_get_color_buffer(&rgb_buf);
						int sizeDepth = export_get_depth_buffer(&depth_buf);
						int sizeStencil = export_get_stencil_buffer(&stencil_buf);

						if (sizeRgb <= 0 || sizeDepth <= 0 || sizeStencil <= 0 || rgb_buf == nullptr || depth_buf == nullptr || stencil_buf == nullptr) {
							LOGE("main", "capture failed because one or more exported buffers are invalid");
							makeCmdStop();
						}
						else {
							bool same_rgb = false;
							bool same_depth = false;
							if (!g_lastRgbBytes.empty() && g_lastRgbBytes.size() == static_cast<size_t>(sizeRgb)) {
								same_rgb = std::memcmp(rgb_buf, g_lastRgbBytes.data(), sizeRgb) == 0;
							}
							if (!g_lastDepthBytes.empty() && g_lastDepthBytes.size() == static_cast<size_t>(sizeDepth)) {
								same_depth = std::memcmp(depth_buf, g_lastDepthBytes.data(), sizeDepth) == 0;
							}

							if (same_rgb || same_depth) {
								LOGW("main", std::string("capture skipped because ") +
									(same_rgb ? "RGB " : "") +
									(same_depth ? "DEPTH " : "") +
									"unchanged from last frame");
								makeCmdStop();
							} 
							else {
								g_lastRgbBytes.assign(reinterpret_cast<unsigned char*>(rgb_buf),
									reinterpret_cast<unsigned char*>(rgb_buf) + sizeRgb);
								g_lastDepthBytes.assign(reinterpret_cast<unsigned char*>(depth_buf),
									reinterpret_cast<unsigned char*>(depth_buf) + sizeDepth);

								screenShot();

								auto raw = fopen(rawPath, "wb");
								if (raw != nullptr) {
									fwrite(stencil_buf, 1, sizeStencil, raw);
									fclose(raw);
									LOGI("main", std::string("write stencil into file: ") + rawPath);
									g_stencilCapturedFilePath = rawPath;
								} else {
									LOGE("main", std::string("failed to open stencil file: ") + rawPath);
								}

								auto depth_raw = fopen(depthPath, "wb");
								if (depth_raw != nullptr) {
									fwrite(depth_buf, 1, sizeDepth, depth_raw);
									fclose(depth_raw);
									LOGI("main", std::string("write depth into file: ") + depthPath);
									g_depthCapturedFilePath = depthPath;
								} else {
									LOGE("main", std::string("failed to open depth file: ") + depthPath);
								}

								makeCmdStop();
							}
						}
					}
				}
			}
		}
	} catch (const std::exception& e) {
		LOGE("main", std::string("clear_depth_stencil_view_hook failed: ") + e.what());
	} catch (...) {
		LOGE("main", "clear_depth_stencil_view_hook failed with unknown exception");
	}
	origMethod(self, dsv, flags, depth, stencil);
}


void presentCallback(void* chain)
{	
	try {
		std::chrono::milliseconds ms = std::chrono::duration_cast< std::chrono::milliseconds >(
			std::chrono::system_clock::now().time_since_epoch()
			);

		if (chain == nullptr) {
			LOGE("main", "presentCallback: swap chain is null");
			return;
		}

		// draw_indexed_count = 0;
		HRESULT hr2 = S_OK, hr1 = S_OK;
		ComPtr<ID3D11Device> dev;
		ComPtr<ID3D11DeviceContext> ctx;
		ComPtr<ID3D11Texture2D> backBuffer;

		auto swapChain = static_cast<IDXGISwapChain*>(chain);
		hr2 = swapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), reinterpret_cast<LPVOID*>(backBuffer.GetAddressOf()));
		if (FAILED(hr2) || backBuffer == nullptr) {
			LOGE("main", std::string("presentCallback: GetBuffer failed, HRESULT: ") + std::to_string(hr2));
			return;
		}

		hr1 = swapChain->GetDevice(__uuidof(ID3D11Device), &dev);
		if (FAILED(hr1) || dev == nullptr) {
			LOGE("main", std::string("presentCallback: GetDevice failed, HRESULT: ") + std::to_string(hr1));
			return;
		}
		dev->GetImmediateContext(&ctx);
		if (ctx == nullptr) {
			LOGE("main", "presentCallback: immediate context is null");
			return;
		}

		ExtractScreenBuffer(ctx.Get(), backBuffer.Get(), hr2);
		ComPtr<ID3D11Multithread> multithread;
		hr2 = ctx.As(&multithread);
		if (SUCCEEDED(hr2) && multithread != nullptr) {
			multithread->SetMultithreadProtected(true);
		} else {
			LOGW("main", std::string("presentCallback: ID3D11Multithread unavailable, HRESULT: ") + std::to_string(hr2));
		}
		hook_function<drawIndexedOffset>(ctx.Get(), &draw_indexed_hook);
		
		hook_function<53>(ctx.Get(), &clear_depth_stencil_view_hook);
	    
		
		ComPtr<ID3D11Resource> colorres;
		ctx->OMGetRenderTargets(1, &lastRtv, nullptr);
		last_capture_color = system_clock::now();
		if (lastRtv != nullptr) {
			lastRtv->GetResource(&colorres);
			if (colorres != nullptr) {
				ExtractColorBuffer(dev.Get(), ctx.Get(), colorres.Get());
			} else {
				LOGW("main", "presentCallback: render target resource is null");
			}
		} else {
			LOGW("main", "presentCallback: render target view is null");
		}
		//lastDsv.Reset();

		lastDsv = nullptr;
		lastRtv = nullptr;

		ms = std::chrono::duration_cast< std::chrono::milliseconds >(
			std::chrono::system_clock::now().time_since_epoch()
			);
	} catch (const std::exception& e) {
		LOGE("main", std::string("presentCallback failed: ") + e.what());
	} catch (...) {
		LOGE("main", "presentCallback failed with unknown exception");
	}
}
