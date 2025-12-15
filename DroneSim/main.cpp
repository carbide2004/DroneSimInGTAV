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
catchState cmdToCatch = catchStop;	
static WCHAR imgPath[fileLength] = L"data\\screen.bmp";
static char rawPath[fileLength] = "data\\stencil.raw";
static char depthPath[fileLength] = "data\\depth.raw";
static char matrixPath[fileLength] = "data\\matrix.txt";
static bool onlyScreen = false, forceSave = false;
std::string g_rgbCapturedFilePath;
std::string g_depthCapturedFilePath;
std::string g_stencilCapturedFilePath;
std::string g_matrixCapturedFilePath = matrixPath;
std::queue<std::string> g_cmdQueue;


inline void makeCmdStart()
{
	cmdToCatch = catchStart;
}
inline void makeCmdStop()
{
	cmdToCatch = catchStop;
}
inline void cmdCatchScreen()
{
	cmdToCatch = catchScreen;
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
		//keyboardHandlerRegister(OnKeyboardMessage);
		scriptRegister(hinstance, scriptMain);
		break;
case DLL_PROCESS_DETACH:
        Logger::shutdown();
		res = MH_Uninitialize();
        if (res != MH_OK) LOGE("main", "Could not deinit MiniHook");
		presentCallbackUnregister(presentCallback);
		//keyboardHandlerUnregister(OnKeyboardMessage);
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
	void** vtbl = *reinterpret_cast<void***>(inst);
    
	//fprintf(f, "Hooking %p at offset %d\n", inst, offset);
	MH_STATUS res = MH_OK;
	DWORD oldProt = 0;
	vtbl += offset;
	//VirtualProtect(vtbl, 8, PAGE_READWRITE, &oldProt);
	if (unhook)
	{
		res = MH_DisableHook(vtbl);
            if(res != MH_OK) LOGE("main", std::string("error ") + std::to_string(res) + " disabling hook at offset " + std::to_string(offset));
		orig<offset, T> = nullptr;
	}
	else {
		if(targets<offset, T> != nullptr && targets<offset, T> != *vtbl)
		{
            LOGW("main", "detected target change, someone else is screwing with our functions");
			res = MH_DisableHook(targets<offset, T>);
            if (res != MH_OK) LOGE("main", std::string("error ") + std::to_string(res) + " disabling hook at offset " + std::to_string(offset));
			res = MH_RemoveHook(targets<offset, T>);
            if (res != MH_OK) LOGE("main", std::string("error ") + std::to_string(res) + " removing hook at offset " + std::to_string(offset));
			targets<offset, T> = nullptr;
			orig<offset, T> = nullptr;
		}
		if (orig<offset, T> == nullptr && targets<offset, T> != *vtbl) {
			std::chrono::milliseconds ms = std::chrono::duration_cast< std::chrono::milliseconds >(
				std::chrono::system_clock::now().time_since_epoch()
				);
			//fprintf(f, "[%I64d] :  create hook\n", ms.count());
			res = MH_CreateHook(*vtbl, hook, &(orig<offset, T>));
            if(res != MH_OK) LOGE("main", std::string("error ") + std::to_string(res) + " creating hook at offset " + std::to_string(offset));
			
		}
		if (targets<offset, T> != *vtbl) {
			res = MH_EnableHook(*vtbl);
            if (res != MH_OK) LOGE("main", std::string("error ") + std::to_string(res) + " enabling hook at offset " + std::to_string(offset));
			targets<offset, T> = *vtbl;
		}
		//*vtbl = reinterpret_cast<long long>(hook);
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
        ExtractConstantBuffer(dev.Get(), self, buf.Get());
    }

	draw_indexed_count += 1;
	origMethod(self, indexCount, startLoc, baseLoc);
}
void clear_render_target_view_hook(ID3D11DeviceContext* self, ID3D11RenderTargetView* rtv, float color[4])
{
	auto origMethod = reinterpret_cast<void (*)(ID3D11DeviceContext*, ID3D11RenderTargetView*, float[4])>(orig<50, ID3D11DeviceContext>);
	
	ComPtr<ID3D11RenderTargetView> curRTV;
	self->OMGetRenderTargets(1, &curRTV, nullptr);
	if (curRTV != nullptr)
	{
		D3D11_TEXTURE2D_DESC desc;
		ComPtr<ID3D11Resource> res;
		ComPtr<ID3D11Texture2D> tex;
		curRTV->GetResource(&res);
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
		LOGE("main", "export screen failed");
	}
	else {
		LOGI("main", "export screen success");
	}
};

void clear_depth_stencil_view_hook(ID3D11DeviceContext* self, ID3D11DepthStencilView* dsv, UINT8 flags, float depth, UINT8 stencil)
{
	auto origMethod = reinterpret_cast<decltype(&clear_depth_stencil_view_hook)>(orig<53, ID3D11DeviceContext>);
	ComPtr<ID3D11DepthStencilView> curDSV;
	self->OMGetRenderTargets(1, nullptr, &curDSV);
	ComPtr<ID3D11Device> dev;
	self->GetDevice(&dev);
	if (curDSV != nullptr) {
		D3D11_TEXTURE2D_DESC desc;
		ComPtr<ID3D11Resource> res;
		ComPtr<ID3D11Texture2D> tex;
		curDSV->GetResource(&res);
		HRESULT hr = S_OK;
		hr = res.As(&tex);
		if (hr != S_OK) return;
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

			if (cmdToCatch == catchStart) {
				void *stencil_buf;
				void *depth_buf;
				int sizeStencil = export_get_stencil_buffer(&stencil_buf);
				int sizeDepth = export_get_depth_buffer(&depth_buf);
                screenShot();

				auto raw = fopen(rawPath, "wb");
				fwrite(stencil_buf, 1, sizeStencil, raw);
				fclose(raw);
                LOGI("main", std::string("write stencil into file: ") + rawPath);
				g_stencilCapturedFilePath = rawPath;

				auto depth_raw = fopen(depthPath, "wb");
				fwrite(depth_buf, 1, sizeDepth, depth_raw);
				fclose(depth_raw);
                LOGI("main", std::string("write depth into file: ") + depthPath);
				g_depthCapturedFilePath = depthPath;

				makeCmdStop();
			}
            
		}
	}
	origMethod(self, dsv, flags, depth, stencil);
}


void presentCallback(void* chain)
{	
    
	std::chrono::milliseconds ms = std::chrono::duration_cast< std::chrono::milliseconds >(
		std::chrono::system_clock::now().time_since_epoch()
		);

	// draw_indexed_count = 0;
	HRESULT hr2 = S_OK, hr1 = S_OK;
	ComPtr<ID3D11Device> dev;
	ComPtr<ID3D11DeviceContext> ctx;
	ComPtr<ID3D11Texture2D> backBuffer;

	auto swapChain = static_cast<IDXGISwapChain*>(chain);
	hr2 = swapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), reinterpret_cast<LPVOID*>(backBuffer.GetAddressOf()));
	if (hr2 != S_OK) throw std::system_error(hr2, std::system_category());

	swapChain = static_cast<IDXGISwapChain*>(chain);
	hr1 = swapChain->GetDevice(__uuidof(ID3D11Device), &dev);
	if (hr1 != S_OK) throw std::system_error(hr1, std::system_category());
	dev->GetImmediateContext(&ctx);

	ExtractScreenBuffer(ctx.Get(), backBuffer.Get(), hr2);
	ComPtr<ID3D11Multithread> multithread;
	hr2 = ctx.As(&multithread);
	if (hr2 != S_OK) throw std::system_error(hr2, std::system_category());
	multithread->SetMultithreadProtected(true);
	hook_function<drawIndexedOffset>(ctx.Get(), &draw_indexed_hook);
	
	hook_function<53>(ctx.Get(), &clear_depth_stencil_view_hook);
    
	
	ComPtr<ID3D11Resource> depthres;
	ComPtr<ID3D11Resource> colorres;
	ctx->OMGetRenderTargets(1, &lastRtv, nullptr);
	last_capture_color = system_clock::now();
	lastRtv->GetResource(&colorres);
	ExtractColorBuffer(dev.Get(), ctx.Get(), colorres.Get());
	//lastDsv.Reset();

	lastDsv = nullptr;
	lastRtv = nullptr;

	ms = std::chrono::duration_cast< std::chrono::milliseconds >(
		std::chrono::system_clock::now().time_since_epoch()
		);

    
}
