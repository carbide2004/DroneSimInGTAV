#include "export.h"
#include "nativeCaller.h"
#include "natives.h"
#include <d3d11.h>
#include <cassert>
#include <wrl/client.h>
#include <system_error>
#include <memory>
#include <vector>
#include <mutex>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <Eigen/Core>
#include <atlimage.h>
#include <windows.h>
#include "logging.h"
#include <string>
#include <fstream>
#include <ScreenGrab.h>
#include <wincodec.h>
#include <cmath>

using Microsoft::WRL::ComPtr;
using std::unique_ptr;
using std::vector;
using std::mutex;
using std::condition_variable;
using std::unique_lock;
using std::swap;
using Eigen::Matrix4f;
using std::chrono::high_resolution_clock;
using std::chrono::time_point;
using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::milliseconds;
static ComPtr<ID3D11Device> lastDev;
static ComPtr<ID3D11DeviceContext> lastCtx;
static ComPtr<ID3D11Texture2D> depthRes;
static ComPtr<ID3D11Texture2D> colorRes;
static ComPtr<ID3D11Buffer> constantBuf;
static ComPtr<ID3D11Texture2D> backBuf;
static vector<unsigned char> depthBuf;
static vector<unsigned char> colorBuf;
static vector<unsigned char> stencilBuf;
static rage_matrices constants;
static int lastColorWidth = 0;
static int lastColorHeight = 0;
static int lastDepthWidth = 0;
static int lastDepthHeight = 0;
static bool request_copy = false;
static mutex copy_mtx;
static condition_variable copy_cv;
static HRESULT screenHr;
static time_point<high_resolution_clock> last_depth_time;
static time_point<high_resolution_clock> last_color_time;
static time_point<high_resolution_clock> last_constant_time;
static time_point<high_resolution_clock> last_screen_time;
static std::chrono::milliseconds capScreen;

float g_rgbdDownsampleFactor = 0.3f;

void writeLog(std::string msg)
{
    LOGI("export", msg);
}

static void unpack_depth(ID3D11Device* dev, ID3D11DeviceContext* ctx, ID3D11Resource* src, vector<unsigned char>& dst, vector<unsigned char>& stencil)
{
	if (dev == nullptr || ctx == nullptr || src == nullptr) {
		throw std::invalid_argument("unpack_depth received null device/context/resource");
	}
	HRESULT hr = S_OK;

	ComPtr<ID3D11Texture2D> src_tex;
	
	hr = src->QueryInterface(__uuidof(ID3D11Texture2D), &src_tex);
	if (hr != S_OK) throw std::system_error(hr, std::system_category());
	D3D11_TEXTURE2D_DESC src_desc;

    src_tex->GetDesc(&src_desc);
    int inW = (int)src_desc.Width;
    int inH = (int)src_desc.Height;
    float s = g_rgbdDownsampleFactor;
    if (s <= 0.0f || s > 1.0f) s = 1.0f;
    int outW = (std::max)(1, (int)lroundf(inW * s));
    int outH = (std::max)(1, (int)lroundf(inH * s));
    lastDepthWidth = outW;
    lastDepthHeight = outH;
	// assert(src_desc.Format == DXGI_FORMAT_R32G8X24_TYPELESS);
	D3D11_MAPPED_SUBRESOURCE src_map = { 0 };
	hr = ctx->Map(src, 0, D3D11_MAP_READ, 0, &src_map);
	if (hr != S_OK) throw std::system_error(hr, std::system_category());
	if (dst.size() != (size_t)outH * (size_t)outW * 4) dst = vector<unsigned char>((size_t)outH * (size_t)outW * 4);
	if (stencil.size() != (size_t)outH * (size_t)outW) stencil = vector<unsigned char>((size_t)outH * (size_t)outW);
	for (int y = 0; y < outH; ++y)
	{
		int sy = (std::min)((int)floorf(y / s), inH - 1);
		for (int x = 0; x < outW; ++x)
		{
			int sx = (std::min)((int)floorf(x / s), inW - 1);
			const float* src_f = (const float*)((const char*)src_map.pData + src_map.RowPitch * sy + (sx * 8));
			unsigned char* dst_p = &dst[outW * 4 * y + (x * 4)];
			unsigned char* stencil_p = &stencil[outW * y + x];
			memmove(dst_p, src_f, 4);
			memmove(stencil_p, src_f + 1, 1);
		}
	}
	ctx->Unmap(src, 0);
}
static ComPtr<ID3D11Texture2D> CreateTexHelper(ID3D11Device* dev, DXGI_FORMAT fmt, int width, int height, int samples)
{
	if (dev == nullptr) {
		throw std::invalid_argument("CreateTexHelper received null device");
	}
	D3D11_TEXTURE2D_DESC desc = { 0 };
	desc.Format = fmt;
	desc.ArraySize = 1;
	desc.BindFlags = 0;
	desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
	desc.Height = height;
	desc.Width = width;
	desc.MipLevels = 1;
	desc.MiscFlags = 0;
	desc.SampleDesc.Count = 1;
	desc.SampleDesc.Quality = 0;
	desc.Usage = D3D11_USAGE_STAGING;
	ComPtr<ID3D11Texture2D> result;
	HRESULT hr = S_OK;
	hr = dev->CreateTexture2D(&desc, nullptr, &result);
	if (hr != S_OK) throw std::system_error(hr, std::system_category());
	return result;

}
static ComPtr<ID3D11Buffer> CreateStagingBuffer(ID3D11Device* dev, int size) {
	if (dev == nullptr || size <= 0) {
		throw std::invalid_argument("CreateStagingBuffer received invalid device or size");
	}
	ComPtr<ID3D11Buffer> result;
	D3D11_BUFFER_DESC desc = { 0 };
	desc.BindFlags = 0;
	desc.ByteWidth = size;
	desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
	desc.MiscFlags = 0;
	desc.Usage = D3D11_USAGE_STAGING;
	HRESULT hr = dev->CreateBuffer(&desc, nullptr, &result);
	if (FAILED(hr) || result == nullptr) throw std::system_error(hr, std::system_category());
	return result;
}

void CreateTextureIfNeeded(ID3D11Device* dev, ID3D11Resource* for_res, ComPtr<ID3D11Texture2D>* tex_target)
{
	if (dev == nullptr || for_res == nullptr || tex_target == nullptr) {
		throw std::invalid_argument("CreateTextureIfNeeded received null argument");
	}
	ComPtr<ID3D11Texture2D> tex;
	HRESULT hr = S_OK;
	D3D11_TEXTURE2D_DESC desc = { 0 };
	hr = for_res->QueryInterface(__uuidof(ID3D11Texture2D), &tex);
	if (hr != S_OK) throw std::system_error(hr, std::system_category());
	tex->GetDesc(&desc);
	if(*tex_target == nullptr)
	{
		*tex_target = CreateTexHelper(dev, desc.Format, desc.Width, desc.Height, desc.SampleDesc.Count);
	}
	D3D11_TEXTURE2D_DESC tex_desc;
	(*tex_target)->GetDesc(&tex_desc);
	if (tex_desc.Width != desc.Width || tex_desc.Height != desc.Height)
	{
		*tex_target = CreateTexHelper(dev, desc.Format, desc.Width, desc.Height, desc.SampleDesc.Count);
	}
}
int getBitsPerPixel(DXGI_FORMAT fmt)
{
	switch(fmt)
	{
	case DXGI_FORMAT_B8G8R8A8_UNORM:
		return 4;
	default:
		return -1;
	}
}
void copyTexToVector(ID3D11Device* dev, ID3D11DeviceContext* ctx, ID3D11Resource* res, vector<unsigned char>& buffer)
{
	if (dev == nullptr || ctx == nullptr || res == nullptr) {
		throw std::invalid_argument("copyTexToVector received null device/context/resource");
	}
	ComPtr<ID3D11Texture2D> tex;
	HRESULT hr;
	hr = res->QueryInterface(__uuidof(ID3D11Texture2D), &tex);
	D3D11_TEXTURE2D_DESC desc;
	if (hr != S_OK) throw std::system_error(hr, std::system_category());
    tex->GetDesc(&desc);
    int inW = (int)desc.Width;
    int inH = (int)desc.Height;
    float s = g_rgbdDownsampleFactor;
    if (s <= 0.0f || s > 1.0f) s = 1.0f;
    int outW = (std::max)(1, (int)lroundf(inW * s));
    int outH = (std::max)(1, (int)lroundf(inH * s));
    lastColorWidth = outW;
    lastColorHeight = outH;
	ComPtr<ID3D11Texture2D> tex_copy;
	CreateTextureIfNeeded(dev, tex.Get(), &tex_copy);
	ctx->CopyResource(tex_copy.Get(), tex.Get());
	D3D11_MAPPED_SUBRESOURCE map;
	auto bpp = getBitsPerPixel(desc.Format);
	if (bpp == -1) throw std::invalid_argument("unsupported resource type");
	hr = ctx->Map(tex_copy.Get(), 0, D3D11_MAP_READ, 0, &map);
	if (hr != S_OK) throw std::system_error(hr, std::system_category());
	if (buffer.size() != (size_t)outH * (size_t)outW * bpp) buffer = vector<unsigned char>((size_t)outH * (size_t)outW * bpp);
	for(int y = 0; y < outH; ++y)
	{
		int sy = (std::min)((int)floorf(y / s), inH - 1);
		for (int x = 0; x < outW; ++x) {
			int sx = (std::min)((int)floorf(x / s), inW - 1);
			unsigned char* p = &buffer[y * outW * bpp + (x * 4)];
			unsigned char* b = (unsigned char*)map.pData + map.RowPitch * sy + (sx * 4);
			p[0] = b[2];
			p[1] = b[1];
			p[2] = b[0];
			p[3] = b[3];
			//memmove(p, b, desc.Width * bpp);
		}

	}
	ctx->Unmap(tex_copy.Get(), 0);
	
}
void CopyIfRequested()
{
	unique_lock<mutex> lk(copy_mtx);
	if(request_copy)
	{
		unpack_depth(lastDev.Get(), lastCtx.Get(), depthRes.Get(), depthBuf, stencilBuf);
		copyTexToVector(lastDev.Get(), lastCtx.Get(), colorRes.Get(), colorBuf);
		request_copy = false;
		lk.unlock(); //unlock the mutex so that and woken threads don't immediately block on it
		copy_cv.notify_all();
	}
}

void ExtractDepthBuffer(ID3D11Device* dev, ID3D11DeviceContext* ctx, ID3D11Resource* res)
{
	std::lock_guard<mutex> lk(copy_mtx);
	if (dev == nullptr || ctx == nullptr || res == nullptr) {
		LOGE("export", "ExtractDepthBuffer: invalid device/context/resource");
		return;
	}
	lastDev = dev;
	lastCtx = ctx;
    CreateTextureIfNeeded(dev, res, &depthRes);
    ctx->CopyResource(depthRes.Get(), res);
    last_depth_time = std::chrono::high_resolution_clock::now();
	//unpack_depth(dev, ctx, res, depthBuf, stencilBuf, screenBuf);
}

void ExtractColorBuffer(ID3D11Device* dev, ID3D11DeviceContext* ctx, ID3D11Resource* tex)
{
	std::lock_guard<mutex> lk(copy_mtx);
	if (dev == nullptr || ctx == nullptr || tex == nullptr) {
		LOGE("export", "ExtractColorBuffer: invalid device/context/resource");
		return;
	}
	lastDev = dev;
	lastCtx = ctx;
	CreateTextureIfNeeded(dev, tex, &colorRes);
	ctx->CopyResource(colorRes.Get(), tex);
	last_color_time = high_resolution_clock::now();
	//copyTexToVector(dev, ctx, tex, colorBuf);
	
}

void ExtractConstantBuffer(ID3D11Device* dev, ID3D11DeviceContext* ctx, ID3D11Buffer* buf) {
	std::lock_guard<mutex> lk(copy_mtx);
	if (dev == nullptr || ctx == nullptr || buf == nullptr) {
		LOGE("export", "ExtractConstantBuffer: invalid device/context/buffer");
		return;
	}
	lastDev = dev;
	lastCtx = ctx;
	D3D11_BUFFER_DESC desc = { 0 };
	buf->GetDesc(&desc);
	if (constantBuf == nullptr) {
		constantBuf = CreateStagingBuffer(dev, desc.ByteWidth);
	}
	ctx->CopyResource(constantBuf.Get(), buf);
	last_constant_time = high_resolution_clock::now();
	
}


//auto screenCap = [](int id) {
//	WCHAR imgPath[35];
//	swprintf(imgPath, 35, L"screen_%d.bmp", id);
//	if (!(export_get_screen_buffer(imgPath, 0))) {
//	}
//	else {
//		std::wstring ws1(imgPath);
//	}
//};

void ExtractScreenBuffer(ID3D11DeviceContext* ctx, ID3D11Texture2D* back, HRESULT hr)
{
	std::lock_guard<mutex> lk(copy_mtx);
	if (ctx == nullptr || back == nullptr) {
		LOGE("export", "ExtractScreenBuffer: invalid context/back buffer");
		return;
	}
	lastCtx = ctx;
	backBuf = back;
	screenHr = hr;
	last_screen_time = high_resolution_clock::now();
}

extern "C" {
	__declspec(dllexport) int export_get_depth_buffer(void** buf)
	{
		try {
			if (buf == nullptr) {
				LOGE("export", "export_get_depth_buffer: output pointer is null");
				return -1;
			}
			std::lock_guard<mutex> lk(copy_mtx);
			if (lastDev == nullptr || lastCtx == nullptr || depthRes == nullptr) {
				LOGE("export", "export_get_depth_buffer: Invalid device/context/resource pointers");
				return -1;
			}
			unpack_depth(lastDev.Get(), lastCtx.Get(), depthRes.Get(), depthBuf, stencilBuf);
			if (depthBuf.empty()) {
				LOGE("export", "export_get_depth_buffer: Depth buffer is empty after unpack");
				return -1;
			}
			*buf = &depthBuf[0];
			return depthBuf.size();
		} catch (const std::system_error& e) {
			LOGE("export", std::string("export_get_depth_buffer: System error - ") + e.what());
			return -1;
		} catch (const std::exception& e) {
			LOGE("export", std::string("export_get_depth_buffer: Exception - ") + e.what());
			return -1;
		} catch (...) {
			LOGE("export", "export_get_depth_buffer: Unknown exception");
			return -1;
		}
    }
	__declspec(dllexport) int export_get_color_buffer(void** buf)
	{
		try {
			if (buf == nullptr) {
				LOGE("export", "export_get_color_buffer: output pointer is null");
				return -1;
			}
			std::lock_guard<mutex> lk(copy_mtx);
			if (lastDev == nullptr || lastCtx == nullptr || colorRes == nullptr) {
				LOGE("export", "export_get_color_buffer: Invalid device/context/resource pointers");
				return -1;
			}
			copyTexToVector(lastDev.Get(), lastCtx.Get(), colorRes.Get(), colorBuf);
			if (colorBuf.empty()) {
				LOGE("export", "export_get_color_buffer: Color buffer is empty after copy");
				return -1;
			}
			*buf = &colorBuf[0];
			return colorBuf.size();
		} catch (const std::system_error& e) {
			LOGE("export", std::string("export_get_color_buffer: System error - ") + e.what());
			return -1;
		} catch (const std::invalid_argument& e) {
			LOGE("export", std::string("export_get_color_buffer: Invalid argument - ") + e.what());
			return -1;
		} catch (const std::exception& e) {
			LOGE("export", std::string("export_get_color_buffer: Exception - ") + e.what());
			return -1;
		} catch (...) {
			LOGE("export", "export_get_color_buffer: Unknown exception");
			return -1;
		}
    }
	__declspec(dllexport) int export_get_stencil_buffer(void** buf)
	{
		try {
			if (buf == nullptr) {
				LOGE("export", "export_get_stencil_buffer: output pointer is null");
				return -1;
			}
			std::lock_guard<mutex> lk(copy_mtx);
			if (lastDev == nullptr || lastCtx == nullptr || depthRes == nullptr) {
				LOGE("export", "export_get_stencil_buffer: Invalid device/context/resource pointers");
				return -1;
			}
			unpack_depth(lastDev.Get(), lastCtx.Get(), depthRes.Get(), depthBuf, stencilBuf);
			if (stencilBuf.empty()) {
				LOGE("export", "export_get_stencil_buffer: Stencil buffer is empty after unpack");
				return -1;
			}
			*buf = &stencilBuf[0];
			return stencilBuf.size();
		} catch (const std::system_error& e) {
			LOGE("export", std::string("export_get_stencil_buffer: System error - ") + e.what());
			return -1;
		} catch (const std::exception& e) {
			LOGE("export", std::string("export_get_stencil_buffer: Exception - ") + e.what());
			return -1;
		} catch (...) {
			LOGE("export", "export_get_stencil_buffer: Unknown exception");
			return -1;
		}
    }
	__declspec(dllexport) int export_get_constant_buffer(rage_matrices* buf) {
		try {
			if (buf == nullptr) {
				LOGE("export", "export_get_constant_buffer: output pointer is null");
				return -1;
			}
			std::lock_guard<mutex> lk(copy_mtx);
			if (constantBuf == nullptr) {
				LOGE("export", "export_get_constant_buffer: Constant buffer is null");
				return -1;
			}
			if (lastCtx == nullptr) {
				LOGE("export", "export_get_constant_buffer: Device context is null");
				return -1;
			}
			D3D11_MAPPED_SUBRESOURCE res = { 0 };
			HRESULT hr = lastCtx->Map(constantBuf.Get(), 0, D3D11_MAP_READ, 0, &res);
			if (FAILED(hr)) {
				LOGE("export", std::string("export_get_constant_buffer: Failed to map constant buffer, HRESULT: ") + std::to_string(hr));
				return -1;
			}
			if (res.pData == nullptr) {
				LOGE("export", "export_get_constant_buffer: Mapped data pointer is null");
				lastCtx->Unmap(constantBuf.Get(), 0);
				return -1;
			}
			memmove(buf, res.pData, sizeof(constants));
			lastCtx->Unmap(constantBuf.Get(), 0);
			return sizeof(rage_matrices);
		} catch (const std::exception& e) {
			LOGE("export", std::string("export_get_constant_buffer: Exception - ") + e.what());
			return -1;
		} catch (...) {
			LOGE("export", "export_get_constant_buffer: Unknown exception");
			return -1;
		}
	}
	__declspec(dllexport) int export_get_screen_buffer(WCHAR *pictureName)
	{
		try {
			if (pictureName == nullptr) {
				LOGE("export", "export_get_screen_buffer: pictureName is null");
				return 0;
			}
			std::lock_guard<mutex> lk(copy_mtx);
			if (lastCtx == nullptr || backBuf == nullptr || !SUCCEEDED(screenHr)) {
				LOGE("export", "export_get_screen_buffer: Invalid context/buffer or screen HR failed");
				return 0;
			}
			HRESULT hr = DirectX::SaveWICTextureToFile(lastCtx.Get(), backBuf.Get(),
				GUID_ContainerFormatBmp, pictureName);
			if (SUCCEEDED(hr)) {
				return 1;
			} else {
				LOGE("export", std::string("export_get_screen_buffer: Failed to save texture, HRESULT: ") + std::to_string(hr));
				return 2;
			}
		} catch (const std::exception& e) {
			LOGE("export", std::string("export_get_screen_buffer: Exception - ") + e.what());
			return 0;
		} catch (...) {
			LOGE("export", "export_get_screen_buffer: Unknown exception");
			return 0;
		}
	}

	__declspec(dllexport) long long int export_get_last_depth_time() {
		return duration_cast<milliseconds>(last_depth_time.time_since_epoch()).count();
	}
	__declspec(dllexport) long long int export_get_last_color_time() {
		return duration_cast<milliseconds>(last_color_time.time_since_epoch()).count();
	}
	__declspec(dllexport) long long int export_get_last_constant_time() {
		return duration_cast<milliseconds>(last_constant_time.time_since_epoch()).count();
	}

    __declspec(dllexport) long long int export_get_current_time() {
        return duration_cast<milliseconds>(high_resolution_clock::now().time_since_epoch()).count();
    }

    __declspec(dllexport) int export_get_last_color_width() { return lastColorWidth; }
    __declspec(dllexport) int export_get_last_color_height() { return lastColorHeight; }
    __declspec(dllexport) int export_get_last_depth_width() { return lastDepthWidth; }
    __declspec(dllexport) int export_get_last_depth_height() { return lastDepthHeight; }
}
