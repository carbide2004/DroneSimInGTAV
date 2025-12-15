#include "server_v2.h"
#include "utils.h"
#include "camera.h"
#include "export.h"
#include <thread>
#include <cstring>

using namespace boost;

static std::unique_ptr<ServerV2> g_serverV2Instance;
static asio::io_context g_io_v2;
static std::unique_ptr<std::thread> g_thread_v2;
static bool g_ws_inited_v2 = false;

extern std::queue<std::string> g_cmdQueue;
extern catchState cmdToCatch;

ServerV2::ServerV2(boost::asio::io_context& io, unsigned short port)
    : acceptor_(io, asio::ip::tcp::endpoint(asio::ip::tcp::v4(), port)), socket_(io) {
    start_accept();
}

void ServerV2::start_accept() {
    acceptor_.async_accept(socket_, [this](const system::error_code& ec) {
        if (!ec) {
            handle_client();
        } else {
            start_accept();
        }
    });
}

bool ServerV2::read_exact(asio::ip::tcp::socket& s, void* buf, size_t len) {
    size_t total = 0;
    unsigned char* p = static_cast<unsigned char*>(buf);
    while (total < len) {
        system::error_code ec;
        size_t n = s.read_some(asio::buffer(p + total, len - total), ec);
        if (ec) return false;
        total += n;
    }
    return true;
}

void ServerV2::handle_client() {
    MsgHeader hdr{};
    if (!read_exact(socket_, &hdr.magic[0], 4)) { socket_.close(); start_accept(); return; }
    if (!read_exact(socket_, &hdr.version, 1)) { socket_.close(); start_accept(); return; }
    if (!read_exact(socket_, &hdr.type, 1)) { socket_.close(); start_accept(); return; }
    if (!read_exact(socket_, &hdr.flags, 1)) { socket_.close(); start_accept(); return; }
    if (!read_exact(socket_, &hdr.reserved, 1)) { socket_.close(); start_accept(); return; }
    if (!read_exact(socket_, &hdr.request_id, 8)) { socket_.close(); start_accept(); return; }
    if (!read_exact(socket_, &hdr.length, 4)) { socket_.close(); start_accept(); return; }
    if (std::memcmp(hdr.magic, "DSV2", 4) != 0) { socket_.close(); start_accept(); return; }
    std::vector<unsigned char> payload(hdr.length);
    if (hdr.length) {
        if (!read_exact(socket_, payload.data(), hdr.length)) { socket_.close(); start_accept(); return; }
    }

    std::vector<unsigned char> resp;
    switch (hdr.type) {
        case MSG_CREATE_CAMERA: {
            g_cmdQueue.push("CREATE_CAMERA");
            uint64_t cam_id = 1;
            MsgHeader rh{}; std::memcpy(rh.magic, "DSV2", 4); rh.version = hdr.version; rh.type = MSG_CREATE_CAMERA; rh.flags = 0; rh.reserved = 0; rh.request_id = hdr.request_id; rh.length = sizeof(cam_id);
            resp.resize(sizeof(rh) + sizeof(cam_id));
            std::memcpy(resp.data(), &rh.magic[0], 4);
            std::memcpy(resp.data() + 4, &rh.version, 1);
            std::memcpy(resp.data() + 5, &rh.type, 1);
            std::memcpy(resp.data() + 6, &rh.flags, 1);
            std::memcpy(resp.data() + 7, &rh.reserved, 1);
            std::memcpy(resp.data() + 8, &rh.request_id, 8);
            std::memcpy(resp.data() + 16, &rh.length, 4);
            std::memcpy(resp.data() + 20, &cam_id, sizeof(cam_id));
            write_response(resp);
            return;
        }
        case MSG_MOVE: {
            if (hdr.length >= sizeof(float) * 3) {
                float dx = *reinterpret_cast<float*>(&payload[0]);
                float dy = *reinterpret_cast<float*>(&payload[4]);
                float dz = *reinterpret_cast<float*>(&payload[8]);
                float ax = std::abs(dx), ay = std::abs(dy), az = std::abs(dz);
                if (ax >= ay && ax >= az) g_cmdQueue.push(dx >= 0 ? "FORWARD" : "BACKWARD");
                else if (ay >= ax && ay >= az) g_cmdQueue.push(dy >= 0 ? "RIGHT" : "LEFT");
                else g_cmdQueue.push(dz >= 0 ? "UP" : "DOWN");
            }
            MsgHeader rh{}; std::memcpy(rh.magic, "DSV2", 4); rh.version = hdr.version; rh.type = MSG_MOVE; rh.flags = 0; rh.reserved = 0; rh.request_id = hdr.request_id; rh.length = 0;
            resp.resize(sizeof(rh));
            std::memcpy(resp.data(), &rh.magic[0], 4);
            std::memcpy(resp.data() + 4, &rh.version, 1);
            std::memcpy(resp.data() + 5, &rh.type, 1);
            std::memcpy(resp.data() + 6, &rh.flags, 1);
            std::memcpy(resp.data() + 7, &rh.reserved, 1);
            std::memcpy(resp.data() + 8, &rh.request_id, 8);
            std::memcpy(resp.data() + 16, &rh.length, 4);
            write_response(resp);
            return;
        }
        case MSG_ROTATE: {
            if (hdr.length >= sizeof(float) * 3) {
                float rz = *reinterpret_cast<float*>(&payload[8]);
                if (rz > 0) g_cmdQueue.push("LEFTROTATE"); else if (rz < 0) g_cmdQueue.push("RIGHTROTATE");
            }
            MsgHeader rh{}; std::memcpy(rh.magic, "DSV2", 4); rh.version = hdr.version; rh.type = MSG_ROTATE; rh.flags = 0; rh.reserved = 0; rh.request_id = hdr.request_id; rh.length = 0;
            resp.resize(sizeof(rh));
            std::memcpy(resp.data(), &rh.magic[0], 4);
            std::memcpy(resp.data() + 4, &rh.version, 1);
            std::memcpy(resp.data() + 5, &rh.type, 1);
            std::memcpy(resp.data() + 6, &rh.flags, 1);
            std::memcpy(resp.data() + 7, &rh.reserved, 1);
            std::memcpy(resp.data() + 8, &rh.request_id, 8);
            std::memcpy(resp.data() + 16, &rh.length, 4);
            write_response(resp);
            return;
        }
        case MSG_SET_FOV: {
            if (hdr.length >= sizeof(float)) {
                float fov = *reinterpret_cast<float*>(&payload[0]);
                std::string s = std::string("SETFOV:") + std::to_string(fov);
                g_cmdQueue.push(s);
            }
            MsgHeader rh{}; std::memcpy(rh.magic, "DSV2", 4); rh.version = hdr.version; rh.type = MSG_SET_FOV; rh.flags = 0; rh.reserved = 0; rh.request_id = hdr.request_id; rh.length = 0;
            resp.resize(sizeof(rh));
            std::memcpy(resp.data(), &rh.magic[0], 4);
            std::memcpy(resp.data() + 4, &rh.version, 1);
            std::memcpy(resp.data() + 5, &rh.type, 1);
            std::memcpy(resp.data() + 6, &rh.flags, 1);
            std::memcpy(resp.data() + 7, &rh.reserved, 1);
            std::memcpy(resp.data() + 8, &rh.request_id, 8);
            std::memcpy(resp.data() + 16, &rh.length, 4);
            write_response(resp);
            return;
        }
        case MSG_CAPTURE: {
            g_cmdQueue.push("REQUEST");
            int tries = 0;
            while (cmdToCatch != catchStop && tries < 3000) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                tries++;
            }
            void* rgb_ptr = nullptr; int rgb_size = export_get_color_buffer(&rgb_ptr);
            void* depth_ptr = nullptr; int depth_size = export_get_depth_buffer(&depth_ptr);
            int w_rgb = export_get_last_color_width();
            int h_rgb = export_get_last_color_height();
            int w_depth = export_get_last_depth_width();
            int h_depth = export_get_last_depth_height();
            uint32_t hdr_len = sizeof(uint32_t) * 4 + rgb_size + depth_size;
            MsgHeader rh{}; std::memcpy(rh.magic, "DSV2", 4); rh.version = hdr.version; rh.type = MSG_CAPTURE; rh.flags = 0; rh.reserved = 0; rh.request_id = hdr.request_id; rh.length = hdr_len;
            resp.resize(sizeof(rh) + hdr_len);
            std::memcpy(resp.data(), &rh.magic[0], 4);
            std::memcpy(resp.data() + 4, &rh.version, 1);
            std::memcpy(resp.data() + 5, &rh.type, 1);
            std::memcpy(resp.data() + 6, &rh.flags, 1);
            std::memcpy(resp.data() + 7, &rh.reserved, 1);
            std::memcpy(resp.data() + 8, &rh.request_id, 8);
            std::memcpy(resp.data() + 16, &rh.length, 4);
            uint32_t* p32 = reinterpret_cast<uint32_t*>(resp.data() + 20);
            p32[0] = static_cast<uint32_t>(rgb_size);
            p32[1] = static_cast<uint32_t>(depth_size);
            p32[2] = static_cast<uint32_t>(w_rgb);
            p32[3] = static_cast<uint32_t>(h_rgb);
            unsigned char* p = resp.data() + 20 + sizeof(uint32_t) * 4;
            if (rgb_size > 0) std::memcpy(p, rgb_ptr, rgb_size);
            p += rgb_size;
            if (depth_size > 0) std::memcpy(p, depth_ptr, depth_size);
            write_response(resp);
            return;
        }
        default: {
            socket_.close(); start_accept(); return;
        }
    }
}

void ServerV2::write_response(const std::vector<unsigned char>& data) {
    system::error_code ec;
    asio::write(socket_, asio::buffer(data), ec);
    socket_.close();
    start_accept();
}

void InitializeServerV2() {
    if (!g_ws_inited_v2) {
        WSADATA wsaData; int r = WSAStartup(MAKEWORD(2,2), &wsaData); if (r == 0) g_ws_inited_v2 = true;
    }
    if (g_serverV2Instance) return;
    g_thread_v2 = std::make_unique<std::thread>([](){
        try {
            g_serverV2Instance = std::make_unique<ServerV2>(g_io_v2, 23456);
            g_io_v2.run();
        } catch (...) {}
    });
    g_thread_v2->detach();
}

void ShutdownServerV2() {
    g_io_v2.stop();
    g_serverV2Instance.reset();
    g_thread_v2.reset();
}
