#include "server_v2.h"
#include "utils.h"
#include "camera.h"
#include "export.h"
#include "logging.h"
#include "command_queue.h"
#include <thread>
#include <cstring>
#include <algorithm>
#include <atomic>

using namespace boost;

static constexpr size_t kWireHeaderSize = 20;

static std::unique_ptr<ServerV2> g_serverV2Instance;
static asio::io_context g_io_v2;
static std::unique_ptr<std::thread> g_thread_v2;
static bool g_ws_inited_v2 = false;

extern volatile catchState cmdToCatch;

extern std::atomic<bool> g_poseReady;
extern float g_pose[6];

extern std::atomic<bool> g_accidentReady;
extern float g_accidentPos[3];

extern std::atomic<bool> g_recordingEnabled;
extern std::atomic<int> g_recordingStep;
extern char g_recordingSessionDir[260];
extern char g_recordingRequestedSession[128];
extern char g_recordingRequestedTask[256];

extern std::atomic<bool> g_fireReady;
extern float g_firePos[3];
extern int g_fireId;

extern std::atomic<bool> g_fightReady;
extern float g_fightPos[3];


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
    try {
        MsgHeader hdr{};
        if (!read_exact(socket_, &hdr.magic[0], 4)) { 
            LOGE("server_v2", "Failed to read magic bytes from client");
            socket_.close(); start_accept(); return; 
        }
        if (!read_exact(socket_, &hdr.version, 1)) { 
            LOGE("server_v2", "Failed to read version from client");
            socket_.close(); start_accept(); return; 
        }
        if (!read_exact(socket_, &hdr.type, 1)) { 
            LOGE("server_v2", "Failed to read message type from client");
            socket_.close(); start_accept(); return; 
        }
        if (!read_exact(socket_, &hdr.flags, 1)) { 
            LOGE("server_v2", "Failed to read flags from client");
            socket_.close(); start_accept(); return; 
        }
        if (!read_exact(socket_, &hdr.reserved, 1)) { 
            LOGE("server_v2", "Failed to read reserved field from client");
            socket_.close(); start_accept(); return; 
        }
        if (!read_exact(socket_, &hdr.request_id, 8)) { 
            LOGE("server_v2", "Failed to read request_id from client");
            socket_.close(); start_accept(); return; 
        }
        if (!read_exact(socket_, &hdr.length, 4)) { 
            LOGE("server_v2", "Failed to read length from client");
            socket_.close(); start_accept(); return; 
        }
        
        if (std::memcmp(hdr.magic, "DSV2", 4) != 0) { 
            LOGE("server_v2", "Invalid magic bytes received");
            socket_.close(); start_accept(); return; 
        }
        
        // 验证payload长度的合理性 (RGBD数据最大约30MB)
        if (hdr.length > 50 * 1024 * 1024) { // 50MB限制，为RGBD数据预留足够空间
            LOGE("server_v2", std::string("Payload too large: ") + std::to_string(hdr.length) + " bytes");
            socket_.close(); start_accept(); return;
        }
        
        std::vector<unsigned char> payload;
        if (hdr.length > 0) {
            payload.resize(hdr.length);
            if (!read_exact(socket_, payload.data(), hdr.length)) { 
                LOGE("server_v2", std::string("Failed to read payload of ") + std::to_string(hdr.length) + " bytes");
                socket_.close(); start_accept(); return; 
            }
        }
        
        LOGD("server_v2", std::string("Received message type ") + std::to_string(hdr.type) + " with payload " + std::to_string(hdr.length) + " bytes");

    std::vector<unsigned char> resp;
    switch (hdr.type) {
        case MSG_CREATE_CAMERA: {
            enqueue_command("CREATE_CAMERA");
            uint64_t cam_id = 1;
            MsgHeader rh{}; std::memcpy(rh.magic, "DSV2", 4); rh.version = hdr.version; rh.type = MSG_CREATE_CAMERA; rh.flags = 0; rh.reserved = 0; rh.request_id = hdr.request_id; rh.length = sizeof(cam_id);
            resp.resize(kWireHeaderSize + sizeof(cam_id));
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
            if (hdr.length >= sizeof(float) * 3 && payload.size() >= sizeof(float) * 3) {
                float dx = *reinterpret_cast<float*>(&payload[0]);
                float dy = *reinterpret_cast<float*>(&payload[4]);
                float dz = *reinterpret_cast<float*>(&payload[8]);
                
                // 验证浮点数的有效性
                if (std::isfinite(dx) && std::isfinite(dy) && std::isfinite(dz)) {
                    std::string s = std::string("MOVE ") + std::to_string(dx) + " " + std::to_string(dy) + " " + std::to_string(dz);
                    enqueue_command(s);
                    LOGD("server_v2", std::string("Enqueue ") + s);
                } else {
                    LOGE("server_v2", "Invalid float values in MOVE command");
                }
            } else {
                LOGE("server_v2", std::string("MSG_MOVE: Invalid payload size. Expected: ") + std::to_string(sizeof(float) * 3) + ", Got: " + std::to_string(hdr.length));
            }
            MsgHeader rh{}; std::memcpy(rh.magic, "DSV2", 4); rh.version = hdr.version; rh.type = MSG_MOVE; rh.flags = 0; rh.reserved = 0; rh.request_id = hdr.request_id; rh.length = 0;
            resp.resize(kWireHeaderSize);
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
                float rx = *reinterpret_cast<float*>(&payload[0]);
                float ry = *reinterpret_cast<float*>(&payload[4]);
                float rz = *reinterpret_cast<float*>(&payload[8]);
                std::string s = std::string("ROTATE ") + std::to_string(rx) + " " + std::to_string(ry) + " " + std::to_string(rz);
                enqueue_command(s);
                LOGD("server_v2", std::string("Enqueue ") + s);
            }
            MsgHeader rh{}; std::memcpy(rh.magic, "DSV2", 4); rh.version = hdr.version; rh.type = MSG_ROTATE; rh.flags = 0; rh.reserved = 0; rh.request_id = hdr.request_id; rh.length = 0;
            resp.resize(kWireHeaderSize);
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
                enqueue_command(s);
            }
            MsgHeader rh{}; std::memcpy(rh.magic, "DSV2", 4); rh.version = hdr.version; rh.type = MSG_SET_FOV; rh.flags = 0; rh.reserved = 0; rh.request_id = hdr.request_id; rh.length = 0;
            resp.resize(kWireHeaderSize);
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
        case MSG_GET_POSE: {
            // 确保每次请求都有干净的状态
            g_poseReady.store(false, std::memory_order_release);
            enqueue_command("GET_POSE");
            LOGD("server_v2", std::string("GET_POSE enqueued"));
            
            // 增加超时时间，给GTA V API更多响应时间
            int tries = 0;
            const int max_tries = 600; // 从300增加到600 (3秒)
            while (!g_poseReady.load(std::memory_order_acquire) && tries < max_tries) { 
                std::this_thread::sleep_for(std::chrono::milliseconds(5)); 
                tries++; 
                
                // 每500ms记录一次等待状态，帮助调试
                if (tries % 100 == 0) {
                    LOGD("server_v2", std::string("GET_POSE waiting... tries: ") + std::to_string(tries) + "/" + std::to_string(max_tries));
                }
            }
            
            MsgHeader rh{}; 
            std::memcpy(rh.magic, "DSV2", 4); 
            rh.version = hdr.version; 
            rh.type = MSG_GET_POSE; 
            rh.flags = 0; 
            rh.reserved = 0; 
            rh.request_id = hdr.request_id;
            
            bool pose_ready = g_poseReady.load(std::memory_order_acquire);
            if (!pose_ready) {
                rh.length = 0;
                resp.resize(kWireHeaderSize);
                std::memcpy(resp.data(), &rh.magic[0], 4);
                std::memcpy(resp.data() + 4, &rh.version, 1);
                std::memcpy(resp.data() + 5, &rh.type, 1);
                std::memcpy(resp.data() + 6, &rh.flags, 1);
                std::memcpy(resp.data() + 7, &rh.reserved, 1);
                std::memcpy(resp.data() + 8, &rh.request_id, 8);
                std::memcpy(resp.data() + 16, &rh.length, 4);
                LOGW("server_v2", std::string("GET_POSE timeout after ") + std::to_string(tries * 5) + "ms - camera mode may be inactive or GTA V API blocked");
            } else {
                rh.length = sizeof(float)*6;
                resp.resize(kWireHeaderSize + sizeof(float)*6);
                std::memcpy(resp.data(), &rh.magic[0], 4);
                std::memcpy(resp.data() + 4, &rh.version, 1);
                std::memcpy(resp.data() + 5, &rh.type, 1);
                std::memcpy(resp.data() + 6, &rh.flags, 1);
                std::memcpy(resp.data() + 7, &rh.reserved, 1);
                std::memcpy(resp.data() + 8, &rh.request_id, 8);
                std::memcpy(resp.data() + 16, &rh.length, 4);
                std::memcpy(resp.data() + 20, &g_pose[0], sizeof(float)*6);
                LOGD("server_v2", std::string("GET_POSE completed in ") + std::to_string(tries * 5) + "ms: " + std::to_string(g_pose[0]) + "," + std::to_string(g_pose[1]) + "," + std::to_string(g_pose[2]));
            }
            write_response(resp);
            return;
        }
        case MSG_SET_TIME: {
            if (hdr.length >= 12) {
                int h = *reinterpret_cast<int*>(&payload[0]);
                int m = *reinterpret_cast<int*>(&payload[4]);
                int s = *reinterpret_cast<int*>(&payload[8]);
                std::string sCmd = std::string("SET_TIME ") + std::to_string(h) + " " + std::to_string(m) + " " + std::to_string(s);
                enqueue_command(sCmd);
            }
            MsgHeader rh{}; std::memcpy(rh.magic, "DSV2", 4); rh.version = hdr.version; rh.type = MSG_SET_TIME; rh.flags = 0; rh.reserved = 0; rh.request_id = hdr.request_id; rh.length = 0;
            resp.resize(kWireHeaderSize);
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
        case MSG_SET_WEATHER: {
            std::string name;
            if (hdr.length > 0) {
                name.assign(reinterpret_cast<char*>(payload.data()), hdr.length);
            }
            std::string sCmd = std::string("SET_WEATHER ") + name;
            enqueue_command(sCmd);
            MsgHeader rh{}; std::memcpy(rh.magic, "DSV2", 4); rh.version = hdr.version; rh.type = MSG_SET_WEATHER; rh.flags = 0; rh.reserved = 0; rh.request_id = hdr.request_id; rh.length = 0;
            resp.resize(kWireHeaderSize);
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
        case MSG_STOP_CAMERA: {
            enqueue_command("STOP_CAMERA");
            MsgHeader rh{}; std::memcpy(rh.magic, "DSV2", 4); rh.version = hdr.version; rh.type = MSG_STOP_CAMERA; rh.flags = 0; rh.reserved = 0; rh.request_id = hdr.request_id; rh.length = 0;
            resp.resize(kWireHeaderSize);
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
        case MSG_CREATE_ACCIDENT: {
            g_accidentReady.store(false, std::memory_order_release);
            enqueue_command("CREATE_ACCIDENT");
            LOGD("server_v2", std::string("CREATE_ACCIDENT enqueued"));
            int tries = 0;
            // Wait for up to 20 seconds for the accident to be set up and collision detected
            while (!g_accidentReady.load(std::memory_order_acquire) && tries < 4000) { std::this_thread::sleep_for(std::chrono::milliseconds(5)); tries++; }
            
            MsgHeader rh{}; std::memcpy(rh.magic, "DSV2", 4); rh.version = hdr.version; rh.type = MSG_CREATE_ACCIDENT; rh.flags = 0; rh.reserved = 0; rh.request_id = hdr.request_id;
            
            bool accident_ready = g_accidentReady.load(std::memory_order_acquire);
            if (!accident_ready) {
                rh.length = 0;
                resp.resize(kWireHeaderSize);
                std::memcpy(resp.data(), &rh.magic[0], 4);
                std::memcpy(resp.data() + 4, &rh.version, 1);
                std::memcpy(resp.data() + 5, &rh.type, 1);
                std::memcpy(resp.data() + 6, &rh.flags, 1);
                std::memcpy(resp.data() + 7, &rh.reserved, 1);
                std::memcpy(resp.data() + 8, &rh.request_id, 8);
                std::memcpy(resp.data() + 16, &rh.length, 4);
                LOGW("server_v2", "CREATE_ACCIDENT timeout");
            } else {
                rh.length = sizeof(float) * 3;
                resp.resize(kWireHeaderSize + sizeof(float) * 3);
                std::memcpy(resp.data(), &rh.magic[0], 4);
                std::memcpy(resp.data() + 4, &rh.version, 1);
                std::memcpy(resp.data() + 5, &rh.type, 1);
                std::memcpy(resp.data() + 6, &rh.flags, 1);
                std::memcpy(resp.data() + 7, &rh.reserved, 1);
                std::memcpy(resp.data() + 8, &rh.request_id, 8);
                std::memcpy(resp.data() + 16, &rh.length, 4);
                std::memcpy(resp.data() + 20, &g_accidentPos[0], sizeof(float) * 3);
                LOGD("server_v2", std::string("CREATE_ACCIDENT ready: ") + std::to_string(g_accidentPos[0]) + "," + std::to_string(g_accidentPos[1]) + "," + std::to_string(g_accidentPos[2]));
            }
            write_response(resp);
            return;
        }
        case MSG_GET_RECORDING_INFO: {
            uint8_t enabled = g_recordingEnabled.load(std::memory_order_acquire) ? 1 : 0;
            int32_t step = static_cast<int32_t>(g_recordingStep.load(std::memory_order_acquire));
            uint16_t path_len = static_cast<uint16_t>(std::min<size_t>(std::strlen(g_recordingSessionDir), 65535));
            uint32_t payload_len = 1 + 4 + 2 + path_len;
            MsgHeader rh{}; std::memcpy(rh.magic, "DSV2", 4); rh.version = hdr.version; rh.type = MSG_GET_RECORDING_INFO; rh.flags = 0; rh.reserved = 0; rh.request_id = hdr.request_id; rh.length = payload_len;
            resp.resize(kWireHeaderSize + payload_len);
            std::memcpy(resp.data(), &rh.magic[0], 4);
            std::memcpy(resp.data() + 4, &rh.version, 1);
            std::memcpy(resp.data() + 5, &rh.type, 1);
            std::memcpy(resp.data() + 6, &rh.flags, 1);
            std::memcpy(resp.data() + 7, &rh.reserved, 1);
            std::memcpy(resp.data() + 8, &rh.request_id, 8);
            std::memcpy(resp.data() + 16, &rh.length, 4);
            unsigned char* p = resp.data() + 20;
            std::memcpy(p, &enabled, 1); p += 1;
            std::memcpy(p, &step, 4); p += 4;
            std::memcpy(p, &path_len, 2); p += 2;
            if (path_len) std::memcpy(p, g_recordingSessionDir, path_len);
            write_response(resp);
            return;
        }
        case MSG_SET_RECORDING_SESSION: {
            std::memset(g_recordingRequestedSession, 0, sizeof(g_recordingRequestedSession));
            std::memset(g_recordingRequestedTask, 0, sizeof(g_recordingRequestedTask));
            if (hdr.length > 0) {
                std::string s(reinterpret_cast<char*>(payload.data()), hdr.length);
                size_t p = s.find('\n');
                std::string session = (p == std::string::npos) ? s : s.substr(0, p);
                std::string task = (p == std::string::npos) ? std::string() : s.substr(p + 1);
                while (!task.empty() && (task.back() == '\n' || task.back() == '\r')) task.pop_back();
                if (!task.empty() && (task.rfind("task=", 0) == 0 || task.rfind("TASK=", 0) == 0)) task = task.substr(5);
                if (!task.empty() && (task.rfind("task:", 0) == 0 || task.rfind("TASK:", 0) == 0)) task = task.substr(5);

                size_t n = std::min<size_t>(session.size(), sizeof(g_recordingRequestedSession) - 1);
                std::memcpy(g_recordingRequestedSession, session.data(), n);
                g_recordingRequestedSession[n] = '\0';

                size_t tn = std::min<size_t>(task.size(), sizeof(g_recordingRequestedTask) - 1);
                std::memcpy(g_recordingRequestedTask, task.data(), tn);
                g_recordingRequestedTask[tn] = '\0';
            }
            MsgHeader rh{}; std::memcpy(rh.magic, "DSV2", 4); rh.version = hdr.version; rh.type = MSG_SET_RECORDING_SESSION; rh.flags = 0; rh.reserved = 0; rh.request_id = hdr.request_id; rh.length = 0;
            resp.resize(kWireHeaderSize);
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
        case MSG_CREATE_FIRE: {
            g_fireReady.store(false, std::memory_order_release);
            enqueue_command("CREATE_FIRE");
            int tries = 0;
            while (!g_fireReady.load(std::memory_order_acquire) && tries < 2000) { std::this_thread::sleep_for(std::chrono::milliseconds(5)); tries++; }

            MsgHeader rh{}; std::memcpy(rh.magic, "DSV2", 4); rh.version = hdr.version; rh.type = MSG_CREATE_FIRE; rh.flags = 0; rh.reserved = 0; rh.request_id = hdr.request_id;
            bool fire_ready = g_fireReady.load(std::memory_order_acquire);
            if (!fire_ready) {
                rh.length = 0;
                resp.resize(kWireHeaderSize);
                std::memcpy(resp.data(), &rh.magic[0], 4);
                std::memcpy(resp.data() + 4, &rh.version, 1);
                std::memcpy(resp.data() + 5, &rh.type, 1);
                std::memcpy(resp.data() + 6, &rh.flags, 1);
                std::memcpy(resp.data() + 7, &rh.reserved, 1);
                std::memcpy(resp.data() + 8, &rh.request_id, 8);
                std::memcpy(resp.data() + 16, &rh.length, 4);
                LOGW("server_v2", "CREATE_FIRE timeout");
            } else {
                rh.length = sizeof(float) * 3 + sizeof(int32_t);
                resp.resize(kWireHeaderSize + rh.length);
                std::memcpy(resp.data(), &rh.magic[0], 4);
                std::memcpy(resp.data() + 4, &rh.version, 1);
                std::memcpy(resp.data() + 5, &rh.type, 1);
                std::memcpy(resp.data() + 6, &rh.flags, 1);
                std::memcpy(resp.data() + 7, &rh.reserved, 1);
                std::memcpy(resp.data() + 8, &rh.request_id, 8);
                std::memcpy(resp.data() + 16, &rh.length, 4);
                std::memcpy(resp.data() + 20, &g_firePos[0], sizeof(float) * 3);
                int32_t fid = static_cast<int32_t>(g_fireId);
                std::memcpy(resp.data() + 20 + sizeof(float) * 3, &fid, sizeof(int32_t));
            }
            write_response(resp);
            return;
        }
        case MSG_CREATE_FIGHT: {
            g_fightReady.store(false, std::memory_order_release);
            enqueue_command("CREATE_FIGHT");
            int tries = 0;
            while (!g_fightReady.load(std::memory_order_acquire) && tries < 2000) { std::this_thread::sleep_for(std::chrono::milliseconds(5)); tries++; }

            MsgHeader rh{}; std::memcpy(rh.magic, "DSV2", 4); rh.version = hdr.version; rh.type = MSG_CREATE_FIGHT; rh.flags = 0; rh.reserved = 0; rh.request_id = hdr.request_id;
            bool fight_ready = g_fightReady.load(std::memory_order_acquire);
            if (!fight_ready) {
                rh.length = 0;
                resp.resize(kWireHeaderSize);
                std::memcpy(resp.data(), &rh.magic[0], 4);
                std::memcpy(resp.data() + 4, &rh.version, 1);
                std::memcpy(resp.data() + 5, &rh.type, 1);
                std::memcpy(resp.data() + 6, &rh.flags, 1);
                std::memcpy(resp.data() + 7, &rh.reserved, 1);
                std::memcpy(resp.data() + 8, &rh.request_id, 8);
                std::memcpy(resp.data() + 16, &rh.length, 4);
                LOGW("server_v2", "CREATE_FIGHT timeout");
            } else {
                rh.length = sizeof(float) * 3;
                resp.resize(kWireHeaderSize + rh.length);
                std::memcpy(resp.data(), &rh.magic[0], 4);
                std::memcpy(resp.data() + 4, &rh.version, 1);
                std::memcpy(resp.data() + 5, &rh.type, 1);
                std::memcpy(resp.data() + 6, &rh.flags, 1);
                std::memcpy(resp.data() + 7, &rh.reserved, 1);
                std::memcpy(resp.data() + 8, &rh.request_id, 8);
                std::memcpy(resp.data() + 16, &rh.length, 4);
                std::memcpy(resp.data() + 20, &g_fightPos[0], sizeof(float) * 3);
            }
            write_response(resp);
            return;
        }
        case MSG_SET_POSTURE: {
            if (hdr.length >= sizeof(float) * 6 && payload.size() >= sizeof(float) * 6) {
                float x = *reinterpret_cast<float*>(&payload[0]);
                float y = *reinterpret_cast<float*>(&payload[4]);
                float z = *reinterpret_cast<float*>(&payload[8]);
                float rx = *reinterpret_cast<float*>(&payload[12]);
                float ry = *reinterpret_cast<float*>(&payload[16]);
                float rz = *reinterpret_cast<float*>(&payload[20]);
                
                // 验证所有浮点数的有效性
                if (std::isfinite(x) && std::isfinite(y) && std::isfinite(z) && 
                    std::isfinite(rx) && std::isfinite(ry) && std::isfinite(rz)) {
                    std::string s = std::string("SET_POSTURE ") + std::to_string(x) + " " + std::to_string(y) + " " + std::to_string(z) + " " + std::to_string(rx) + " " + std::to_string(ry) + " " + std::to_string(rz);
                    enqueue_command(s);
                    LOGD("server_v2", std::string("Enqueue ") + s);
                } else {
                    LOGE("server_v2", "Invalid float values in SET_POSTURE command");
                }
            } else {
                LOGE("server_v2", std::string("MSG_SET_POSTURE: Invalid payload size. Expected: ") + std::to_string(sizeof(float) * 6) + ", Got: " + std::to_string(hdr.length));
            }
            MsgHeader rh{}; std::memcpy(rh.magic, "DSV2", 4); rh.version = hdr.version; rh.type = MSG_SET_POSTURE; rh.flags = 0; rh.reserved = 0; rh.request_id = hdr.request_id; rh.length = 0;
            resp.resize(kWireHeaderSize);
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
        case MSG_TELEPORT_PLAYER: {
            if (hdr.length >= sizeof(float) * 3) {
                float x = *reinterpret_cast<float*>(&payload[0]);
                float y = *reinterpret_cast<float*>(&payload[4]);
                float z = *reinterpret_cast<float*>(&payload[8]);
                std::string s = std::string("TELEPORT_PLAYER ") + std::to_string(x) + " " + std::to_string(y) + " " + std::to_string(z);
                enqueue_command(s);
                LOGD("server_v2", std::string("Enqueue ") + s);
            }
            MsgHeader rh{}; std::memcpy(rh.magic, "DSV2", 4); rh.version = hdr.version; rh.type = MSG_TELEPORT_PLAYER; rh.flags = 0; rh.reserved = 0; rh.request_id = hdr.request_id; rh.length = 0;
            resp.resize(kWireHeaderSize);
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
        case MSG_RESTORE_PLAYER: {
            enqueue_command("RESTORE_PLAYER");
            MsgHeader rh{}; std::memcpy(rh.magic, "DSV2", 4); rh.version = hdr.version; rh.type = MSG_RESTORE_PLAYER; rh.flags = 0; rh.reserved = 0; rh.request_id = hdr.request_id; rh.length = 0;
            resp.resize(kWireHeaderSize);
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
            LOGD("server_v2", "Processing MSG_CAPTURE request");
            enqueue_command("REQUEST");
            int tries = 0;
            while (cmdToCatch != catchStop && tries < 3000) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                tries++;
            }
            
            if (tries >= 3000) {
                LOGW("server_v2", "MSG_CAPTURE: Timeout waiting for capture completion");
            }
            
            void* rgb_ptr = nullptr; 
            void* depth_ptr = nullptr;
            int rgb_size = 0;
            int depth_size = 0;
            
            try {
                rgb_size = export_get_color_buffer(&rgb_ptr);
                depth_size = export_get_depth_buffer(&depth_ptr);
                
                if (rgb_ptr == nullptr || depth_ptr == nullptr) {
                    LOGE("server_v2", "MSG_CAPTURE: Null buffer pointers returned");
                    // 返回空响应
                    MsgHeader rh{}; 
                    std::memcpy(rh.magic, "DSV2", 4); 
                    rh.version = hdr.version; 
                    rh.type = MSG_CAPTURE; 
                    rh.flags = 0; 
                    rh.reserved = 0; 
                    rh.request_id = hdr.request_id; 
                    rh.length = 0;
                    resp.resize(kWireHeaderSize);
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
                
                LOGD("server_v2", std::string("Capture: rgb_size=") + std::to_string(rgb_size) + ", depth_size=" + std::to_string(depth_size));
                
                int w_rgb = export_get_last_color_width();
                int h_rgb = export_get_last_color_height();
                int w_depth = export_get_last_depth_width();
                int h_depth = export_get_last_depth_height();
                
                // 验证尺寸的合理性
                if (w_rgb <= 0 || h_rgb <= 0 || w_depth <= 0 || h_depth <= 0) {
                    LOGE("server_v2", std::string("Invalid image dimensions: rgb(") + std::to_string(w_rgb) + "x" + std::to_string(h_rgb) + "), depth(" + std::to_string(w_depth) + "x" + std::to_string(h_depth) + ")");
                    // 返回空响应
                    MsgHeader rh{}; 
                    std::memcpy(rh.magic, "DSV2", 4); 
                    rh.version = hdr.version; 
                    rh.type = MSG_CAPTURE; 
                    rh.flags = 0; 
                    rh.reserved = 0; 
                    rh.request_id = hdr.request_id; 
                    rh.length = 0;
                    resp.resize(kWireHeaderSize);
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
                
                uint32_t hdr_len = sizeof(uint32_t) * 4 + rgb_size + depth_size;
                MsgHeader rh{}; 
                std::memcpy(rh.magic, "DSV2", 4); 
                rh.version = hdr.version; 
                rh.type = MSG_CAPTURE; 
                rh.flags = 0; 
                rh.reserved = 0; 
                rh.request_id = hdr.request_id; 
                rh.length = hdr_len;
                
                resp.resize(kWireHeaderSize + hdr_len);
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
                if (rgb_size > 0) {
                    std::memcpy(p, rgb_ptr, rgb_size);
                }
                p += rgb_size;
                if (depth_size > 0) {
                    std::memcpy(p, depth_ptr, depth_size);
                }
                
            } catch (const std::exception& e) {
                LOGE("server_v2", std::string("Exception in MSG_CAPTURE: ") + e.what());
                // 返回空响应
                MsgHeader rh{}; 
                std::memcpy(rh.magic, "DSV2", 4); 
                rh.version = hdr.version; 
                rh.type = MSG_CAPTURE; 
                rh.flags = 0; 
                rh.reserved = 0; 
                rh.request_id = hdr.request_id; 
                rh.length = 0;
                resp.resize(kWireHeaderSize);
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
            write_response(resp);
            return;
        }
        default: {
            LOGE("server_v2", std::string("Unknown message type: ") + std::to_string(hdr.type));
            socket_.close(); start_accept(); return;
        }
    }
    } catch (const std::exception& e) {
        LOGE("server_v2", std::string("Exception in handle_client: ") + e.what());
        try {
            socket_.close();
        } catch (...) {
            // 忽略关闭socket时的异常
        }
        start_accept();
        return;
    } catch (...) {
        LOGE("server_v2", "Unknown exception in handle_client");
        try {
            socket_.close();
        } catch (...) {
            // 忽略关闭socket时的异常
        }
        start_accept();
        return;
    }
}

void ServerV2::write_response(const std::vector<unsigned char>& data) {
    try {
        system::error_code ec;
        size_t bytes_written = asio::write(socket_, asio::buffer(data), ec);
        if (ec) {
            LOGE("server_v2", std::string("Failed to write response: ") + ec.message());
        } else {
            LOGD("server_v2", std::string("Response sent successfully: ") + std::to_string(bytes_written) + " bytes");
        }
    } catch (const std::exception& e) {
        LOGE("server_v2", std::string("Exception in write_response: ") + e.what());
    } catch (...) {
        LOGE("server_v2", "Unknown exception in write_response");
    }
    
    try {
        socket_.close();
    } catch (...) {
        // 忽略关闭socket时的异常
    }
    start_accept();
}

void InitializeServerV2() {
    if (!g_ws_inited_v2) {
        WSADATA wsaData; 
        int r = WSAStartup(MAKEWORD(2,2), &wsaData); 
        if (r == 0) {
            g_ws_inited_v2 = true;
            LOGD("server_v2", "WSAStartup successful");
        } else {
            LOGE("server_v2", std::string("WSAStartup failed with error: ") + std::to_string(r));
            return;
        }
    }
    
    if (g_serverV2Instance) {
        LOGD("server_v2", "ServerV2 already initialized");
        return;
    }
    
    try {
        g_thread_v2 = std::make_unique<std::thread>([](){
            try {
                LOGD("server_v2", "Starting ServerV2 on port 23456");
                g_serverV2Instance = std::make_unique<ServerV2>(g_io_v2, 23456);
                g_io_v2.run();
                LOGD("server_v2", "ServerV2 io_context finished");
            } catch (const std::exception& e) {
                LOGE("server_v2", std::string("Exception in server thread: ") + e.what());
            } catch (...) {
                LOGE("server_v2", "Unknown exception in server thread");
            }
        });
        g_thread_v2->detach();
        LOGD("server_v2", "ServerV2 thread started and detached");
    } catch (const std::exception& e) {
        LOGE("server_v2", std::string("Failed to create server thread: ") + e.what());
    } catch (...) {
        LOGE("server_v2", "Unknown exception creating server thread");
    }
}

void ShutdownServerV2() {
    g_io_v2.stop();
    g_serverV2Instance.reset();
    g_thread_v2.reset();
}
