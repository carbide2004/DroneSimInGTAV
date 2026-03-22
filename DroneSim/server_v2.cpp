#include "server_v2.h"
#include "utils.h"
#include "camera.h"
#include "export.h"
#include "logging.h"
#include "command_queue.h"
#include <thread>
#include <cstring>
#include <algorithm>

using namespace boost;

static std::unique_ptr<ServerV2> g_serverV2Instance;
static asio::io_context g_io_v2;
static std::unique_ptr<std::thread> g_thread_v2;
static bool g_ws_inited_v2 = false;

extern volatile catchState cmdToCatch;

extern volatile bool g_poseReady;
extern float g_pose[6];

extern volatile bool g_accidentReady;
extern float g_accidentPos[3];

extern volatile bool g_recordingEnabled;
extern volatile int g_recordingStep;
extern char g_recordingSessionDir[260];
extern char g_recordingRequestedSession[128];
extern char g_recordingRequestedTask[256];

extern volatile bool g_fireReady;
extern float g_firePos[3];
extern int g_fireId;

extern volatile bool g_fightReady;
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
        if (!read_exact(socket_, &hdr.magic[0], 4)) { socket_.close(); start_accept(); return; }
        if (!read_exact(socket_, &hdr.version, 1)) { socket_.close(); start_accept(); return; }
        if (!read_exact(socket_, &hdr.type, 1)) { socket_.close(); start_accept(); return; }
        if (!read_exact(socket_, &hdr.flags, 1)) { socket_.close(); start_accept(); return; }
        if (!read_exact(socket_, &hdr.reserved, 1)) { socket_.close(); start_accept(); return; }
        if (!read_exact(socket_, &hdr.request_id, 8)) { socket_.close(); start_accept(); return; }
        if (!read_exact(socket_, &hdr.length, 4)) { socket_.close(); start_accept(); return; }
        if (std::memcmp(hdr.magic, "DSV2", 4) != 0) { socket_.close(); start_accept(); return; }
        
        // Validate payload length to prevent excessive memory allocation
        const uint32_t MAX_PAYLOAD_SIZE = 10 * 1024 * 1024; // 10MB max
        if (hdr.length > MAX_PAYLOAD_SIZE) {
            LOGW("server_v2", std::string("Payload too large: ") + std::to_string(hdr.length) + " bytes");
            socket_.close(); 
            start_accept(); 
            return;
        }
        
        std::vector<unsigned char> payload;
        if (hdr.length > 0) {
            try {
                payload.resize(hdr.length);
                if (!read_exact(socket_, payload.data(), hdr.length)) { 
                    socket_.close(); 
                    start_accept(); 
                    return; 
                }
            } catch (const std::exception& e) {
                LOGE("server_v2", std::string("Failed to allocate payload buffer: ") + e.what());
                socket_.close(); 
                start_accept(); 
                return;
            }
        }

    std::vector<unsigned char> resp;
    switch (hdr.type) {
        case MSG_CREATE_CAMERA: {
            enqueue_command("CREATE_CAMERA");
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
                // Validate payload alignment for float access
                if (payload.size() >= sizeof(float) * 3 && 
                    reinterpret_cast<uintptr_t>(payload.data()) % alignof(float) == 0) {
                    float dx = *reinterpret_cast<float*>(&payload[0]);
                    float dy = *reinterpret_cast<float*>(&payload[4]);
                    float dz = *reinterpret_cast<float*>(&payload[8]);
                    std::string s = std::string("MOVE ") + std::to_string(dx) + " " + std::to_string(dy) + " " + std::to_string(dz);
                    enqueue_command(s);
                    LOGD("server_v2", std::string("Enqueue ") + s);
                } else {
                    LOGW("server_v2", "MSG_MOVE: Invalid payload alignment or size");
                }
            } else {
                LOGW("server_v2", "MSG_MOVE: Insufficient payload length");
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
                // Validate payload alignment for float access
                if (payload.size() >= sizeof(float) * 3 && 
                    reinterpret_cast<uintptr_t>(payload.data()) % alignof(float) == 0) {
                    float rx = *reinterpret_cast<float*>(&payload[0]);
                    float ry = *reinterpret_cast<float*>(&payload[4]);
                    float rz = *reinterpret_cast<float*>(&payload[8]);
                    std::string s = std::string("ROTATE ") + std::to_string(rx) + " " + std::to_string(ry) + " " + std::to_string(rz);
                    enqueue_command(s);
                    LOGD("server_v2", std::string("Enqueue ") + s);
                } else {
                    LOGW("server_v2", "MSG_ROTATE: Invalid payload alignment or size");
                }
            } else {
                LOGW("server_v2", "MSG_ROTATE: Insufficient payload length");
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
                // Validate payload alignment for float access
                if (payload.size() >= sizeof(float) && 
                    reinterpret_cast<uintptr_t>(payload.data()) % alignof(float) == 0) {
                    float fov = *reinterpret_cast<float*>(&payload[0]);
                    std::string s = std::string("SETFOV:") + std::to_string(fov);
                    enqueue_command(s);
                } else {
                    LOGW("server_v2", "MSG_SET_FOV: Invalid payload alignment or size");
                }
            } else {
                LOGW("server_v2", "MSG_SET_FOV: Insufficient payload length");
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
        case MSG_GET_POSE: {
            g_poseReady = false;
            enqueue_command("GET_POSE");
            LOGD("server_v2", std::string("GET_POSE enqueued"));
            int tries = 0;
            while (!g_poseReady && tries < 300) { std::this_thread::sleep_for(std::chrono::milliseconds(5)); tries++; }
            MsgHeader rh{}; std::memcpy(rh.magic, "DSV2", 4); rh.version = hdr.version; rh.type = MSG_GET_POSE; rh.flags = 0; rh.reserved = 0; rh.request_id = hdr.request_id;
            if (!g_poseReady) {
                rh.length = 0;
                resp.resize(sizeof(rh));
                std::memcpy(resp.data(), &rh.magic[0], 4);
                std::memcpy(resp.data() + 4, &rh.version, 1);
                std::memcpy(resp.data() + 5, &rh.type, 1);
                std::memcpy(resp.data() + 6, &rh.flags, 1);
                std::memcpy(resp.data() + 7, &rh.reserved, 1);
                std::memcpy(resp.data() + 8, &rh.request_id, 8);
                std::memcpy(resp.data() + 16, &rh.length, 4);
                LOGW("server_v2", "GET_POSE timeout or camera mode inactive");
            } else {
                rh.length = sizeof(float)*6;
                resp.resize(sizeof(rh) + sizeof(float)*6);
                std::memcpy(resp.data(), &rh.magic[0], 4);
                std::memcpy(resp.data() + 4, &rh.version, 1);
                std::memcpy(resp.data() + 5, &rh.type, 1);
                std::memcpy(resp.data() + 6, &rh.flags, 1);
                std::memcpy(resp.data() + 7, &rh.reserved, 1);
                std::memcpy(resp.data() + 8, &rh.request_id, 8);
                std::memcpy(resp.data() + 16, &rh.length, 4);
                std::memcpy(resp.data() + 20, &g_pose[0], sizeof(float)*6);
                LOGD("server_v2", std::string("GET_POSE ready: ") + std::to_string(g_pose[0]) + "," + std::to_string(g_pose[1]) + "," + std::to_string(g_pose[2]));
            }
            write_response(resp);
            return;
        }
        case MSG_SET_TIME: {
            if (hdr.length >= 12) {
                // Validate payload alignment for int access
                if (payload.size() >= 12 && 
                    reinterpret_cast<uintptr_t>(payload.data()) % alignof(int) == 0) {
                    int h = *reinterpret_cast<int*>(&payload[0]);
                    int m = *reinterpret_cast<int*>(&payload[4]);
                    int s = *reinterpret_cast<int*>(&payload[8]);
                    std::string sCmd = std::string("SET_TIME ") + std::to_string(h) + " " + std::to_string(m) + " " + std::to_string(s);
                    enqueue_command(sCmd);
                } else {
                    LOGW("server_v2", "MSG_SET_TIME: Invalid payload alignment or size");
                }
            } else {
                LOGW("server_v2", "MSG_SET_TIME: Insufficient payload length");
            }
            MsgHeader rh{}; std::memcpy(rh.magic, "DSV2", 4); rh.version = hdr.version; rh.type = MSG_SET_TIME; rh.flags = 0; rh.reserved = 0; rh.request_id = hdr.request_id; rh.length = 0;
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
        case MSG_SET_WEATHER: {
            std::string name;
            if (hdr.length > 0) {
                name.assign(reinterpret_cast<char*>(payload.data()), hdr.length);
            }
            std::string sCmd = std::string("SET_WEATHER ") + name;
            enqueue_command(sCmd);
            MsgHeader rh{}; std::memcpy(rh.magic, "DSV2", 4); rh.version = hdr.version; rh.type = MSG_SET_WEATHER; rh.flags = 0; rh.reserved = 0; rh.request_id = hdr.request_id; rh.length = 0;
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
        case MSG_STOP_CAMERA: {
            enqueue_command("STOP_CAMERA");
            MsgHeader rh{}; std::memcpy(rh.magic, "DSV2", 4); rh.version = hdr.version; rh.type = MSG_STOP_CAMERA; rh.flags = 0; rh.reserved = 0; rh.request_id = hdr.request_id; rh.length = 0;
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
        case MSG_CREATE_ACCIDENT: {
            g_accidentReady = false;
            enqueue_command("CREATE_ACCIDENT");
            LOGD("server_v2", std::string("CREATE_ACCIDENT enqueued"));
            int tries = 0;
            // Wait for up to 20 seconds for the accident to be set up and collision detected
            while (!g_accidentReady && tries < 4000) { std::this_thread::sleep_for(std::chrono::milliseconds(5)); tries++; }
            
            MsgHeader rh{}; std::memcpy(rh.magic, "DSV2", 4); rh.version = hdr.version; rh.type = MSG_CREATE_ACCIDENT; rh.flags = 0; rh.reserved = 0; rh.request_id = hdr.request_id;
            
            if (!g_accidentReady) {
                rh.length = 0;
                resp.resize(sizeof(rh));
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
                resp.resize(sizeof(rh) + sizeof(float) * 3);
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
            uint8_t enabled = g_recordingEnabled ? 1 : 0;
            int32_t step = static_cast<int32_t>(g_recordingStep);
            uint16_t path_len = static_cast<uint16_t>(std::min<size_t>(std::strlen(g_recordingSessionDir), 65535));
            uint32_t payload_len = 1 + 4 + 2 + path_len;
            MsgHeader rh{}; std::memcpy(rh.magic, "DSV2", 4); rh.version = hdr.version; rh.type = MSG_GET_RECORDING_INFO; rh.flags = 0; rh.reserved = 0; rh.request_id = hdr.request_id; rh.length = payload_len;
            resp.resize(sizeof(rh) + payload_len);
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
        case MSG_CREATE_FIRE: {
            g_fireReady = false;
            enqueue_command("CREATE_FIRE");
            int tries = 0;
            while (!g_fireReady && tries < 2000) { std::this_thread::sleep_for(std::chrono::milliseconds(5)); tries++; }

            MsgHeader rh{}; std::memcpy(rh.magic, "DSV2", 4); rh.version = hdr.version; rh.type = MSG_CREATE_FIRE; rh.flags = 0; rh.reserved = 0; rh.request_id = hdr.request_id;
            if (!g_fireReady) {
                rh.length = 0;
                resp.resize(sizeof(rh));
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
                resp.resize(sizeof(rh) + rh.length);
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
            g_fightReady = false;
            enqueue_command("CREATE_FIGHT");
            int tries = 0;
            while (!g_fightReady && tries < 2000) { std::this_thread::sleep_for(std::chrono::milliseconds(5)); tries++; }

            MsgHeader rh{}; std::memcpy(rh.magic, "DSV2", 4); rh.version = hdr.version; rh.type = MSG_CREATE_FIGHT; rh.flags = 0; rh.reserved = 0; rh.request_id = hdr.request_id;
            if (!g_fightReady) {
                rh.length = 0;
                resp.resize(sizeof(rh));
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
                resp.resize(sizeof(rh) + rh.length);
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
            if (hdr.length >= sizeof(float) * 6) {
                // Validate payload alignment for float access
                if (payload.size() >= sizeof(float) * 6 && 
                    reinterpret_cast<uintptr_t>(payload.data()) % alignof(float) == 0) {
                    float x = *reinterpret_cast<float*>(&payload[0]);
                    float y = *reinterpret_cast<float*>(&payload[4]);
                    float z = *reinterpret_cast<float*>(&payload[8]);
                    float rx = *reinterpret_cast<float*>(&payload[12]);
                    float ry = *reinterpret_cast<float*>(&payload[16]);
                    float rz = *reinterpret_cast<float*>(&payload[20]);
                    std::string s = std::string("SET_POSTURE ") + std::to_string(x) + " " + std::to_string(y) + " " + std::to_string(z) + " " + std::to_string(rx) + " " + std::to_string(ry) + " " + std::to_string(rz);
                    enqueue_command(s);
                    LOGD("server_v2", std::string("Enqueue ") + s);
                } else {
                    LOGW("server_v2", "MSG_SET_POSTURE: Invalid payload alignment or size");
                }
            } else {
                LOGW("server_v2", "MSG_SET_POSTURE: Insufficient payload length");
            }
            MsgHeader rh{}; std::memcpy(rh.magic, "DSV2", 4); rh.version = hdr.version; rh.type = MSG_SET_POSTURE; rh.flags = 0; rh.reserved = 0; rh.request_id = hdr.request_id; rh.length = 0;
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
        case MSG_TELEPORT_PLAYER: {
            if (hdr.length >= sizeof(float) * 3) {
                // Validate payload alignment for float access
                if (payload.size() >= sizeof(float) * 3 && 
                    reinterpret_cast<uintptr_t>(payload.data()) % alignof(float) == 0) {
                    float x = *reinterpret_cast<float*>(&payload[0]);
                    float y = *reinterpret_cast<float*>(&payload[4]);
                    float z = *reinterpret_cast<float*>(&payload[8]);
                    std::string s = std::string("TELEPORT_PLAYER ") + std::to_string(x) + " " + std::to_string(y) + " " + std::to_string(z);
                    enqueue_command(s);
                    LOGD("server_v2", std::string("Enqueue ") + s);
                } else {
                    LOGW("server_v2", "MSG_TELEPORT_PLAYER: Invalid payload alignment or size");
                }
            } else {
                LOGW("server_v2", "MSG_TELEPORT_PLAYER: Insufficient payload length");
            }
            MsgHeader rh{}; std::memcpy(rh.magic, "DSV2", 4); rh.version = hdr.version; rh.type = MSG_TELEPORT_PLAYER; rh.flags = 0; rh.reserved = 0; rh.request_id = hdr.request_id; rh.length = 0;
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
        case MSG_RESTORE_PLAYER: {
            enqueue_command("RESTORE_PLAYER");
            MsgHeader rh{}; std::memcpy(rh.magic, "DSV2", 4); rh.version = hdr.version; rh.type = MSG_RESTORE_PLAYER; rh.flags = 0; rh.reserved = 0; rh.request_id = hdr.request_id; rh.length = 0;
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
            enqueue_command("REQUEST");
            int tries = 0;
            while (cmdToCatch != catchStop && tries < 3000) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                tries++;
            }
            
            // Get buffer pointers and sizes
            void* rgb_ptr = nullptr; 
            int rgb_size = export_get_color_buffer(&rgb_ptr);
            void* depth_ptr = nullptr; 
            int depth_size = export_get_depth_buffer(&depth_ptr);
            
            LOGD("server_v2", std::string("Capture: rgb_size=") + std::to_string(rgb_size) + ", depth_size=" + std::to_string(depth_size));
            
            // Validate buffer sizes to prevent crashes
            const int MAX_BUFFER_SIZE = 50 * 1024 * 1024; // 50MB max per buffer
            if (rgb_size < 0 || rgb_size > MAX_BUFFER_SIZE) {
                LOGW("server_v2", std::string("Invalid RGB buffer size: ") + std::to_string(rgb_size));
                rgb_size = 0;
                rgb_ptr = nullptr;
            }
            if (depth_size < 0 || depth_size > MAX_BUFFER_SIZE) {
                LOGW("server_v2", std::string("Invalid depth buffer size: ") + std::to_string(depth_size));
                depth_size = 0;
                depth_ptr = nullptr;
            }
            
            // Validate pointers
            if (rgb_size > 0 && rgb_ptr == nullptr) {
                LOGW("server_v2", "RGB buffer pointer is null despite non-zero size");
                rgb_size = 0;
            }
            if (depth_size > 0 && depth_ptr == nullptr) {
                LOGW("server_v2", "Depth buffer pointer is null despite non-zero size");
                depth_size = 0;
            }
            
            int w_rgb = export_get_last_color_width();
            int h_rgb = export_get_last_color_height();
            int w_depth = export_get_last_depth_width();
            int h_depth = export_get_last_depth_height();
            
            // Validate dimensions
            if (w_rgb <= 0 || h_rgb <= 0 || w_rgb > 10000 || h_rgb > 10000) {
                LOGW("server_v2", std::string("Invalid RGB dimensions: ") + std::to_string(w_rgb) + "x" + std::to_string(h_rgb));
                w_rgb = h_rgb = 0;
                rgb_size = 0;
                rgb_ptr = nullptr;
            }
            if (w_depth <= 0 || h_depth <= 0 || w_depth > 10000 || h_depth > 10000) {
                LOGW("server_v2", std::string("Invalid depth dimensions: ") + std::to_string(w_depth) + "x" + std::to_string(h_depth));
                w_depth = h_depth = 0;
                depth_size = 0;
                depth_ptr = nullptr;
            }
            
            uint32_t hdr_len = sizeof(uint32_t) * 6 + static_cast<uint32_t>(rgb_size) + static_cast<uint32_t>(depth_size);
            
            // Validate total response size
            const uint32_t MAX_RESPONSE_SIZE = 100 * 1024 * 1024; // 100MB max response
            if (hdr_len > MAX_RESPONSE_SIZE) {
                LOGE("server_v2", std::string("Response too large: ") + std::to_string(hdr_len) + " bytes");
                // Send empty response
                MsgHeader rh{}; 
                std::memcpy(rh.magic, "DSV2", 4); 
                rh.version = hdr.version; 
                rh.type = MSG_CAPTURE; 
                rh.flags = 0; 
                rh.reserved = 0; 
                rh.request_id = hdr.request_id; 
                rh.length = 0;
                resp.resize(sizeof(rh));
                std::memcpy(resp.data(), &rh, sizeof(rh));
                write_response(resp);
                return;
            }
            
            // Create response
            MsgHeader rh{}; 
            std::memcpy(rh.magic, "DSV2", 4); 
            rh.version = hdr.version; 
            rh.type = MSG_CAPTURE; 
            rh.flags = 0; 
            rh.reserved = 0; 
            rh.request_id = hdr.request_id; 
            rh.length = hdr_len;
            
            try {
                resp.resize(sizeof(rh) + hdr_len);
            } catch (const std::exception& e) {
                LOGE("server_v2", std::string("Failed to allocate response buffer: ") + e.what());
                // Send empty response
                rh.length = 0;
                resp.resize(sizeof(rh));
                std::memcpy(resp.data(), &rh, sizeof(rh));
                write_response(resp);
                return;
            }
            
            // Copy header
            std::memcpy(resp.data(), &rh, sizeof(rh));
            
            // Copy payload header (dimensions and sizes)
            uint32_t* p32 = reinterpret_cast<uint32_t*>(resp.data() + sizeof(rh));
            p32[0] = static_cast<uint32_t>(rgb_size);
            p32[1] = static_cast<uint32_t>(depth_size);
            p32[2] = static_cast<uint32_t>(w_rgb);
            p32[3] = static_cast<uint32_t>(h_rgb);
            p32[4] = static_cast<uint32_t>(w_depth);
            p32[5] = static_cast<uint32_t>(h_depth);
            
            // Copy image data with additional safety checks
            unsigned char* p = resp.data() + sizeof(rh) + sizeof(uint32_t) * 6;
            if (rgb_size > 0 && rgb_ptr != nullptr) {
                try {
                    // Verify we have enough space in response buffer
                    if (p + rgb_size <= resp.data() + resp.size()) {
                        std::memcpy(p, rgb_ptr, rgb_size);
                    } else {
                        LOGE("server_v2", "RGB buffer copy would exceed response buffer bounds");
                        // Send empty response instead
                        MsgHeader empty_rh{}; 
                        std::memcpy(empty_rh.magic, "DSV2", 4); 
                        empty_rh.version = hdr.version; 
                        empty_rh.type = MSG_CAPTURE; 
                        empty_rh.flags = 0; 
                        empty_rh.reserved = 0; 
                        empty_rh.request_id = hdr.request_id; 
                        empty_rh.length = 0;
                        resp.resize(sizeof(empty_rh));
                        std::memcpy(resp.data(), &empty_rh, sizeof(empty_rh));
                        write_response(resp);
                        return;
                    }
                } catch (const std::exception& e) {
                    LOGE("server_v2", std::string("RGB buffer copy failed: ") + e.what());
                    return;
                }
            }
            p += rgb_size;
            if (depth_size > 0 && depth_ptr != nullptr) {
                try {
                    // Verify we have enough space in response buffer
                    if (p + depth_size <= resp.data() + resp.size()) {
                        std::memcpy(p, depth_ptr, depth_size);
                    } else {
                        LOGE("server_v2", "Depth buffer copy would exceed response buffer bounds");
                        // Send empty response instead
                        MsgHeader empty_rh{}; 
                        std::memcpy(empty_rh.magic, "DSV2", 4); 
                        empty_rh.version = hdr.version; 
                        empty_rh.type = MSG_CAPTURE; 
                        empty_rh.flags = 0; 
                        empty_rh.reserved = 0; 
                        empty_rh.request_id = hdr.request_id; 
                        empty_rh.length = 0;
                        resp.resize(sizeof(empty_rh));
                        std::memcpy(resp.data(), &empty_rh, sizeof(empty_rh));
                        write_response(resp);
                        return;
                    }
                } catch (const std::exception& e) {
                    LOGE("server_v2", std::string("Depth buffer copy failed: ") + e.what());
                    return;
                }
            }
            
            write_response(resp);
            return;
        }
        default: {
            socket_.close(); start_accept(); return;
        }
    }
    
    } catch (const std::exception& e) {
        LOGE("server_v2", std::string("Exception in handle_client: ") + e.what());
        try {
            socket_.close();
        } catch (...) {
            // Ignore close errors
        }
        start_accept();
    } catch (...) {
        LOGE("server_v2", "Unknown exception in handle_client");
        try {
            socket_.close();
        } catch (...) {
            // Ignore close errors
        }
        start_accept();
    }
}

void ServerV2::write_response(const std::vector<unsigned char>& data) {
    try {
        system::error_code ec;
        asio::write(socket_, asio::buffer(data), ec);
        if (ec) {
            LOGW("server_v2", std::string("Write error: ") + ec.message());
        }
    } catch (const std::exception& e) {
        LOGE("server_v2", std::string("Write exception: ") + e.what());
    } catch (...) {
        LOGE("server_v2", "Unknown write exception");
    }
    
    try {
        socket_.close();
    } catch (...) {
        // Ignore close errors
    }
    
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
