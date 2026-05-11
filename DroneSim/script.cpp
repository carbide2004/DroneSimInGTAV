#include "script.h"
#include "export.h"
#include "main.h"
#include "utils.h"
#include "camera.h"
#include "server.h"
#include "server_v2.h"
#include "logging.h"
#include "keyboard.h"
#include <string>
#include <fstream>
#include <algorithm>
#include <cstring>
#include <cstdio>
#include <set>
#include <atlimage.h>
#include <time.h>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <atomic>
#include "command_queue.h"
 
std::atomic<bool> g_poseReady{false};
float g_pose[6] = {0};

std::atomic<bool> g_accidentReady{false};
float g_accidentPos[3] = {0};

std::atomic<bool> g_recordingEnabled{false};
std::atomic<int> g_recordingStep{0};
char g_recordingSessionDir[260] = {0};
char g_recordingRequestedSession[128] = {0};
char g_recordingRequestedTask[256] = {0};

std::atomic<bool> g_fireReady{false};
float g_firePos[3] = {0};
int g_fireId = -1;

std::atomic<bool> g_arrestReady{false};
float g_arrestPos[3] = {0};

static std::vector<int> g_spawnedFireIds;
static std::vector<Vehicle> g_spawnedVehicles;
static std::vector<Ped> g_spawnedPeds;

// 简单的3D位置类，替代可能有问题的Vector3
class Position3D {
public:
    float x, y, z;

    Position3D() : x(0.0f), y(0.0f), z(0.0f) {}
    Position3D(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}

    // 计算到另一个位置的距离
    float distance_to(Position3D other) const {
        float dx = x - other.x;
        float dy = y - other.y;
        float dz = z - other.z;
        return sqrtf(dx * dx + dy * dy + dz * dz);
    }
};

enum AutoCollectEvent {
    AUTO_EVENT_ACCIDENT = 1,
    AUTO_EVENT_FIRE = 2,
    AUTO_EVENT_ARREST = 3
};


// Fire maintenance system
static bool g_fireMaintenanceActive = false;
static Position3D g_fireMaintenancePos;
static Vehicle g_fireVehicle = 0;
static int g_fireMaintenanceTimer = 0;

scriptStatusEnum scriptStatus = scriptStop;

extern std::atomic<catchState> cmdToCatch;

static std::ofstream g_recordingStepsFile;

static bool ensure_dir(const std::string& path) {
    if (path.empty()) return false;
    if (CreateDirectoryA(path.c_str(), nullptr)) return true;
    DWORD e = GetLastError();
    return e == ERROR_ALREADY_EXISTS;
}

static std::string make_timestamp_session() {
    auto now = std::chrono::system_clock::now();
    std::time_t tt = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    localtime_s(&tm, &tt);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &tm);
    return std::string(buf);
}

static std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 16);
    for (char c : s) {
        if (c == '\\' || c == '"') {
            out.push_back('\\');
            out.push_back(c);
        } else if (c == '\n') {
            out += "\\n";
        } else if (c == '\r') {
            out += "\\r";
        } else if (c == '\t') {
            out += "\\t";
        } else {
            out.push_back(c);
        }
    }
    return out;
}

static void write_metadata(const std::string& base, AutoCollectEvent event_type, const Position3D& target, 
                          const Position3D& start_pos, float start_yaw, int final_steps, const std::string& task) {
    if (base.empty()) return;
    std::ofstream f(base + "\\metadata.jsonl", std::ios::out | std::ios::binary);
    if (!f.is_open()) return;
    
    // 获取当前时间戳
    char timestamp[32];
    time_t rawtime;
    struct tm timeinfo;
    time(&rawtime);
    localtime_s(&timeinfo, &rawtime);
    strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", &timeinfo);
    
    // 根据事件类型确定异常类型
    std::string anomaly_type;
    if (event_type == AUTO_EVENT_FIRE) {
        anomaly_type = "fire";
    } else if (event_type == AUTO_EVENT_ARREST) {
        anomaly_type = "arrest";
    } else if (event_type == AUTO_EVENT_ACCIDENT) {
        anomaly_type = "accident";
    }
    
    // 构建JSON条目
    std::string json_entry = "{";
    json_entry += "\"scenario_id\": \"manual_" + std::string(timestamp) + "\",";
    json_entry += "\"anomaly_type\": \"" + anomaly_type + "\",";
    json_entry += "\"anomaly_position\": {\"x\": " + std::to_string(target.x) + ", \"y\": " + std::to_string(target.y) + ", \"z\": " + std::to_string(target.z) + "},";
    json_entry += "\"start_pose\": {\"x\": " + std::to_string(start_pos.x) + ", \"y\": " + std::to_string(start_pos.y) + ", \"z\": " + std::to_string(start_pos.z) + ", \"rx\": 0.0, \"ry\": 0.0, \"rz\": " + std::to_string(start_yaw) + "},";
    json_entry += "\"expected_steps\": " + std::to_string(final_steps) + ",";
    json_entry += "\"task_description\": \"" + json_escape(task) + "\",";
    json_entry += "\"created_time\": \"" + std::string(timestamp) + "\"";
    json_entry += "}";
    
    f << json_entry << "\n";
    f.flush();
}

static void start_recording_session(const char* task) {
    if (g_recordingEnabled.load(std::memory_order_acquire)) return;
    ensure_dir("data");
    ensure_dir("data\\manual");
    std::string name = (g_recordingRequestedSession[0] != '\0') ? std::string(g_recordingRequestedSession) : make_timestamp_session();
    std::string base = std::string("data\\manual\\") + name;
    ensure_dir(base);
    ensure_dir(base + "\\RGB");
    ensure_dir(base + "\\Depth");
    std::memset(g_recordingSessionDir, 0, sizeof(g_recordingSessionDir));
    size_t n = std::min<size_t>(base.size(), sizeof(g_recordingSessionDir) - 1);
    std::memcpy(g_recordingSessionDir, base.data(), n);
    g_recordingStep.store(0, std::memory_order_release);
    if (g_recordingStepsFile.is_open()) g_recordingStepsFile.close();
    g_recordingStepsFile.open(base + "\\steps.jsonl", std::ios::out | std::ios::binary);
    g_recordingEnabled.store(g_recordingStepsFile.is_open(), std::memory_order_release);
    if (g_recordingEnabled.load(std::memory_order_acquire)) LOGI("script", std::string("Recording started: ") + g_recordingSessionDir);
    else LOGE("script", "Recording start failed");
}

static void stop_recording_session(AutoCollectEvent event_type, const Position3D& target, 
                                   const Position3D& start_pos, float start_yaw, int final_steps, const std::string& task) {
    if (!g_recordingEnabled.load(std::memory_order_acquire)) return;
    g_recordingEnabled.store(false, std::memory_order_release);
    if (g_recordingStepsFile.is_open()) {
        g_recordingStepsFile.flush();
        g_recordingStepsFile.close();
    }
    
    // 写入元数据
    if (g_recordingSessionDir[0] != '\0') {
        std::string base(g_recordingSessionDir);
        write_metadata(base, event_type, target, start_pos, start_yaw, final_steps, task);
    }
    
    LOGI("script", "Recording stopped");
}

// 保持向后兼容的重载函数
static void stop_recording_session() {
    if (!g_recordingEnabled.load(std::memory_order_acquire)) return;
    g_recordingEnabled.store(false, std::memory_order_release);
    if (g_recordingStepsFile.is_open()) {
        g_recordingStepsFile.flush();
        g_recordingStepsFile.close();
    }
    LOGI("script", "Recording stopped");
}

static void record_step(const char* action, float dx, float dy, float dz, float drx, float dry, float drz) {
    if (!g_recordingEnabled.load(std::memory_order_acquire)) return;
    if (g_recordingSessionDir[0] == '\0') return;
    if (!g_recordingStepsFile.is_open()) return;
    Any cam = CAM::GET_RENDERING_CAM();
    Vector3 cam_pos = CAM::GET_CAM_COORD(cam);
    Vector3 cam_rot = CAM::GET_CAM_ROT(cam, 2);
    
    // 转换为Position3D进行处理
    Position3D pos(cam_pos.x, cam_pos.y, cam_pos.z);
    Position3D rot(cam_rot.x, cam_rot.y, cam_rot.z);
    makeCmdStart();
    int tries = 0;
    while (cmdToCatch.load(std::memory_order_acquire) != catchStop && tries < 6000) { WAIT(0); tries++; }
    std::vector<unsigned char> rgb_data;
    std::vector<unsigned char> depth_data;
    int w = 0, h = 0, depth_w = 0, depth_h = 0;
    bool has_rgbd = export_copy_rgbd_snapshot(rgb_data, depth_data, w, h, depth_w, depth_h);
    int rgb_size = has_rgbd ? static_cast<int>(rgb_data.size()) : -1;
    int depth_size = has_rgbd ? static_cast<int>(depth_data.size()) : -1;
    int step = g_recordingStep.load(std::memory_order_acquire);
    char namebuf[64];
    sprintf_s(namebuf, "step_%06d.bin", step);
    std::string rgb_rel = std::string("RGB/") + namebuf;
    std::string depth_rel = std::string("Depth/") + namebuf;
    std::string rgb_path = std::string(g_recordingSessionDir) + "\\RGB\\" + namebuf;
    std::string depth_path = std::string(g_recordingSessionDir) + "\\Depth\\" + namebuf;
    {
        std::ofstream f(rgb_path, std::ios::binary);
        if (!rgb_data.empty()) f.write(reinterpret_cast<const char*>(rgb_data.data()), rgb_data.size());
    }
    {
        std::ofstream f(depth_path, std::ios::binary);
        if (!depth_data.empty()) f.write(reinterpret_cast<const char*>(depth_data.data()), depth_data.size());
    }
    g_recordingStepsFile
        << "{\"step\":" << step
        << ",\"action\":{\"name\":\"" << action << "\",\"dx\":" << dx << ",\"dy\":" << dy << ",\"dz\":" << dz
        << ",\"drx\":" << drx << ",\"dry\":" << dry << ",\"drz\":" << drz << "}"
        << ",\"pose\":{\"x\":" << pos.x << ",\"y\":" << pos.y << ",\"z\":" << pos.z
        << ",\"rx\":" << rot.x << ",\"ry\":" << rot.y << ",\"rz\":" << rot.z << "}"
        << ",\"rgb\":{\"path\":\"" << rgb_rel << "\",\"width\":" << w << ",\"height\":" << h << ",\"bytes\":" << rgb_size << "}"
        << ",\"depth\":{\"path\":\"" << depth_rel << "\",\"width\":" << depth_w << ",\"height\":" << depth_h << ",\"bytes\":" << depth_size << "}"
        << "}\n";
    g_recordingStepsFile.flush();
    g_recordingStep.store(step + 1, std::memory_order_release);
}

static void create_fire_near_pos(float ox, float oy, float oz) {
    g_fireReady.store(false, std::memory_order_release);
    Vector3 nodePos; float nodeHeading = 0.0f;
    bool ok = PATHFIND::GET_CLOSEST_VEHICLE_NODE_WITH_HEADING(ox, oy, oz, &nodePos, &nodeHeading, 1, 3.0, 0);
    float px = ok ? nodePos.x : ox;
    float py = ok ? nodePos.y : oy;
    float pz = ok ? nodePos.z : oz;
    float gz = pz;
    bool hasGround = GAMEPLAY::GET_GROUND_Z_FOR_3D_COORD(px, py, pz, &gz, false);
    if (!hasGround) gz = pz;

    g_firePos[0] = px;
    g_firePos[1] = py;
    g_firePos[2] = gz;

    Hash vh = GAMEPLAY::GET_HASH_KEY("blista");
    bool v_ok = STREAMING::IS_MODEL_VALID(vh) && STREAMING::IS_MODEL_A_VEHICLE(vh);
    if (v_ok && !STREAMING::HAS_MODEL_LOADED(vh)) {
        STREAMING::REQUEST_MODEL(vh);
        int tries = 0;
        while (!STREAMING::HAS_MODEL_LOADED(vh) && tries < 200) { WAIT(0); tries++; }
    }

    Vehicle created_vehicle = 0;
    if (v_ok && STREAMING::HAS_MODEL_LOADED(vh)) {
        float heading = ok ? nodeHeading : 0.0f;
        created_vehicle = VEHICLE::CREATE_VEHICLE(vh, g_firePos[0], g_firePos[1], g_firePos[2], heading, true, false);
        
        if (created_vehicle != 0) {
            g_spawnedVehicles.push_back(created_vehicle);
            STREAMING::SET_MODEL_AS_NO_LONGER_NEEDED(vh);
            ENTITY::SET_ENTITY_AS_MISSION_ENTITY(created_vehicle, true, true);

            // Step 1: 定位，此时不冻结
            ENTITY::SET_ENTITY_COORDS(created_vehicle, g_firePos[0], g_firePos[1], g_firePos[2], true, false, false, true);
            VEHICLE::SET_VEHICLE_ON_GROUND_PROPERLY(created_vehicle);

            // Step 2: 引擎必须开启，起火链路才能激活
            VEHICLE::SET_VEHICLE_ENGINE_ON(created_vehicle, true, true, false);
            VEHICLE::SET_VEHICLE_HANDBRAKE(created_vehicle, true);

            // Step 3: 施加致命伤害
            VEHICLE::SET_VEHICLE_PETROL_TANK_HEALTH(created_vehicle, -1000.0f);
            VEHICLE::SET_VEHICLE_ENGINE_HEALTH(created_vehicle, -1000.0f);
            VEHICLE::SET_VEHICLE_DAMAGE(created_vehicle, 0.0f, 0.0f, 0.0f, 2000.0f, 5.0f, true);

            // Step 4: 等待伤害状态生效（关键）
            WAIT(500);

            // Add a visible vehicle explosion, then keep persistent fire active.
            FIRE::ADD_EXPLOSION(g_firePos[0], g_firePos[1], g_firePos[2] + 0.8f, 2, 0.6f, true, false, 0.45f);
            WAIT(250);

            // Step 5: 点火
            g_fireId = static_cast<int>(FIRE::START_ENTITY_FIRE(created_vehicle));
            int fid;
            fid = FIRE::START_SCRIPT_FIRE(g_firePos[0], g_firePos[1], g_firePos[2] + 0.5, 25, true);
            g_spawnedFireIds.push_back(fid);
            fid = FIRE::START_SCRIPT_FIRE(g_firePos[0] + 1.0f, g_firePos[1], g_firePos[2] + 0.5, 20, true);
            g_spawnedFireIds.push_back(fid);
            fid = FIRE::START_SCRIPT_FIRE(g_firePos[0] - 1.0f, g_firePos[1], g_firePos[2] + 0.5, 20, true);
            g_spawnedFireIds.push_back(fid);
            fid = FIRE::START_SCRIPT_FIRE(g_firePos[0], g_firePos[1] + 1.0f, g_firePos[2] + 0.5, 20, true);
            g_spawnedFireIds.push_back(fid);
            fid = FIRE::START_SCRIPT_FIRE(g_firePos[0], g_firePos[1] - 1.0f, g_firePos[2] + 0.5, 20, true);
            g_spawnedFireIds.push_back(fid);

            // Step 6: 等待确认起火，最多等约1秒
            int burn_tries = 0;
            while (!FIRE::IS_ENTITY_ON_FIRE(created_vehicle) && burn_tries < 100) {
                WAIT(10);
                burn_tries++;
            }

            // Step 8: 确认起火后再冻结位置
            ENTITY::FREEZE_ENTITY_POSITION(created_vehicle, true);

            LOGI("script", std::string("CREATE_FIRE vehicle fire at ") +
                std::to_string(g_firePos[0]) + "," +
                std::to_string(g_firePos[1]) + "," +
                std::to_string(g_firePos[2]) +
                (FIRE::IS_ENTITY_ON_FIRE(created_vehicle) ? " [ON FIRE]" : " [FIRE FAILED]"));
        } else {
            STREAMING::SET_MODEL_AS_NO_LONGER_NEEDED(vh);
        }
    }
    
    g_fireReady.store(true, std::memory_order_release);
}

static void create_fire_near_camera() {
    Any cam = CAM::GET_RENDERING_CAM();
    Vector3 camPos = CAM::GET_CAM_COORD(cam);
    create_fire_near_pos(camPos.x, camPos.y, camPos.z);
}

static void create_arrest_near_pos(float ox, float oy, float oz) {
    g_arrestReady.store(false, std::memory_order_release);
    Vector3 nodePos; float nodeHeading = 0.0f;
    bool ok = PATHFIND::GET_CLOSEST_VEHICLE_NODE_WITH_HEADING(ox, oy, oz, &nodePos, &nodeHeading, 1, 3.0, 0);
    float px = ok ? nodePos.x : ox;
    float py = ok ? nodePos.y : oy;
    float pz = ok ? nodePos.z : oz;
    float gz = pz;
    bool hasGround = GAMEPLAY::GET_GROUND_Z_FOR_3D_COORD(px, py, pz, &gz, false);
    if (!hasGround) gz = pz;
    g_arrestPos[0] = px;
    g_arrestPos[1] = py;
    g_arrestPos[2] = gz;

    const Hash police_vehicle_models[] = {
        GAMEPLAY::GET_HASH_KEY("police"),
        GAMEPLAY::GET_HASH_KEY("police2"),
        GAMEPLAY::GET_HASH_KEY("sheriff"),
    };
    const int police_vehicle_model_count = static_cast<int>(sizeof(police_vehicle_models) / sizeof(police_vehicle_models[0]));

    Hash police_vehicle_model = 0;
    for (int i = 0; i < police_vehicle_model_count; ++i) {
        Hash h = police_vehicle_models[i];
        if (!STREAMING::IS_MODEL_VALID(h) || !STREAMING::IS_MODEL_A_VEHICLE(h)) continue;
        if (!STREAMING::HAS_MODEL_LOADED(h)) {
            STREAMING::REQUEST_MODEL(h);
            int tries = 0;
            while (!STREAMING::HAS_MODEL_LOADED(h) && tries < 200) { WAIT(0); tries++; }
        }
        if (STREAMING::HAS_MODEL_LOADED(h)) {
            police_vehicle_model = h;
            break;
        }
    }

    Hash cop_model = GAMEPLAY::GET_HASH_KEY("s_m_y_cop_01");
    if (!STREAMING::IS_MODEL_VALID(cop_model)) cop_model = GAMEPLAY::GET_HASH_KEY("s_m_y_sheriff_01");
    if (STREAMING::IS_MODEL_VALID(cop_model) && !STREAMING::HAS_MODEL_LOADED(cop_model)) {
        STREAMING::REQUEST_MODEL(cop_model);
        int tries = 0;
        while (!STREAMING::HAS_MODEL_LOADED(cop_model) && tries < 200) { WAIT(0); tries++; }
    }

    Hash suspect_model = GAMEPLAY::GET_HASH_KEY("g_m_y_lost_01");
    if (!STREAMING::IS_MODEL_VALID(suspect_model)) suspect_model = GAMEPLAY::GET_HASH_KEY("a_m_m_skater_01");
    if (STREAMING::IS_MODEL_VALID(suspect_model) && !STREAMING::HAS_MODEL_LOADED(suspect_model)) {
        STREAMING::REQUEST_MODEL(suspect_model);
        int tries = 0;
        while (!STREAMING::HAS_MODEL_LOADED(suspect_model) && tries < 200) { WAIT(0); tries++; }
    }

    const float vehicle_offsets[][2] = {
        { 3.6f,  1.8f},
        {-3.6f, -1.8f},
    };
    const float vehicle_heading_offsets[] = {155.0f, -25.0f};
    if (police_vehicle_model != 0) {
        for (int i = 0; i < 2; ++i) {
            float x = g_arrestPos[0] + vehicle_offsets[i][0];
            float y = g_arrestPos[1] + vehicle_offsets[i][1];
            float h = nodeHeading + vehicle_heading_offsets[i];
            Vehicle v = VEHICLE::CREATE_VEHICLE(police_vehicle_model, x, y, g_arrestPos[2], h, true, false);
            if (v == 0) continue;
            g_spawnedVehicles.push_back(v);
            ENTITY::SET_ENTITY_AS_MISSION_ENTITY(v, true, true);
            VEHICLE::SET_VEHICLE_ON_GROUND_PROPERLY(v);
            VEHICLE::SET_VEHICLE_ENGINE_ON(v, true, true, false);
            VEHICLE::SET_VEHICLE_HANDBRAKE(v, true);
            VEHICLE::SET_VEHICLE_SIREN(v, true);
            ENTITY::FREEZE_ENTITY_POSITION(v, true);
        }
        STREAMING::SET_MODEL_AS_NO_LONGER_NEEDED(police_vehicle_model);
    } else {
        LOGW("script", "CREATE_ARREST police vehicle model unavailable");
    }

    if (STREAMING::IS_MODEL_VALID(suspect_model) && STREAMING::HAS_MODEL_LOADED(suspect_model)) {
        Ped suspect = PED::CREATE_PED(26, suspect_model, g_arrestPos[0], g_arrestPos[1], g_arrestPos[2], nodeHeading + 180.0f, true, true);
        if (suspect != 0) {
            g_spawnedPeds.push_back(suspect);
            ENTITY::SET_ENTITY_AS_MISSION_ENTITY(suspect, true, true);
            ENTITY::SET_ENTITY_INVINCIBLE(suspect, true);
            ENTITY::FREEZE_ENTITY_POSITION(suspect, true);
            PED::SET_PED_CAN_BE_KNOCKED_OFF_VEHICLE(suspect, false);
            PED::SET_PED_CAN_BE_DRAGGED_OUT(suspect, false);
        }
        STREAMING::SET_MODEL_AS_NO_LONGER_NEEDED(suspect_model);
    } else {
        LOGW("script", "CREATE_ARREST suspect model unavailable");
    }

    const float cop_offsets[][2] = {
        { 1.8f,  0.7f},
        {-1.8f, -0.7f},
        { 0.6f, -1.9f},
        {-0.6f,  1.9f},
    };
    if (STREAMING::IS_MODEL_VALID(cop_model) && STREAMING::HAS_MODEL_LOADED(cop_model)) {
        for (int i = 0; i < 4; ++i) {
            float x = g_arrestPos[0] + cop_offsets[i][0];
            float y = g_arrestPos[1] + cop_offsets[i][1];
            float dx = g_arrestPos[0] - x;
            float dy = g_arrestPos[1] - y;
            float h = atan2f(-dx, dy) * (180.0f / 3.14159f);
            Ped cop = PED::CREATE_PED(6, cop_model, x, y, g_arrestPos[2], h, true, true);
            if (cop == 0) continue;
            g_spawnedPeds.push_back(cop);
            ENTITY::SET_ENTITY_AS_MISSION_ENTITY(cop, true, true);
            ENTITY::SET_ENTITY_INVINCIBLE(cop, true);
            ENTITY::FREEZE_ENTITY_POSITION(cop, true);
            PED::SET_PED_CAN_BE_KNOCKED_OFF_VEHICLE(cop, false);
            PED::SET_PED_CAN_BE_DRAGGED_OUT(cop, false);
        }
        STREAMING::SET_MODEL_AS_NO_LONGER_NEEDED(cop_model);
    } else {
        LOGW("script", "CREATE_ARREST cop model unavailable");
    }

    g_arrestReady.store(true, std::memory_order_release);
    LOGI("script", std::string("CREATE_ARREST police arrest scene at ") + std::to_string(g_arrestPos[0]) + "," + std::to_string(g_arrestPos[1]) + "," + std::to_string(g_arrestPos[2]));
}

static void create_arrest_near_camera() {
    Any cam = CAM::GET_RENDERING_CAM();
    Vector3 camPos = CAM::GET_CAM_COORD(cam);
    create_arrest_near_pos(camPos.x, camPos.y, camPos.z);
}

static void create_accident_near_pos(float ox, float oy, float oz) {
    g_accidentReady.store(false, std::memory_order_release);
    Vector3 nodePos{}; float nodeHeading = 0.0f;
    if (!PATHFIND::GET_CLOSEST_VEHICLE_NODE_WITH_HEADING(ox, oy, oz, &nodePos, &nodeHeading, 1, 3.0, 0)) {
        g_accidentPos[0] = ox; g_accidentPos[1] = oy; g_accidentPos[2] = oz;
        g_accidentReady.store(true, std::memory_order_release);
        LOGE("script", "CREATE_ACCIDENT could not find closest vehicle node");
        return;
    }

    // Clustered multi-vehicle crash scene: spawn several cars near anomaly center and
    // apply heavy body damage to mimic post-collision deformation.
    const Hash vehicle_models[] = {
        GAMEPLAY::GET_HASH_KEY("adder"),
        GAMEPLAY::GET_HASH_KEY("zentorno"),
        GAMEPLAY::GET_HASH_KEY("sultan"),
        GAMEPLAY::GET_HASH_KEY("oracle"),
    };
    const int model_count = static_cast<int>(sizeof(vehicle_models) / sizeof(vehicle_models[0]));

    bool any_model_ready = false;
    for (int i = 0; i < model_count; ++i) {
        Hash h = vehicle_models[i];
        if (!STREAMING::IS_MODEL_VALID(h) || !STREAMING::IS_MODEL_A_VEHICLE(h)) continue;
        if (!STREAMING::HAS_MODEL_LOADED(h)) {
            STREAMING::REQUEST_MODEL(h);
            int tries = 0;
            while (!STREAMING::HAS_MODEL_LOADED(h) && tries < 200) { WAIT(0); tries++; }
        }
        if (STREAMING::HAS_MODEL_LOADED(h)) any_model_ready = true;
    }

    if (!any_model_ready) {
        g_accidentPos[0] = nodePos.x; g_accidentPos[1] = nodePos.y; g_accidentPos[2] = nodePos.z;
        g_accidentReady.store(true, std::memory_order_release);
        LOGW("script", "CREATE_ACCIDENT no valid vehicle model loaded");
        return;
    }

    float gz = nodePos.z;
    GAMEPLAY::GET_GROUND_Z_FOR_3D_COORD(nodePos.x, nodePos.y, nodePos.z, &gz, false);
    g_accidentPos[0] = nodePos.x; g_accidentPos[1] = nodePos.y; g_accidentPos[2] = gz;

    // Relative offsets (meters) to form a dense crash cluster.
    const float offsets[][2] = {
        { 0.0f,  0.0f},
        { 2.8f,  1.0f},
        {-2.6f,  1.2f},
        { 1.2f, -2.8f},
    };
    const float heading_offsets[] = {0.0f, 35.0f, -40.0f, 95.0f};
    const int spawn_count = 4;

    int created = 0;
    for (int i = 0; i < spawn_count; ++i) {
        Hash model = vehicle_models[i % model_count];
        if (!STREAMING::HAS_MODEL_LOADED(model)) continue;

        float x = nodePos.x + offsets[i][0];
        float y = nodePos.y + offsets[i][1];
        float h = nodeHeading + heading_offsets[i];
        Vehicle v = VEHICLE::CREATE_VEHICLE(model, x, y, gz, h, true, false);
        if (v == 0) continue;
        created++;
        g_spawnedVehicles.push_back(v);

        ENTITY::SET_ENTITY_AS_MISSION_ENTITY(v, true, true);
        VEHICLE::SET_VEHICLE_ON_GROUND_PROPERLY(v);
        VEHICLE::SET_VEHICLE_HANDBRAKE(v, true);
        VEHICLE::SET_VEHICLE_ENGINE_ON(v, false, true, false);

        // Simulate damaged shells: multiple impact points + broken windows/doors + dead engine.
        VEHICLE::SET_VEHICLE_ENGINE_HEALTH(v, -2500.0f);
        VEHICLE::SET_VEHICLE_BODY_HEALTH(v, 120.0f);
        VEHICLE::SET_VEHICLE_PETROL_TANK_HEALTH(v, 50.0f);
        VEHICLE::SET_VEHICLE_DAMAGE(v,  1.6f,  0.0f, 0.0f, 1800.0f, 6.0f, true);
        VEHICLE::SET_VEHICLE_DAMAGE(v, -1.4f,  0.4f, 0.0f, 1600.0f, 5.0f, true);
        VEHICLE::SET_VEHICLE_DAMAGE(v,  0.0f, -1.2f, 0.0f, 1400.0f, 5.0f, true);
        VEHICLE::SET_VEHICLE_TYRE_BURST(v, 0, true, 1000.0f);
        VEHICLE::SET_VEHICLE_TYRE_BURST(v, 1, true, 1000.0f);
        VEHICLE::SMASH_VEHICLE_WINDOW(v, 0);
        VEHICLE::SMASH_VEHICLE_WINDOW(v, 1);
        VEHICLE::SET_VEHICLE_DOOR_BROKEN(v, 0, true);
        VEHICLE::SET_VEHICLE_DOOR_BROKEN(v, 1, true);
    }

    for (int i = 0; i < model_count; ++i) {
        Hash h = vehicle_models[i];
        if (STREAMING::HAS_MODEL_LOADED(h)) STREAMING::SET_MODEL_AS_NO_LONGER_NEEDED(h);
    }

    g_accidentReady.store(true, std::memory_order_release);
    if (created > 0) {
        LOGI("script", std::string("CREATE_ACCIDENT clustered vehicles created: ") + std::to_string(created));
    } else {
        LOGW("script", "CREATE_ACCIDENT failed to create vehicles, reporting node position only");
    }
}

static void create_accident_near_camera() {
    Any cam = CAM::GET_RENDERING_CAM();
    Vector3 camPos = CAM::GET_CAM_COORD(cam);
    create_accident_near_pos(camPos.x, camPos.y, camPos.z);
}


static float wrap_angle_deg(float a) {
    while (a > 180.0f) a -= 360.0f;
    while (a < -180.0f) a += 360.0f;
    return a;
}

static float quantize_deg(float a, float step) {
    if (step <= 0.0f) return a;
    return roundf(a / step) * step;
}

static float yaw_to_target_deg(Position3D from, Position3D to) {
    float dx = to.x - from.x;
    float dy = to.y - from.y;
    float yaw = atan2f(-dx, dy) * (180.0f / 3.14159f);
    return yaw;
}

// 检测从当前位置到目标位置是否会发生碰撞
static bool check_collision_raycast(Position3D from, Position3D to) {
    // 使用raycast检查从当前位置到目标位置是否有障碍物
    int raycast_handle = WORLDPROBE::_0x7EE9F5D83DD4F90E(
        from.x, from.y, from.z,
        to.x, to.y, to.z,
        1, // 只检查世界几何体
        0, // 忽略实体
        4  // 标准碰撞检测
    );
    
    // 等待raycast完成
    int result_ready = 0;
    BOOL hit = FALSE;
    Vector3 hit_pos{}, hit_normal{};
    Entity hit_entity = 0;
    
    // 等待raycast结果（最多等待几帧）
    for (int wait_frames = 0; wait_frames < 5; wait_frames++) {
        result_ready = WORLDPROBE::_GET_RAYCAST_RESULT(raycast_handle, &hit, &hit_pos, &hit_normal, &hit_entity);
        if (result_ready != 1) break; // 1表示还在处理中
        WAIT(0);
    }
    
    // 如果raycast击中了什么，说明有碰撞
    if (result_ready == 2 && hit == TRUE) {
        LOGW("script", "Collision detected: raycast hit obstacle at (" + 
             std::to_string(hit_pos.x) + "," + std::to_string(hit_pos.y) + "," + std::to_string(hit_pos.z) + ")");
        return true;
    }
    
    return false;
}

// 删除录制会话的文件夹
static void delete_recording_session() {
    if (g_recordingSessionDir[0] == '\0') return;
    
    std::string session_dir = std::string(g_recordingSessionDir);
    LOGW("script", "Deleting invalid recording session: " + session_dir);
    
    // 关闭文件
    if (g_recordingStepsFile.is_open()) {
        g_recordingStepsFile.close();
    }
    
    // 删除文件夹及其内容（Windows命令）
    std::string delete_cmd = "rmdir /s /q \"" + session_dir + "\"";
    system(delete_cmd.c_str());
    
    // 清空会话目录记录
    std::memset(g_recordingSessionDir, 0, sizeof(g_recordingSessionDir));
    g_recordingEnabled.store(false, std::memory_order_release);
    
    LOGI("script", "Recording session deleted due to collision");
}

static void clear_spawned_entities() {
    LOGI("script", "Clearing spawned vehicles and peds...");

    FIRE::STOP_ENTITY_FIRE(g_fireId);

    for (int fireId : g_spawnedFireIds) {
        FIRE::REMOVE_SCRIPT_FIRE(fireId);
    }
    g_spawnedFireIds.clear();


    // 1. 清理车辆
    for (Vehicle veh : g_spawnedVehicles) {
        if (ENTITY::DOES_ENTITY_EXIST(veh)) {
            ENTITY::SET_ENTITY_AS_MISSION_ENTITY(veh, true, true); // 获取控制权
            ENTITY::DELETE_ENTITY(&veh);
        }
    }
    g_spawnedVehicles.clear();

    // 2. 清理 NPC (Peds)
    for (Ped ped : g_spawnedPeds) {
        if (ENTITY::DOES_ENTITY_EXIST(ped)) {
            ENTITY::SET_ENTITY_AS_MISSION_ENTITY(ped, true, true);
            ENTITY::DELETE_ENTITY(&ped);
        }
    }
    g_spawnedPeds.clear();

    LOGI("script", "Cleanup completed.");
}

// 基于道路节点朝向的起始位置选择函数
static Position3D find_good_start_position(Position3D target, float offset_distance = 50.0f) {
    // 找到离目标点最近的道路节点及其朝向
    Vector3 node_pos;
    float node_heading = 0.0f;
    bool found_node = PATHFIND::GET_CLOSEST_VEHICLE_NODE_WITH_HEADING(target.x, target.y, target.z, &node_pos, &node_heading, 1, 3.0f, 0);

    if (!found_node) {
        LOGW("script", "Could not find closest vehicle node with heading, using fallback position");
        Position3D fallback(target.x + offset_distance, target.y, target.z + 20.0f);
        return fallback;
    }

    // 转换到弧度
    float angle_rad = node_heading * (3.14159f / 180.0f);
    
    // 计算方向向量
    float dir_x = -sinf(angle_rad);
    float dir_y = cosf(angle_rad);

    // 从道路节点沿着道路朝向的反方向偏移指定距离
    float start_x = target.x - dir_x * offset_distance;
    float start_y = target.y - dir_y * offset_distance;

    // 获取地面高度
    float ground_z = node_pos.z;
    if (GAMEPLAY::GET_GROUND_Z_FOR_3D_COORD(start_x, start_y, node_pos.z + 10.0f, &ground_z, false)) {
        // 成功获取地面高度
    }
    else if (GAMEPLAY::GET_GROUND_Z_FOR_3D_COORD(start_x, start_y, 1000.0f, &ground_z, false)) {
        // 从高空获取地面高度
    }
    else {
        // 使用道路节点高度
        ground_z = node_pos.z;
    }

    Position3D start_pos(start_x, start_y, ground_z + 15.0f);

    LOGD("script", "Target: (" + std::to_string(target.x) + "," + std::to_string(target.y) + "," + std::to_string(target.z) + ")");
    LOGD("script", "Start:  (" + std::to_string(start_pos.x) + "," + std::to_string(start_pos.y) + "," + std::to_string(start_pos.z) + ")");
    LOGD("script", "Direction: (" + std::to_string(dir_x) + "," + std::to_string(dir_y) + ")");

    return start_pos;
}


static void setup_player_for_collection(bool protect) {
    Ped player = PLAYER::PLAYER_PED_ID();
    if (!player) return;
    if (protect) {
        ENTITY::SET_ENTITY_INVINCIBLE(player, true);
        ENTITY::SET_ENTITY_VISIBLE(player, false, false);
        ENTITY::SET_ENTITY_CAN_BE_DAMAGED(player, false);
        ENTITY::SET_ENTITY_PROOFS(player, true, true, true, true, true, true, true, true);
        PLAYER::SET_MAX_WANTED_LEVEL(0);
        PLAYER::SET_POLICE_IGNORE_PLAYER(player, true);
    } else {
        ENTITY::SET_ENTITY_INVINCIBLE(player, false);
        ENTITY::SET_ENTITY_VISIBLE(player, true, false);
        ENTITY::SET_ENTITY_CAN_BE_DAMAGED(player, true);
        PLAYER::SET_POLICE_IGNORE_PLAYER(player, false);
    }
}

static Position3D get_random_road_node() {
    Entity playerPed = PLAYER::PLAYER_PED_ID();
    Vector3 playerPos = ENTITY::GET_ENTITY_COORDS(playerPed, true);
    const float MIN_RADIUS = 100.0f;
    const float MAX_RADIUS = 400.0f;

    for (int attempts = 0; attempts < 20; attempts++) {
        float rnd_r = (float)rand() / (float)RAND_MAX;
        float rnd_a = (float)rand() / (float)RAND_MAX;
        float radius = MIN_RADIUS + rnd_r * (MAX_RADIUS - MIN_RADIUS);
        float angle = rnd_a * 6.2831853f;
        float target_x = playerPos.x + radius * cosf(angle);
        float target_y = playerPos.y + radius * sinf(angle);
        int nth = (rand() % 4) + 1;

        Vector3 node_pos;
        if (PATHFIND::GET_NTH_CLOSEST_VEHICLE_NODE(target_x, target_y, playerPos.z, nth, &node_pos, 1, 3.0f, 0)) {
            if (node_pos.x != 0.0f && node_pos.y != 0.0f) {
                return Position3D(node_pos.x, node_pos.y, node_pos.z + 1.0f);
            }
        }
    }
    LOGW("script", "Fallback to player pos");
    return Position3D(playerPos.x, playerPos.y, playerPos.z);
}

static bool generate_scenario(AutoCollectEvent event_type, Position3D& out_target, Position3D& out_start, float& out_yaw) {
    clear_spawned_entities();

    Position3D target_node = get_random_road_node();

    bool was_camera = (scriptStatus == cameraMode);
    if (was_camera) { StopCamera(); scriptStatus = scriptStop; WAIT(500); }

    Ped player = PLAYER::PLAYER_PED_ID();
    if (player) {
        ENTITY::SET_ENTITY_COORDS(player, target_node.x, target_node.y, target_node.z, true, false, false, true);
        WAIT(1000);
    }
    
    // 等待地图资源和碰撞网格流式加载完成，防止异常事件生成时穿模或找不到路网节点
    WAIT(5000); 

    if (was_camera) { startNewCamera(); scriptStatus = cameraMode; WAIT(500); }

    if (event_type == AUTO_EVENT_FIRE) {
        create_fire_near_pos(target_node.x, target_node.y, target_node.z);
        WAIT(1000);
        if (!g_fireReady.load(std::memory_order_acquire)) return false;
        out_target = Position3D(g_firePos[0], g_firePos[1], g_firePos[2] + 1.0f);
    } else if (event_type == AUTO_EVENT_ARREST) {
        create_arrest_near_pos(target_node.x, target_node.y, target_node.z);
        WAIT(1000);
        if (!g_arrestReady.load(std::memory_order_acquire)) return false;
        out_target = Position3D(g_arrestPos[0], g_arrestPos[1], g_arrestPos[2] + 1.0f);
    } else {
        create_accident_near_pos(target_node.x, target_node.y, target_node.z);
        WAIT(1000);
        if (!g_accidentReady.load(std::memory_order_acquire)) return false;
        out_target = Position3D(g_accidentPos[0], g_accidentPos[1], g_accidentPos[2] + 1.0f);
    }

    out_start = find_good_start_position(out_target, 50.0f);

    Any cam = CAM::GET_RENDERING_CAM();
    CAM::SET_CAM_COORD(cam, out_start.x, out_start.y, out_start.z);
    if (false) {
        out_yaw = quantize_deg(yaw_to_target_deg(out_start, out_target), YAW_STEPSIZE);
    }
    else {
        out_yaw = quantize_deg((float)rand() / (float)RAND_MAX * 360.0f, YAW_STEPSIZE);
    }
    CAM::SET_CAM_ROT(cam, 0.0f, 0.0f, out_yaw, 2);

    WAIT(100);
    Vector3 actual_cam_pos = CAM::GET_CAM_COORD(cam);
    out_start = Position3D(actual_cam_pos.x, actual_cam_pos.y, actual_cam_pos.z);

    return true;
}

static void run_auto_collect(AutoCollectEvent event_type) {
    Position3D target, start;
    float yaw;

    if (!generate_scenario(event_type, target, start, yaw)) {
        LOGW("script", "Failed to generate scenario");
        return;
    }

    const char* task = "";
    if (event_type == AUTO_EVENT_FIRE) task = "find the exploded car";
    else if (event_type == AUTO_EVENT_ARREST) task = "find the police arrest scene";
    else task = "find crashed cars";

    start_recording_session(task);

    int maxSteps = 100;
    bool reached = false;
    int step = 0;

    for (; step < maxSteps; step++) {
        Any cam = CAM::GET_RENDERING_CAM();
        Vector3 cam_pos = CAM::GET_CAM_COORD(cam);
        Vector3 rot = CAM::GET_CAM_ROT(cam, 2);

        Position3D pos(cam_pos.x, cam_pos.y, cam_pos.z);
        float dist = pos.distance_to(target);
        float dz = target.z - pos.z;

        float desiredYaw = quantize_deg(yaw_to_target_deg(pos, target), YAW_STEPSIZE);
        float delta = wrap_angle_deg(desiredYaw - rot.z);

        if (fabsf(delta) > YAW_STEPSIZE * 2 / 3) {
            float step_val = (delta > 0.0f) ? YAW_STEPSIZE : -YAW_STEPSIZE;
            record_step(delta > 0.0f ? "AUTO_YAW_LEFT" : "AUTO_YAW_RIGHT", 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, step_val);
            rotateCameraDelta(0.0f, 0.0f, step_val);
        } 
        else if (dist <= STEPSIZE * 4.0f && fabsf(dz) <= STEPSIZE) { 
            reached = true; 
            break; 
        }
        else if (fabsf(dz) > STEPSIZE) {
            float z_step = (dz > 0.0f) ? STEPSIZE : -STEPSIZE;
            Position3D next_pos(pos.x, pos.y, pos.z + z_step);
            if (check_collision_raycast(pos, next_pos)) goto STOP_COLLISION;
            record_step(z_step > 0.0f ? "AUTO_UP" : "AUTO_DOWN", 0.0f, 0.0f, z_step, 0.0f, 0.0f, 0.0f);
            moveCameraDelta(0.0f, 0.0f, z_step);
        } 
        else {
            float forward_rad = rot.z * (3.14159f / 180.0f);
            Position3D next_pos(pos.x + cosf(forward_rad) * STEPSIZE, pos.y + sinf(forward_rad) * STEPSIZE, pos.z);
            if (check_collision_raycast(pos, next_pos)) goto STOP_COLLISION;
            record_step("AUTO_FORWARD", STEPSIZE, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
            moveCameraDelta(STEPSIZE, 0.0f, 0.0f);
        }
        WAIT(0);
    }

    record_step(reached ? "AUTO_STOP_REACHED" : "AUTO_STOP_MAXSTEPS", 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    stop_recording_session(event_type, target, start, yaw, step + 1, task);
    WAIT(1000);
    return;

STOP_COLLISION:
    delete_recording_session();
}

static void run_automated_collection(AutoCollectEvent event_type, int collection_count = 10) {
    static bool automated_active = false;
    if (automated_active) {
        LOGW("script", "Automated collection already running");
        return;
    }
    automated_active = true;
    LOGI("script", "Starting automated collection: " + std::to_string(collection_count) + " samples");

    setup_player_for_collection(true);
    srand(static_cast<unsigned int>(time(nullptr)));

    for (int i = 0; i < collection_count; i++) {
        LOGI("script", "Starting automated collection attempt " + std::to_string(i + 1) + "/" + std::to_string(collection_count));
        run_auto_collect(event_type);
        clear_spawned_entities();
        WAIT(1000);
    }

    setup_player_for_collection(false);
    automated_active = false;
    LOGI("script", "Automated collection completed");
}

static void run_continuous_manual_collection(AutoCollectEvent event_type, int collection_count = 10) {
    static bool manual_active = false;
    if (manual_active) {
        LOGW("script", "Continuous manual collection already running");
        return;
    }
    manual_active = true;
    LOGI("script", "Starting continuous manual collection: " + std::to_string(collection_count) + " samples");

    setup_player_for_collection(true);
    srand(static_cast<unsigned int>(time(nullptr)));

    for (int i = 0; i < collection_count; i++) {
        LOGI("script", "Starting manual collection attempt " + std::to_string(i + 1) + "/" + std::to_string(collection_count));
        
        Position3D target, start;
        float yaw;

        if (!generate_scenario(event_type, target, start, yaw)) {
            LOGW("script", "Failed to generate scenario, skipping");
            continue;
        }

        const char* task = "";
        if (event_type == AUTO_EVENT_FIRE) task = "find the exploded car";
        else if (event_type == AUTO_EVENT_ARREST) task = "find the police arrest scene";
        else task = "find crashed cars";

        start_recording_session(task);
        LOGI("script", "Scenario ready. Please manually navigate to the target.");

        bool reached = false;
        int steps = 0;

        while (!reached) {
            Any reach_cam = CAM::GET_RENDERING_CAM();
            Vector3 reach_pos = CAM::GET_CAM_COORD(reach_cam);
            float reach_dist = Position3D(reach_pos.x, reach_pos.y, reach_pos.z).distance_to(target);
            float reach_dz = target.z - reach_pos.z;
            if (reach_dist <= STEPSIZE * 4.0f && fabsf(reach_dz) <= STEPSIZE) {
                reached = true;
                record_step("AUTO_STOP_REACHED", 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
                LOGI("script", "Target reached manually!");
                continue;
            }

            if (F11.isKeyDown()) {
                stop_recording_session();
                StopCamera();
                scriptStatus = scriptStop;
                goto EXIT_MANUAL_COLLECTION;
            }
            
            // 提供手动确认到达的按键，避免因微小距离偏差卡死在当前场景
            if (F7.isKeyDown()) {
                record_step("AUTO_STOP_FAILED", 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
                LOGI("script", "manually failed.");
                reached = true;
                continue;
            }

            bool moved = false;

            Any cam = CAM::GET_RENDERING_CAM();
            Vector3 cam_pos = CAM::GET_CAM_COORD(cam);
            Vector3 cam_rot = CAM::GET_CAM_ROT(cam, 2);

            Position3D cur_pos(cam_pos.x, cam_pos.y, cam_pos.z);
            Position3D next_pos;

            if (W.isKeyDown()) {
                next_pos = Position3D(cur_pos.x + cosf(cam_rot.z * (3.14159f / 180.0f)) * STEPSIZE, cur_pos.y + sinf(cam_rot.z * (3.14159f / 180.0f)) * STEPSIZE, cur_pos.z);
                if (check_collision_raycast(cur_pos, next_pos)) continue;
                record_step("AUTO_FORWARD", STEPSIZE, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
                moveCameraDelta(STEPSIZE, 0.0f, 0.0f);
                moved = true;
            } else if (shift.isKeyDown()) {
                next_pos = Position3D(cur_pos.x, cur_pos.y, cur_pos.z + STEPSIZE);
                if (check_collision_raycast(cur_pos, next_pos)) continue;
                record_step("AUTO_UP", 0.0f, 0.0f, STEPSIZE, 0.0f, 0.0f, 0.0f);
                moveCameraDelta(0.0f, 0.0f, STEPSIZE);
                moved = true;
            } else if (ctrl.isKeyDown()) {
                next_pos = Position3D(cur_pos.x, cur_pos.y, cur_pos.z - STEPSIZE);
                if (check_collision_raycast(cur_pos, next_pos)) continue;
                record_step("AUTO_DOWN", 0.0f, 0.0f, -STEPSIZE, 0.0f, 0.0f, 0.0f);
                moveCameraDelta(0.0f, 0.0f, -STEPSIZE);
                moved = true;
            } else if (Q.isKeyDown()) {
                record_step("AUTO_YAW_LEFT", 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, YAW_STEPSIZE);
                rotateCameraDelta(0.0f, 0.0f, YAW_STEPSIZE);
                moved = true;
            } else if (E.isKeyDown()) {
                record_step("AUTO_YAW_RIGHT", 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, -YAW_STEPSIZE);
                rotateCameraDelta(0.0f, 0.0f, -YAW_STEPSIZE);
                moved = true;
            }

            if (moved) {
                steps++;
            }
            WAIT(0);
        }

        stop_recording_session(event_type, target, start, yaw, steps + 1, task);
        clear_spawned_entities();
        WAIT(1000);
    }

EXIT_MANUAL_COLLECTION:
    setup_player_for_collection(false);
    manual_active = false;
    LOGI("script", "Continuous manual collection finished.");
}

void scriptMain()
{

	int sleepTime = 0;
	//setStatusText("DroneSim start fine!!!");
    
    InitializeServerV2();
    LOGI("script", "DroneSim script started");

    //setStatusText("Awaiting client commands.");
    LOGI("script", "Awaiting client commands");

    ensure_dir("data");

	while (true)
	{
        // 在循环最开始就记录日志，确认循环是否在运行
        static int loop_counter = 0;
        loop_counter++;
        if (loop_counter % 100 == 0) { // 每100次循环记录一次
            LOGD("script", std::string("Main loop iteration: ") + std::to_string(loop_counter));
        }
        
        if (scriptStatus == cameraMode)
        {
            if (W.isKeyDown())
            {
                record_step("AUTO_FORWARD", STEPSIZE, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
                moveCameraDelta(STEPSIZE, 0.0f, 0.0f);
            }
            if (shift.isKeyDown())
            {
                record_step("AUTO_UP", 0.0f, 0.0f, STEPSIZE, 0.0f, 0.0f, 0.0f);
                moveCameraDelta(0.0f, 0.0f, STEPSIZE);
            }
            if (ctrl.isKeyDown())
            {
                record_step("AUTO_DOWN", 0.0f, 0.0f, -STEPSIZE, 0.0f, 0.0f, 0.0f);
                moveCameraDelta(0.0f, 0.0f, -STEPSIZE);
            }
            if (Q.isKeyDown())
            {
                record_step("AUTO_YAW_LEFT", 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, YAW_STEPSIZE);
                rotateCameraDelta(0.0f, 0.0f, YAW_STEPSIZE);
            }
            if (E.isKeyDown())
            {
                record_step("AUTO_YAW_RIGHT", 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, -YAW_STEPSIZE);
                rotateCameraDelta(0.0f, 0.0f, -YAW_STEPSIZE);
            }
            if (F12.isKeyDown())
            {
                LOGD("script", "F12 pressed - Starting automated batch collection");
                // 默认使用火灾进行批量自动采集，你可以根据需求改成 AUTO_EVENT_ACCIDENT 等
                run_automated_collection(AUTO_EVENT_FIRE, 100);
            }
            if (F6.isKeyDown())
            {
                LOGD("script", "F6 pressed - Starting continuous manual collection");
                // 默认使用火灾进行连贯手动采集，你可以根据需求改成 AUTO_EVENT_ACCIDENT 等
                run_continuous_manual_collection(AUTO_EVENT_FIRE, 100);
            }
            if (F7.isKeyDown())
            {
                record_step("AUTO_STOP_REACHED", 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
                stop_recording_session();
            }
        }
        if (F10.isKeyDown()) {
            startNewCamera();
            scriptStatus = cameraMode;
            // setStatusText("Camera mode enabled.");
            LOGI("script", "Camera created and mode enabled");
        }
        if (F11.isKeyDown()) {
            StopCamera();
            scriptStatus = scriptStop;
            stop_recording_session();
            LOGI("script", "Camera stopped and returned to player view");
        }

        std::string cmd;
        if (!try_dequeue_command(cmd)) { 
            // 添加心跳日志，确认主循环在运行
            static int heartbeat_counter = 0;
            heartbeat_counter++;
            if (heartbeat_counter % 1000 == 0) { // 每1000次循环记录一次
                LOGD("script", std::string("Script main loop heartbeat: ") + std::to_string(heartbeat_counter) + ", queue size: " + std::to_string(command_queue_size()));
            }
            
            WAIT(0); 
            continue; 
        }
        
        // 添加命令处理的调试日志
        LOGD("script", std::string("Processing command: ") + cmd + ", scriptStatus: " + std::to_string(scriptStatus));
        
        if (cmd == "CREATE_CAMERA")
        {
            startNewCamera();
            scriptStatus = cameraMode;
            // setStatusText("Camera mode enabled.");
            LOGI("script", "Camera created and mode enabled");
        }
        else if (cmd == "STOP_CAMERA")
        {
            StopCamera();
            scriptStatus = scriptStop;
            stop_recording_session();
            // setStatusText("Camera mode disabled.");
            LOGI("script", "Camera stopped and returned to player view");
        }
        else if (cmd == "CREATE_FIRE")
        {
            create_fire_near_camera();
        }
        else if (cmd == "CREATE_ARREST")
        {
            create_arrest_near_camera();
        }
        else if (cmd == "CREATE_ACCIDENT")
        {
            create_accident_near_camera();
        }
        else if (cmd == "GET_POSE")
        {
            LOGD("script", "GET_POSE command received, starting processing");
            Vector3 cam_pos{}; Vector3 cam_rot{};
            Any cam = CAM::GET_RENDERING_CAM();
            LOGD("script", std::string("GET_POSE: Got camera handle: ") + std::to_string(cam));
            
            if (cam != 0) {
                cam_pos = CAM::GET_CAM_COORD(cam);
                LOGD("script", std::string("GET_POSE: Got position: ") + std::to_string(cam_pos.x) + "," + std::to_string(cam_pos.y) + "," + std::to_string(cam_pos.z));
                
                cam_rot = CAM::GET_CAM_ROT(cam, 2);
                LOGD("script", std::string("GET_POSE: Got rotation: ") + std::to_string(cam_rot.x) + "," + std::to_string(cam_rot.y) + "," + std::to_string(cam_rot.z));
                
                // 转换为Position3D进行处理
                Position3D pos(cam_pos.x, cam_pos.y, cam_pos.z);
                Position3D rot(cam_rot.x, cam_rot.y, cam_rot.z);
                
                g_pose[0]=pos.x; g_pose[1]=pos.y; g_pose[2]=pos.z;
                g_pose[3]=rot.x; g_pose[4]=rot.y; g_pose[5]=rot.z;
                g_poseReady.store(true, std::memory_order_release);
                LOGD("script", std::string("GET_POSE completed successfully: ") + std::to_string(g_pose[0]) + " " + std::to_string(g_pose[1]) + " " + std::to_string(g_pose[2]) + " " + std::to_string(g_pose[3]) + " " + std::to_string(g_pose[4]) + " " + std::to_string(g_pose[5]));
            } else {
                LOGE("script", "GET_POSE: No active camera found (cam handle is 0)");
                // 即使失败也设置g_poseReady，避免server_v2无限等待
                g_poseReady.store(true, std::memory_order_release);
            }
        }
        else if (scriptStatus == cameraMode) {
            if (cmd == "REQUEST")
            {
                LOGD("script", "start capture");
                makeCmdStart();
            }
            else if (cmd.rfind("MOVE ", 0) == 0)
            {
                auto s = cmd.substr(5);
                std::stringstream ss(s);
                float dx=0,dy=0,dz=0; ss >> dx >> dy >> dz;
                moveCameraDelta(dx,dy,dz);
                LOGI("script", std::string("MOVE ") + std::to_string(dx) + "," + std::to_string(dy) + "," + std::to_string(dz));
            }
            else if (cmd.rfind("ROTATE ", 0) == 0)
            {
                auto s = cmd.substr(7);
                std::stringstream ss(s);
                float rx=0,ry=0,rz=0; ss >> rx >> ry >> rz;
                rotateCameraDelta(rx,ry,rz);
                LOGI("script", std::string("ROTATE ") + std::to_string(rx) + "," + std::to_string(ry) + "," + std::to_string(rz));
            }
            else if (cmd.rfind("SETFOV:", 0) == 0)
            {
                float fov = std::stof(cmd.substr(7));
                Any cam = CAM::GET_RENDERING_CAM();
                if (cam)
                {
                    CAM::SET_CAM_FOV(cam, fov);
                    LOGI("script", std::string("Set FOV to ") + std::to_string(fov));
                }
            }
            else if (cmd.rfind("SET_TIME ", 0) == 0)
            {
                std::stringstream ss(cmd.substr(9)); int h=12,m=0,s=0; ss>>h>>m>>s;
                TIME::SET_CLOCK_TIME(h, m, s);
                LOGI("script", std::string("Set time to ")+std::to_string(h)+":"+std::to_string(m)+":"+std::to_string(s));
            }
            else if (cmd.rfind("SET_WEATHER ", 0) == 0)
            {
                std::string name = cmd.substr(12);
                GAMEPLAY::CLEAR_WEATHER_TYPE_PERSIST();
                GAMEPLAY::SET_WEATHER_TYPE_NOW_PERSIST((char*)name.c_str());
                LOGI("script", std::string("Set weather to ")+name);
            }
            else if (cmd.rfind("SET_POSTURE ", 0) == 0)
            {
                auto s = cmd.substr(12);
                std::stringstream ss(s);
                float x=0, y=0, z=0, rx=0, ry=0, rz=0;
                ss >> x >> y >> z >> rx >> ry >> rz;
                
                Any cam = CAM::GET_RENDERING_CAM();
                if (cam) {
                    CAM::SET_CAM_COORD(cam, x, y, z);
                    CAM::SET_CAM_ROT(cam, rx, ry, rz, 2);
                    LOGI("script", std::string("Set posture to pos(") + std::to_string(x) + "," + std::to_string(y) + "," + std::to_string(z) + 
                         ") rot(" + std::to_string(rx) + "," + std::to_string(ry) + "," + std::to_string(rz) + ")");
                }
            }
        }
        else if (cmd.rfind("TELEPORT_PLAYER ", 0) == 0)
        {
            auto s = cmd.substr(16);
            std::stringstream ss(s);
            float x=0, y=0, z=0;
            ss >> x >> y >> z;
            
            // Get player ped
            Ped player = PLAYER::PLAYER_PED_ID();
            if (player) {
                // Temporarily switch to player view for teleportation
                bool was_camera_mode = (scriptStatus == cameraMode);
                if (was_camera_mode) {
                    StopCamera();
                    scriptStatus = scriptStop;
                    WAIT(100);  // Wait for camera to stop
                }
                
                // Make player invincible and invisible during scenario setup.
                PLAYER::SET_PLAYER_INVINCIBLE(PLAYER::PLAYER_ID(), true);
                ENTITY::SET_ENTITY_VISIBLE(player, false, false);
                ENTITY::SET_ENTITY_CAN_BE_DAMAGED(player, false);
                ENTITY::SET_ENTITY_PROOFS(player, true, true, true, true, true, true, true, true);
                PLAYER::SET_MAX_WANTED_LEVEL(0);
                PLAYER::SET_POLICE_IGNORE_PLAYER(player, true);
                
                // Teleport player directly to anomaly position
                ENTITY::SET_ENTITY_COORDS(player, x, y, z, true, false, false, true);
                WAIT(100);  // Wait for teleportation to complete
                
                // Switch back to camera mode if we were in camera mode
                if (was_camera_mode) {
                    startNewCamera();
                    scriptStatus = cameraMode;
                    WAIT(100);  // Wait for camera to start
                }
                
                LOGI("script", std::string("Teleported player to anomaly center (") + std::to_string(x) + "," + std::to_string(y) + "," + std::to_string(z) + ") - invincible and invisible");
            } else {
                LOGW("script", "Failed to get player ped for teleportation");
            }
        }
        else if (cmd == "RESTORE_PLAYER")
        {
            // Restore player to normal state after scenario interaction.
            Ped player = PLAYER::PLAYER_PED_ID();
            if (player) {
                // Temporarily switch to player view for restoration
                bool was_camera_mode = (scriptStatus == cameraMode);
                if (was_camera_mode) {
                    StopCamera();
                    scriptStatus = scriptStop;
                    WAIT(100);  // Wait for camera to stop
                }
                
                // Restore visibility and mortality
                ENTITY::SET_ENTITY_VISIBLE(player, true, false);
                PLAYER::SET_PLAYER_INVINCIBLE(PLAYER::PLAYER_ID(), false);
                
                // Teleport player to a safe surface location away from anomaly
                Any cam = CAM::GET_RENDERING_CAM();
                Position3D safe_pos;
                
                if (cam) {
                    Vector3 cam_pos = CAM::GET_CAM_COORD(cam);
                    float ground_z = cam_pos.z;
                    GAMEPLAY::GET_GROUND_Z_FOR_3D_COORD(cam_pos.x, cam_pos.y, cam_pos.z, &ground_z, false);
                    // Move player 20m away from camera position
                    safe_pos.x = cam_pos.x + 20.0f;
                    safe_pos.y = cam_pos.y + 20.0f;
                    safe_pos.z = ground_z + 1.0f;
                } else {
                    // Fallback position if no camera
                    safe_pos.x = 0.0f;
                    safe_pos.y = 0.0f;
                    safe_pos.z = 30.0f;
                }
                
                ENTITY::SET_ENTITY_COORDS(player, safe_pos.x, safe_pos.y, safe_pos.z, true, false, false, true);
                WAIT(100);  // Wait for teleportation to complete
                
                // Switch back to camera mode if we were in camera mode
                if (was_camera_mode) {
                    startNewCamera();
                    scriptStatus = cameraMode;
                    WAIT(100);  // Wait for camera to start
                }
                
                LOGI("script", std::string("Player restored to normal state at (") + std::to_string(safe_pos.x) + "," + std::to_string(safe_pos.y) + "," + std::to_string(safe_pos.z) + ")");
            }
        }
		WAIT(0);
	}
}
