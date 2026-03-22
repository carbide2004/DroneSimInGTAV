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
#include "command_queue.h"
 
volatile bool g_poseReady = false;
float g_pose[6] = {0};

volatile bool g_accidentReady = false;
float g_accidentPos[3] = {0};

volatile bool g_recordingEnabled = false;
volatile int g_recordingStep = 0;
char g_recordingSessionDir[260] = {0};
char g_recordingRequestedSession[128] = {0};
char g_recordingRequestedTask[256] = {0};

volatile bool g_fireReady = false;
float g_firePos[3] = {0};
int g_fireId = -1;

volatile bool g_fightReady = false;
float g_fightPos[3] = {0};

// Verification mode variables
static bool g_verificationMode = false;
static int g_verificationSteps = 0;
static Vector3 g_anomalyPos;
static std::string g_anomalyType;

// Fire maintenance system
static bool g_fireMaintenanceActive = false;
static Vector3 g_fireMaintenancePos;
static Vehicle g_fireVehicle = 0;
static int g_fireMaintenanceTimer = 0;

scriptStatusEnum scriptStatus = scriptStop;

extern volatile catchState cmdToCatch;

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

static void write_metadata(const std::string& base, const std::string& task) {
    if (base.empty()) return;
    std::ofstream f(base + "\\metadata.jsonl", std::ios::out | std::ios::binary);
    if (!f.is_open()) return;
    std::string t = task;
    f << "{\"task\":\"" << json_escape(t) << "\"}\n";
    f.flush();
}

static void start_recording_session(const char* task) {
    if (g_recordingEnabled) return;
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
    g_recordingStep = 0;
    if (g_recordingStepsFile.is_open()) g_recordingStepsFile.close();
    g_recordingStepsFile.open(base + "\\steps.jsonl", std::ios::out | std::ios::binary);
    g_recordingEnabled = g_recordingStepsFile.is_open();
    if (g_recordingEnabled) {
        std::string task_str;
        if (task && task[0] != '\0') task_str = std::string(task);
        else if (g_recordingRequestedTask[0] != '\0') task_str = std::string(g_recordingRequestedTask);
        write_metadata(base, task_str);
    }
    if (g_recordingEnabled) LOGI("script", std::string("Recording started: ") + g_recordingSessionDir);
    else LOGE("script", "Recording start failed");
}

static void stop_recording_session() {
    if (!g_recordingEnabled) return;
    g_recordingEnabled = false;
    if (g_recordingStepsFile.is_open()) {
        g_recordingStepsFile.flush();
        g_recordingStepsFile.close();
    }
    LOGI("script", "Recording stopped");
}

static void record_step(const char* action, float dx, float dy, float dz, float drx, float dry, float drz) {
    if (!g_recordingEnabled) return;
    if (g_recordingSessionDir[0] == '\0') return;
    if (!g_recordingStepsFile.is_open()) return;
    Any cam = CAM::GET_RENDERING_CAM();
    Vector3 pos = CAM::GET_CAM_COORD(cam);
    Vector3 rot = CAM::GET_CAM_ROT(cam, 2);
    makeCmdStart();
    int tries = 0;
    while (cmdToCatch != catchStop && tries < 6000) { WAIT(0); tries++; }
    void* rgb_ptr = nullptr; int rgb_size = export_get_color_buffer(&rgb_ptr);
    void* depth_ptr = nullptr; int depth_size = export_get_depth_buffer(&depth_ptr);
    int w = export_get_last_color_width();
    int h = export_get_last_color_height();
    int step = g_recordingStep;
    char namebuf[64];
    sprintf_s(namebuf, "step_%06d.bin", step);
    std::string rgb_rel = std::string("RGB/") + namebuf;
    std::string depth_rel = std::string("Depth/") + namebuf;
    std::string rgb_path = std::string(g_recordingSessionDir) + "\\RGB\\" + namebuf;
    std::string depth_path = std::string(g_recordingSessionDir) + "\\Depth\\" + namebuf;
    {
        std::ofstream f(rgb_path, std::ios::binary);
        if (rgb_size > 0 && rgb_ptr) f.write(reinterpret_cast<const char*>(rgb_ptr), rgb_size);
    }
    {
        std::ofstream f(depth_path, std::ios::binary);
        if (depth_size > 0 && depth_ptr) f.write(reinterpret_cast<const char*>(depth_ptr), depth_size);
    }
    g_recordingStepsFile
        << "{\"step\":" << step
        << ",\"action\":{\"name\":\"" << action << "\",\"dx\":" << dx << ",\"dy\":" << dy << ",\"dz\":" << dz
        << ",\"drx\":" << drx << ",\"dry\":" << dry << ",\"drz\":" << drz << "}"
        << ",\"pose\":{\"x\":" << pos.x << ",\"y\":" << pos.y << ",\"z\":" << pos.z
        << ",\"rx\":" << rot.x << ",\"ry\":" << rot.y << ",\"rz\":" << rot.z << "}"
        << ",\"rgb\":{\"path\":\"" << rgb_rel << "\",\"width\":" << w << ",\"height\":" << h << ",\"bytes\":" << rgb_size << "}"
        << ",\"depth\":{\"path\":\"" << depth_rel << "\",\"width\":" << export_get_last_depth_width() << ",\"height\":" << export_get_last_depth_height() << ",\"bytes\":" << depth_size << "}"
        << "}\n";
    g_recordingStepsFile.flush();
    g_recordingStep = step + 1;
}

static void maintain_fire() {
    if (!g_fireMaintenanceActive) return;
    
    g_fireMaintenanceTimer++;
    
    // Check and restart fires every 5 seconds (300 frames at 60fps)
    if (g_fireMaintenanceTimer % 300 == 0) {
        // Restart vehicle fire if vehicle exists and is not on fire
        if (g_fireVehicle != 0 && ENTITY::DOES_ENTITY_EXIST(g_fireVehicle)) {
            if (!FIRE::IS_ENTITY_ON_FIRE(g_fireVehicle)) {
                FIRE::START_ENTITY_FIRE(g_fireVehicle);
                LOGI("script", "Restarted vehicle fire");
            }
        }
        
        // Restart script fires at maintenance position
        FIRE::START_SCRIPT_FIRE(g_fireMaintenancePos.x, g_fireMaintenancePos.y, g_fireMaintenancePos.z, 25, true);
        FIRE::START_SCRIPT_FIRE(g_fireMaintenancePos.x + 1.0f, g_fireMaintenancePos.y, g_fireMaintenancePos.z, 20, true);
        FIRE::START_SCRIPT_FIRE(g_fireMaintenancePos.x - 1.0f, g_fireMaintenancePos.y, g_fireMaintenancePos.z, 20, true);
        FIRE::START_SCRIPT_FIRE(g_fireMaintenancePos.x, g_fireMaintenancePos.y + 1.0f, g_fireMaintenancePos.z, 20, true);
        FIRE::START_SCRIPT_FIRE(g_fireMaintenancePos.x, g_fireMaintenancePos.y - 1.0f, g_fireMaintenancePos.z, 20, true);
        
        LOGD("script", "Fire maintenance cycle completed");
    }
}

static void start_fire_maintenance(float x, float y, float z, Vehicle vehicle = 0) {
    g_fireMaintenanceActive = true;
    g_fireMaintenancePos.x = x;
    g_fireMaintenancePos.y = y;
    g_fireMaintenancePos.z = z;
    g_fireVehicle = vehicle;
    g_fireMaintenanceTimer = 0;
    LOGI("script", std::string("Started fire maintenance at (") + std::to_string(x) + "," + std::to_string(y) + "," + std::to_string(z) + ")");
}

static void stop_fire_maintenance() {
    g_fireMaintenanceActive = false;
    g_fireVehicle = 0;
    g_fireMaintenanceTimer = 0;
    LOGI("script", "Stopped fire maintenance");
}
static void create_fire_near_pos(float ox, float oy, float oz) {
    g_fireReady = false;
    Vector3 nodePos; float nodeHeading = 0.0f;
    bool ok = PATHFIND::GET_CLOSEST_VEHICLE_NODE_WITH_HEADING(ox, oy, oz, &nodePos, &nodeHeading, 1, 3.0, 0);
    float px = ok ? nodePos.x : ox;
    float py = ok ? nodePos.y : oy;
    float pz = ok ? nodePos.z : oz;
    float gz = pz;
    bool hasGround = GAMEPLAY::GET_GROUND_Z_FOR_3D_COORD(px, py, pz, &gz, false);
    if (!hasGround) gz = pz;
    
    // Use exact position instead of vehicle position to avoid offset
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
        STREAMING::SET_MODEL_AS_NO_LONGER_NEEDED(vh);
        ENTITY::SET_ENTITY_AS_MISSION_ENTITY(created_vehicle, true, true);
        
        // Ensure vehicle stays at exact position
        ENTITY::FREEZE_ENTITY_POSITION(created_vehicle, true);
        VEHICLE::SET_VEHICLE_ON_GROUND_PROPERLY(created_vehicle);
        VEHICLE::SET_VEHICLE_HANDBRAKE(created_vehicle, true);
        VEHICLE::SET_VEHICLE_ENGINE_ON(created_vehicle, false, true, false);
        
        // Force vehicle to exact coordinates to prevent offset
        ENTITY::SET_ENTITY_COORDS(created_vehicle, g_firePos[0], g_firePos[1], g_firePos[2], true, false, false, true);
        
        // Damage vehicle to make it burn
        VEHICLE::SET_VEHICLE_PETROL_TANK_HEALTH(created_vehicle, -1000.0f);
        VEHICLE::SET_VEHICLE_ENGINE_HEALTH(created_vehicle, -1000.0f);
        VEHICLE::SET_VEHICLE_DAMAGE(created_vehicle, 0.0f, 0.0f, 0.0f, 2000.0f, 5.0f, true);
        
        // Create initial fire effects
        g_fireId = static_cast<int>(FIRE::START_ENTITY_FIRE(created_vehicle));
        
        // Add explosion for visual effect
        FIRE::ADD_EXPLOSION(g_firePos[0], g_firePos[1], g_firePos[2] + 0.5f, 2, 10.0f, true, false, 1.0f);
        
        // Wait for fire to start
        int burn_tries = 0;
        while (!FIRE::IS_ENTITY_ON_FIRE(created_vehicle) && burn_tries < 60) { WAIT(0); burn_tries++; }
        
        LOGI("script", std::string("CREATE_FIRE vehicle fire with maintenance at ") + std::to_string(g_firePos[0]) + "," + std::to_string(g_firePos[1]) + "," + std::to_string(g_firePos[2]));
    } else {
        // Fallback: create script fires
        g_fireId = static_cast<int>(FIRE::START_SCRIPT_FIRE(g_firePos[0], g_firePos[1], g_firePos[2], 25, true));
        LOGW("script", "CREATE_FIRE vehicle model unavailable, using script fires with maintenance");
    }
    
    // Start fire maintenance system
    start_fire_maintenance(g_firePos[0], g_firePos[1], g_firePos[2], created_vehicle);
    
    g_fireReady = true;
}

static void create_fire_near_camera() {
    Any cam = CAM::GET_RENDERING_CAM();
    Vector3 camPos = CAM::GET_CAM_COORD(cam);
    create_fire_near_pos(camPos.x, camPos.y, camPos.z);
}

static void create_fight_near_pos(float ox, float oy, float oz) {
    g_fightReady = false;
    Vector3 nodePos; float nodeHeading = 0.0f;
    bool ok = PATHFIND::GET_CLOSEST_VEHICLE_NODE_WITH_HEADING(ox, oy, oz, &nodePos, &nodeHeading, 1, 3.0, 0);
    float px = ok ? nodePos.x : ox;
    float py = ok ? nodePos.y : oy;
    float pz = ok ? nodePos.z : oz;
    float gz = pz;
    bool hasGround = GAMEPLAY::GET_GROUND_Z_FOR_3D_COORD(px, py, pz, &gz, false);
    if (!hasGround) gz = pz;
    g_fightPos[0] = px;
    g_fightPos[1] = py;
    g_fightPos[2] = gz;

    Hash model = GAMEPLAY::GET_HASH_KEY("g_m_y_lost_01");
    if (!STREAMING::IS_MODEL_VALID(model)) model = GAMEPLAY::GET_HASH_KEY("a_m_m_business_01");
    if (!STREAMING::IS_MODEL_VALID(model)) model = GAMEPLAY::GET_HASH_KEY("a_m_m_skater_01");
    if (STREAMING::IS_MODEL_VALID(model) && !STREAMING::HAS_MODEL_LOADED(model)) {
        STREAMING::REQUEST_MODEL(model);
        int tries = 0;
        while (!STREAMING::HAS_MODEL_LOADED(model) && tries < 200) { WAIT(0); tries++; }
    }
    if (!STREAMING::IS_MODEL_VALID(model) || !STREAMING::HAS_MODEL_LOADED(model)) {
        LOGW("script", "CREATE_FIGHT could not load ped model");
        g_fightReady = true;
        return;
    }

    Hash groupA = 0, groupB = 0;
    char nameA[] = "FIGHT_A";
    char nameB[] = "FIGHT_B";
    PED::ADD_RELATIONSHIP_GROUP(nameA, &groupA);
    PED::ADD_RELATIONSHIP_GROUP(nameB, &groupB);
    PED::SET_RELATIONSHIP_BETWEEN_GROUPS(5, groupA, groupB);
    PED::SET_RELATIONSHIP_BETWEEN_GROUPS(5, groupB, groupA);

    const int n = 6;
    Ped peds[n]{};
    for (int i = 0; i < n; i++) {
        float a = (2.0f * 3.14159f) * (static_cast<float>(i) / static_cast<float>(n));
        float r = 2.0f;
        float x = g_fightPos[0] + cosf(a) * r;
        float y = g_fightPos[1] + sinf(a) * r;
        float z = g_fightPos[2];
        peds[i] = PED::CREATE_PED(26, model, x, y, z, nodeHeading, true, true);
        ENTITY::SET_ENTITY_AS_MISSION_ENTITY(peds[i], true, true);
        PED::SET_PED_RELATIONSHIP_GROUP_HASH(peds[i], (i < n / 2) ? groupA : groupB);
        
        // Make fighting NPCs invincible so they don't die during verification
        ENTITY::SET_ENTITY_INVINCIBLE(peds[i], true);
        PED::SET_PED_CAN_BE_KNOCKED_OFF_VEHICLE(peds[i], false);
        PED::SET_PED_CAN_BE_DRAGGED_OUT(peds[i], false);
    }
    STREAMING::SET_MODEL_AS_NO_LONGER_NEEDED(model);

    for (int i = 0; i < n / 2; i++) {
        AI::TASK_COMBAT_PED(peds[i], peds[n / 2 + (i % (n / 2))], 0, 16);
    }
    for (int i = n / 2; i < n; i++) {
        AI::TASK_COMBAT_PED(peds[i], peds[i - n / 2], 0, 16);
    }

    g_fightReady = true;
    LOGI("script", std::string("CREATE_FIGHT at ") + std::to_string(g_fightPos[0]) + "," + std::to_string(g_fightPos[1]) + "," + std::to_string(g_fightPos[2]));
}

static void create_fight_near_camera() {
    Any cam = CAM::GET_RENDERING_CAM();
    Vector3 camPos = CAM::GET_CAM_COORD(cam);
    create_fight_near_pos(camPos.x, camPos.y, camPos.z);
}

static void create_accident_near_pos(float ox, float oy, float oz) {
    g_accidentReady = false;
    Vector3 nodePos{}; float nodeHeading = 0.0f;
    if (!PATHFIND::GET_CLOSEST_VEHICLE_NODE_WITH_HEADING(ox, oy, oz, &nodePos, &nodeHeading, 1, 3.0, 0)) {
        g_accidentPos[0] = ox; g_accidentPos[1] = oy; g_accidentPos[2] = oz;
        g_accidentReady = true;
        LOGE("script", "CREATE_ACCIDENT could not find closest vehicle node");
        return;
    }

    Hash h1 = GAMEPLAY::GET_HASH_KEY("adder");
    Hash h2 = GAMEPLAY::GET_HASH_KEY("zentorno");
    bool ok1 = STREAMING::IS_MODEL_VALID(h1) && STREAMING::IS_MODEL_A_VEHICLE(h1);
    bool ok2 = STREAMING::IS_MODEL_VALID(h2) && STREAMING::IS_MODEL_A_VEHICLE(h2);
    Vehicle v1 = 0, v2 = 0;
    bool proceed = ok1 && ok2;
    if (proceed && !STREAMING::HAS_MODEL_LOADED(h1)) {
        STREAMING::REQUEST_MODEL(h1);
        int tries = 0;
        while (!STREAMING::HAS_MODEL_LOADED(h1) && tries < 200) { WAIT(0); tries++; }
    }
    if (proceed && !STREAMING::HAS_MODEL_LOADED(h1)) {
        proceed = false;
        LOGW("script", "CREATE_ACCIDENT model h1 not loaded");
    }
    if (proceed && !STREAMING::HAS_MODEL_LOADED(h2)) {
        STREAMING::REQUEST_MODEL(h2);
        int tries2 = 0;
        while (!STREAMING::HAS_MODEL_LOADED(h2) && tries2 < 200) { WAIT(0); tries2++; }
    }
    if (proceed && !STREAMING::HAS_MODEL_LOADED(h2)) {
        proceed = false;
        LOGW("script", "CREATE_ACCIDENT model h2 not loaded");
    }

    if (!proceed) {
        g_accidentPos[0] = nodePos.x; g_accidentPos[1] = nodePos.y; g_accidentPos[2] = nodePos.z;
        g_accidentReady = true;
        LOGW("script", "CREATE_ACCIDENT model invalid or not loaded, reporting node position");
        return;
    }

    float gz = nodePos.z;
    GAMEPLAY::GET_GROUND_Z_FOR_3D_COORD(nodePos.x, nodePos.y, nodePos.z, &gz, false);
    float offset = 80.0f;
    v1 = VEHICLE::CREATE_VEHICLE(h1, nodePos.x, nodePos.y, gz, nodeHeading, true, false);
    STREAMING::SET_MODEL_AS_NO_LONGER_NEEDED(h1);
    VEHICLE::SET_VEHICLE_ON_GROUND_PROPERLY(v1);
    VEHICLE::SET_VEHICLE_HANDBRAKE(v1, true);

    float xr = nodePos.x - sin(nodeHeading * 3.14159f / 180.0f) * offset;
    float yr = nodePos.y + cos(nodeHeading * 3.14159f / 180.0f) * offset;
    v2 = VEHICLE::CREATE_VEHICLE(h2, xr, yr, gz, nodeHeading + 180.0f, true, false);
    STREAMING::SET_MODEL_AS_NO_LONGER_NEEDED(h2);
    VEHICLE::SET_VEHICLE_ON_GROUND_PROPERLY(v2);

    Hash ph = GAMEPLAY::GET_HASH_KEY("a_m_m_skidrow_01");
    bool pok = STREAMING::IS_MODEL_VALID(ph);
    if (pok && !STREAMING::HAS_MODEL_LOADED(ph)) {
        STREAMING::REQUEST_MODEL(ph);
        int ptries = 0;
        while (!STREAMING::HAS_MODEL_LOADED(ph) && ptries < 200) { WAIT(0); ptries++; }
    }
    if (STREAMING::HAS_MODEL_LOADED(ph)) {
        Ped d = PED::CREATE_PED(26, ph, xr, yr, gz, nodeHeading + 180.0f, true, true);
        STREAMING::SET_MODEL_AS_NO_LONGER_NEEDED(ph);
        PED::SET_PED_INTO_VEHICLE(d, v2, -1);
        VEHICLE::SET_VEHICLE_ENGINE_ON(v2, true, true, false);
        AI::TASK_VEHICLE_DRIVE_TO_COORD(d, v2, nodePos.x, nodePos.y, gz, 90.0f, 0, h2, 787004, 1.0f, true);
    } else {
        VEHICLE::SET_VEHICLE_FORWARD_SPEED(v2, 70.0f);
    }

    int frames = 0;
    bool collided = false;
    while (frames < 300) {
        if (ENTITY::IS_ENTITY_TOUCHING_ENTITY(v1, v2) || ENTITY::GET_ENTITY_HEALTH(v1) < 900 || ENTITY::GET_ENTITY_HEALTH(v2) < 900) {
            collided = true;
            break;
        }
        Vector3 p1 = ENTITY::GET_ENTITY_COORDS(v1, true);
        Vector3 p2 = ENTITY::GET_ENTITY_COORDS(v2, true);
        float dist = sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2) + pow(p1.z - p2.z, 2));
        if (dist < 4.0f) {
            collided = true;
            break;
        }
        WAIT(0);
        frames++;
    }

    if (collided) {
        Vector3 p1 = ENTITY::GET_ENTITY_COORDS(v1, true);
        Vector3 p2 = ENTITY::GET_ENTITY_COORDS(v2, true);
        g_accidentPos[0] = (p1.x + p2.x) / 2.0f;
        g_accidentPos[1] = (p1.y + p2.y) / 2.0f;
        g_accidentPos[2] = (p1.z + p2.z) / 2.0f;
        g_accidentReady = true;
        LOGI("script", "CREATE_ACCIDENT detected and reported");
    } else {
        g_accidentPos[0] = nodePos.x; g_accidentPos[1] = nodePos.y; g_accidentPos[2] = nodePos.z;
        g_accidentReady = true;
        LOGW("script", "CREATE_ACCIDENT setup but collision not detected");
    }
}

static void create_accident_near_camera() {
    Any cam = CAM::GET_RENDERING_CAM();
    Vector3 camPos = CAM::GET_CAM_COORD(cam);
    create_accident_near_pos(camPos.x, camPos.y, camPos.z);
}

static void save_verification_sample() {
    if (!g_verificationMode) return;
    
    // Get current camera pose
    Any cam = CAM::GET_RENDERING_CAM();
    Vector3 pos = CAM::GET_CAM_COORD(cam);
    Vector3 rot = CAM::GET_CAM_ROT(cam, 2);
    
    // Create timestamp
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    struct tm tm_info;
    localtime_s(&tm_info, &time_t);
    char timestamp[32];
    strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", &tm_info);
    
    // Determine task description
    std::string task_desc;
    if (g_anomalyType == "fire") {
        task_desc = "find the exploded car";
    } else if (g_anomalyType == "fight") {
        task_desc = "find the street fight";
    }

    g_verificationSteps -= 12; // Subtract 12 steps for camera rotation
    
    // Create verification directory
    ensure_dir("data");
    ensure_dir("data\\verification");
    
    // Create JSON entry (single line for JSONL format)
    std::string json_entry = "{";
    json_entry += "\"scenario_id\": \"verification_" + std::string(timestamp) + "\",";
    json_entry += "\"anomaly_type\": \"" + g_anomalyType + "\",";
    json_entry += "\"anomaly_position\": {\"x\": " + std::to_string(g_anomalyPos.x) + 
                  ", \"y\": " + std::to_string(g_anomalyPos.y) + 
                  ", \"z\": " + std::to_string(g_anomalyPos.z) + "},";
    json_entry += "\"start_pose\": {\"x\": " + std::to_string(pos.x) + 
                  ", \"y\": " + std::to_string(pos.y) + 
                  ", \"z\": " + std::to_string(pos.z) + 
                  ", \"rx\": " + std::to_string(rot.x) + 
                  ", \"ry\": " + std::to_string(rot.y) + 
                  ", \"rz\": " + std::to_string(rot.z) + "},";
    json_entry += "\"expected_steps\": " + std::to_string(g_verificationSteps) + ",";
    json_entry += "\"task_description\": \"" + task_desc + "\",";
    json_entry += "\"created_time\": \"" + std::string(timestamp) + "\"";
    json_entry += "}";
    
    // Append to verification file
    std::ofstream file("data\\verification\\samples.jsonl", std::ios::app);
    if (file.is_open()) {
        file << json_entry << std::endl;  // 添加换行符
        file.close();
        LOGI("script", std::string("Verification sample saved: ") + std::to_string(g_verificationSteps) + " steps");
    } else {
        LOGE("script", "Failed to save verification sample");
    }
    
    // Reset verification mode
    g_verificationMode = false;
    g_verificationSteps = 0;
}

static void start_verification_mode() {
    if (g_verificationMode) return;  // Already in verification mode
    
    // Randomly choose between fire and fight (0 or 1)
    srand(static_cast<unsigned int>(time(nullptr)));
    int choice = rand() % 2;
    
    Vector3 anomalyPos;
    
    if (choice == 0) {
        // Create fire
        create_fire_near_camera();
        g_anomalyType = "fire";
        // Wait for fire to be created and position to be set
        WAIT(500);  // Increased wait time
        if (g_fireReady) {
            anomalyPos.x = g_firePos[0];
            anomalyPos.y = g_firePos[1];
            anomalyPos.z = g_firePos[2];
        } else {
            LOGE("script", "Fire creation failed");
            return;
        }
    } else {
        // Create fight
        create_fight_near_camera();
        g_anomalyType = "fight";
        // Wait for fight to be created and position to be set
        WAIT(500);  // Increased wait time
        if (g_fightReady) {
            anomalyPos.x = g_fightPos[0];
            anomalyPos.y = g_fightPos[1];
            anomalyPos.z = g_fightPos[2];
        } else {
            LOGE("script", "Fight creation failed");
            return;
        }
    }
    
    g_anomalyPos = anomalyPos;
    
    // Get ground height for the anomaly position
    float groundZ = anomalyPos.z;
    bool hasGround = GAMEPLAY::GET_GROUND_Z_FOR_3D_COORD(anomalyPos.x, anomalyPos.y, anomalyPos.z, &groundZ, false);
    if (hasGround) {
        anomalyPos.z = groundZ;  // Update local variable
        g_anomalyPos.z = groundZ;  // Update stored position
    }
    
    // Move camera to 2m above the anomaly (using ground level)
    Any cam = CAM::GET_RENDERING_CAM();
    if (cam) {
        float targetX = anomalyPos.x;
        float targetY = anomalyPos.y;
        float targetZ = anomalyPos.z + 2.0f;
        
        CAM::SET_CAM_COORD(cam, targetX, targetY, targetZ);
        
        LOGI("script", std::string("Verification mode started: ") + g_anomalyType + 
             " at (" + std::to_string(anomalyPos.x) + ", " + 
             std::to_string(anomalyPos.y) + ", " + std::to_string(anomalyPos.z) + 
             ") camera at (" + std::to_string(targetX) + ", " + 
             std::to_string(targetY) + ", " + std::to_string(targetZ) + ")");
    } else {
        LOGE("script", "Failed to get camera handle");
        return;
    }
    
    // Initialize verification mode
    g_verificationMode = true;
    g_verificationSteps = 0;
    
    WAIT(0);  // Allow one frame for camera to update
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

static float yaw_to_target_deg(const Vector3& from, const Vector3& to) {
    float dx = to.x - from.x;
    float dy = to.y - from.y;
    float yaw = atan2f(-dx, dy) * (180.0f / 3.14159f);
    return yaw;
}

enum AutoCollectEvent {
    AUTO_EVENT_ACCIDENT = 1,
    AUTO_EVENT_FIRE = 2,
    AUTO_EVENT_FIGHT = 3
};

static void run_manual_collect(AutoCollectEvent event_type) {
    if (g_recordingEnabled) return;
    if (event_type == AUTO_EVENT_FIRE) start_recording_session("find the exploded car");
    else if (event_type == AUTO_EVENT_FIGHT) start_recording_session("find the street fight");
    else start_recording_session("find crashed cars");

    if (event_type == AUTO_EVENT_FIRE) create_fire_near_camera();
    else if (event_type == AUTO_EVENT_FIGHT) create_fight_near_camera();
    else create_accident_near_camera();
}

static void run_auto_collect(AutoCollectEvent event_type) {
    static bool active = false;
    if (active) return;
    active = true;

    const char* task = nullptr;
    if (event_type == AUTO_EVENT_FIRE) task = "find the exploded car";
    else if (event_type == AUTO_EVENT_ACCIDENT) task = "find crashed cars";
    else if (event_type == AUTO_EVENT_FIGHT) task = "find the street fight";
    if (!g_recordingEnabled) start_recording_session(task);

    Vector3 center{};
    if (event_type == AUTO_EVENT_FIRE) {
        create_fire_near_camera();
        center.x = g_firePos[0]; center.y = g_firePos[1]; center.z = g_firePos[2];
    } else if (event_type == AUTO_EVENT_FIGHT) {
        create_fight_near_camera();
        center.x = g_fightPos[0]; center.y = g_fightPos[1]; center.z = g_fightPos[2];
    } else {
        create_accident_near_camera();
        center.x = g_accidentPos[0]; center.y = g_accidentPos[1]; center.z = g_accidentPos[2];
    }
    Vector3 target{};
    target.x = center.x;
    target.y = center.y;
    target.z = center.z + 5.0f;

    Vector3 centerNode{}; float centerHeading = 0.0f;
    bool okCenterNode = PATHFIND::GET_CLOSEST_VEHICLE_NODE_WITH_HEADING(center.x, center.y, center.z, &centerNode, &centerHeading, 1, 3.0, 0);
    float rad = (okCenterNode ? centerHeading : 0.0f) * (3.14159f / 180.0f);
    float dirx = -sinf(rad);
    float diry = cosf(rad);
    float startDist = 100.0f;
    float candx = center.x - dirx * startDist;
    float candy = center.y - diry * startDist;
    float candz = center.z;

    Vector3 startNode{}; float startHeading = 0.0f;
    bool okStartNode = PATHFIND::GET_CLOSEST_VEHICLE_NODE_WITH_HEADING(candx, candy, candz, &startNode, &startHeading, 1, 3.0, 0);
    float sx = okStartNode ? startNode.x : candx;
    float sy = okStartNode ? startNode.y : candy;
    float sz0 = okStartNode ? startNode.z : candz;
    float gz = sz0;
    bool hasGround = GAMEPLAY::GET_GROUND_Z_FOR_3D_COORD(sx, sy, sz0, &gz, false);
    if (!hasGround) gz = sz0;
    float sz = gz + 10.0f;

    Any cam = CAM::GET_RENDERING_CAM();
    CAM::SET_CAM_COORD(cam, sx, sy, sz);
    Vector3 startPos{}; startPos.x = sx; startPos.y = sy; startPos.z = sz;
    float yaw = quantize_deg(yaw_to_target_deg(startPos, target), YAW_STEPSIZE);
    CAM::SET_CAM_ROT(cam, 0.0f, 0.0f, yaw, 2);
    WAIT(0);

    int maxSteps = 500;
    bool reached = false;
    for (int step = 0; step < maxSteps; step++) {
        cam = CAM::GET_RENDERING_CAM();
        Vector3 pos = CAM::GET_CAM_COORD(cam);
        Vector3 rot = CAM::GET_CAM_ROT(cam, 2);
        float dx = target.x - pos.x;
        float dy = target.y - pos.y;
        float dz = target.z - pos.z;
        float dist = sqrtf(dx * dx + dy * dy + dz * dz);
        if (dist <= 10.0f) { reached = true; break; }

        if (fabsf(dz) > 1.0f) {
            if (dz > 0.0f) {
                record_step("AUTO_UP", 0.0f, 0.0f, STEPSIZE, 0.0f, 0.0f, 0.0f);
                moveCameraDelta(0.0f, 0.0f, STEPSIZE);
            } 
            else {
                record_step("AUTO_DOWN", 0.0f, 0.0f, -STEPSIZE, 0.0f, 0.0f, 0.0f);
                moveCameraDelta(0.0f, 0.0f, -STEPSIZE);
            }
        } 
        else {
            float desiredYaw = quantize_deg(yaw_to_target_deg(pos, target), YAW_STEPSIZE);
            float delta = wrap_angle_deg(desiredYaw - rot.z);
            if (fabsf(delta) >= YAW_STEPSIZE) {
                if (delta > 0.0f) {
                    record_step("AUTO_YAW_LEFT", 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, YAW_STEPSIZE);
                    rotateCameraDelta(0.0f, 0.0f, YAW_STEPSIZE);
                } 
                else {
                    record_step("AUTO_YAW_RIGHT", 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, -YAW_STEPSIZE);
                    rotateCameraDelta(0.0f, 0.0f, -YAW_STEPSIZE);
                }
            } 
            else {
                record_step("AUTO_FORWARD", STEPSIZE, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
                moveCameraDelta(STEPSIZE, 0.0f, 0.0f);
            }
        }
        WAIT(0);
    }

    record_step(reached ? "AUTO_STOP_REACHED" : "AUTO_STOP_MAXSTEPS", 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    stop_recording_session();
    active = false;
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
                if (g_verificationMode) g_verificationSteps++;
            }
            if (shift.isKeyDown())
            {
                record_step("AUTO_UP", 0.0f, 0.0f, STEPSIZE, 0.0f, 0.0f, 0.0f);
                moveCameraDelta(0.0f, 0.0f, STEPSIZE);
                if (g_verificationMode) g_verificationSteps++;
            }
            if (ctrl.isKeyDown())
            {
                record_step("AUTO_DOWN", 0.0f, 0.0f, -STEPSIZE, 0.0f, 0.0f, 0.0f);
                moveCameraDelta(0.0f, 0.0f, -STEPSIZE);
                if (g_verificationMode) g_verificationSteps++;
            }
            if (Q.isKeyDown())
            {
                record_step("AUTO_YAW_LEFT", 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, YAW_STEPSIZE);
                rotateCameraDelta(0.0f, 0.0f, YAW_STEPSIZE);
                if (g_verificationMode) g_verificationSteps++;
            }
            if (E.isKeyDown())
            {
                record_step("AUTO_YAW_RIGHT", 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, -YAW_STEPSIZE);
                rotateCameraDelta(0.0f, 0.0f, -YAW_STEPSIZE);
                if (g_verificationMode) g_verificationSteps++;
            }
            if (F5.isKeyDown())
            {
                run_auto_collect(AUTO_EVENT_FIRE);
            }
            if (F6.isKeyDown())
            {
                run_manual_collect(AUTO_EVENT_FIRE);
            }
            if (F7.isKeyDown())
            {
                record_step("AUTO_STOP_REACHED", 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
                stop_recording_session();
            }
            if (F8.isKeyDown())
            {
                start_verification_mode();
            }
            if (F9.isKeyDown())
            {
                save_verification_sample();
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
            // Stop fire maintenance when exiting camera mode
            stop_fire_maintenance();
            // Reset verification mode when exiting camera mode
            g_verificationMode = false;
            g_verificationSteps = 0;
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
            
            // Maintain fire effects during verification
            maintain_fire();
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
            // Stop fire maintenance when camera stops
            stop_fire_maintenance();
            // setStatusText("Camera mode disabled.");
            LOGI("script", "Camera stopped and returned to player view");
        }
        else if (cmd == "CREATE_FIRE")
        {
            create_fire_near_camera();
        }
        else if (cmd == "CREATE_FIGHT")
        {
            create_fight_near_camera();
        }
        else if (cmd == "CREATE_ACCIDENT")
        {
            create_accident_near_camera();
        }
        else if (cmd == "GET_POSE")
        {
            LOGD("script", "GET_POSE command received, starting processing");
            Vector3 pos{}; Vector3 rot{};
            Any cam = CAM::GET_RENDERING_CAM();
            LOGD("script", std::string("GET_POSE: Got camera handle: ") + std::to_string(cam));
            
            if (cam != 0) {
                pos = CAM::GET_CAM_COORD(cam);
                LOGD("script", std::string("GET_POSE: Got position: ") + std::to_string(pos.x) + "," + std::to_string(pos.y) + "," + std::to_string(pos.z));
                
                rot = CAM::GET_CAM_ROT(cam, 2);
                LOGD("script", std::string("GET_POSE: Got rotation: ") + std::to_string(rot.x) + "," + std::to_string(rot.y) + "," + std::to_string(rot.z));
                
                g_pose[0]=pos.x; g_pose[1]=pos.y; g_pose[2]=pos.z;
                g_pose[3]=rot.x; g_pose[4]=rot.y; g_pose[5]=rot.z;
                g_poseReady = true;
                LOGD("script", std::string("GET_POSE completed successfully: ") + std::to_string(g_pose[0]) + " " + std::to_string(g_pose[1]) + " " + std::to_string(g_pose[2]) + " " + std::to_string(g_pose[3]) + " " + std::to_string(g_pose[4]) + " " + std::to_string(g_pose[5]));
            } else {
                LOGE("script", "GET_POSE: No active camera found (cam handle is 0)");
                // 即使失败也设置g_poseReady，避免server_v2无限等待
                g_poseReady = true;
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
                
                // Make player invincible and invisible for verification
                PLAYER::SET_PLAYER_INVINCIBLE(PLAYER::PLAYER_ID(), true);
                ENTITY::SET_ENTITY_VISIBLE(player, false, false);
                
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
            // Stop fire maintenance when restoring player
            stop_fire_maintenance();
            
            // Restore player to normal state after verification
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
                Vector3 safe_pos{};
                if (cam) {
                    Vector3 camPos = CAM::GET_CAM_COORD(cam);
                    float ground_z = camPos.z;
                    GAMEPLAY::GET_GROUND_Z_FOR_3D_COORD(camPos.x, camPos.y, camPos.z, &ground_z, false);
                    // Move player 20m away from camera position
                    safe_pos.x = camPos.x + 20.0f;
                    safe_pos.y = camPos.y + 20.0f;
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
