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
#include "command_queue.h"
 
volatile bool g_poseReady = false;
float g_pose[6] = {0};

volatile bool g_accidentReady = false;
float g_accidentPos[3] = {0};

volatile bool g_recordingEnabled = false;
volatile int g_recordingStep = 0;
char g_recordingSessionDir[260] = {0};
char g_recordingRequestedSession[128] = {0};

volatile bool g_fireReady = false;
float g_firePos[3] = {0};
int g_fireId = -1;

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

static void start_recording_session() {
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

    if (v_ok && STREAMING::HAS_MODEL_LOADED(vh)) {
        float heading = ok ? nodeHeading : 0.0f;
        Vehicle v = VEHICLE::CREATE_VEHICLE(vh, g_firePos[0], g_firePos[1], g_firePos[2], heading, true, false);
        STREAMING::SET_MODEL_AS_NO_LONGER_NEEDED(vh);
        ENTITY::SET_ENTITY_AS_MISSION_ENTITY(v, true, true);
        VEHICLE::SET_VEHICLE_ON_GROUND_PROPERLY(v);
        VEHICLE::SET_VEHICLE_HANDBRAKE(v, true);
        VEHICLE::SET_VEHICLE_ENGINE_ON(v, false, true, false);
        Vector3 vpos = ENTITY::GET_ENTITY_COORDS(v, true);
        g_firePos[0] = vpos.x;
        g_firePos[1] = vpos.y;
        g_firePos[2] = vpos.z;
        VEHICLE::SET_VEHICLE_PETROL_TANK_HEALTH(v, -1000.0f);
        VEHICLE::SET_VEHICLE_ENGINE_HEALTH(v, -1000.0f);
        VEHICLE::SET_VEHICLE_DAMAGE(v, 0.0f, 0.0f, 0.0f, 2000.0f, 5.0f, true);
        g_fireId = static_cast<int>(FIRE::START_ENTITY_FIRE(v));
        FIRE::ADD_EXPLOSION(g_firePos[0], g_firePos[1], g_firePos[2] + 0.5f, 2, 10.0f, true, false, 1.0f);
        int burn_tries = 0;
        while (!FIRE::IS_ENTITY_ON_FIRE(v) && burn_tries < 60) { WAIT(0); burn_tries++; }
    } else {
        g_fireId = static_cast<int>(FIRE::START_SCRIPT_FIRE(g_firePos[0], g_firePos[1], g_firePos[2], 25, true));
        LOGW("script", "CREATE_FIRE vehicle model unavailable, using script fire");
    }
    g_fireReady = true;
    LOGI("script", std::string("CREATE_FIRE script fire at ") + std::to_string(g_firePos[0]) + "," + std::to_string(g_firePos[1]) + "," + std::to_string(g_firePos[2]));
}

static void create_fire_near_camera() {
    Any cam = CAM::GET_RENDERING_CAM();
    Vector3 camPos = CAM::GET_CAM_COORD(cam);
    create_fire_near_pos(camPos.x, camPos.y, camPos.z);
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

static void run_auto_collect() {
    static bool active = false;
    if (active) return;
    active = true;

    if (!g_recordingEnabled) start_recording_session();

    create_fire_near_camera();

    Vector3 fire{};
    fire.x = g_firePos[0];
    fire.y = g_firePos[1];
    fire.z = g_firePos[2];
    Vector3 target{};
    target.x = fire.x;
    target.y = fire.y;
    target.z = fire.z + 5.0f;

    Vector3 fireNode{}; float fireHeading = 0.0f;
    bool okFireNode = PATHFIND::GET_CLOSEST_VEHICLE_NODE_WITH_HEADING(fire.x, fire.y, fire.z, &fireNode, &fireHeading, 1, 3.0, 0);
    float rad = (okFireNode ? fireHeading : 0.0f) * (3.14159f / 180.0f);
    float dirx = -sinf(rad);
    float diry = cosf(rad);
    float startDist = 100.0f;
    float candx = fire.x - dirx * startDist;
    float candy = fire.y - diry * startDist;
    float candz = fire.z;

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
    float yaw = quantize_deg(yaw_to_target_deg(startPos, target), 15.0f);
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
                record_step("AUTO_UP", 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f);
                moveCameraDelta(0.0f, 0.0f, 1.0f);
            } 
            else {
                record_step("AUTO_DOWN", 0.0f, 0.0f, -1.0f, 0.0f, 0.0f, 0.0f);
                moveCameraDelta(0.0f, 0.0f, -1.0f);
            }
        } 
        else {
            float desiredYaw = quantize_deg(yaw_to_target_deg(pos, target), 15.0f);
            float delta = wrap_angle_deg(desiredYaw - rot.z);
            if (fabsf(delta) >= 15.0f) {
                if (delta > 0.0f) {
                    record_step("AUTO_YAW_LEFT", 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 15.0f);
                    rotateCameraDelta(0.0f, 0.0f, 15.0f);
                } 
                else {
                    record_step("AUTO_YAW_RIGHT", 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, -15.0f);
                    rotateCameraDelta(0.0f, 0.0f, -15.0f);
                }
            } 
            else {
                record_step("AUTO_FORWARD", 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
                moveCameraDelta(1.0f, 0.0f, 0.0f);
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

	while (true)
	{
        if (scriptStatus == cameraMode)
        {
            if (W.isKeyDown())
            {
                record_step("W", 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
                moveCameraDelta(1.0f, 0.0f, 0.0f);
            }
            if (S.isKeyDown())
            {
                record_step("S", -1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
                moveCameraDelta(-1.0f, 0.0f, 0.0f);
            }
            if (A.isKeyDown())
            {
                record_step("A", 0.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f);
                moveCameraDelta(0.0f, -1.0f, 0.0f);
            }
            if (D.isKeyDown())
            {
                record_step("D", 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f);
                moveCameraDelta(0.0f, 1.0f, 0.0f);
            }
            if (shift.isKeyDown())
            {
                record_step("SHIFT", 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f);
                moveCameraDelta(0.0f, 0.0f, 1.0f);
            }
            if (ctrl.isKeyDown())
            {
                record_step("CTRL", 0.0f, 0.0f, -1.0f, 0.0f, 0.0f, 0.0f);
                moveCameraDelta(0.0f, 0.0f, -1.0f);
            }
            if (Q.isKeyDown())
            {
                record_step("Q", 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 15.0f);
                rotateCameraDelta(0.0f, 0.0f, 15.0f);
            }
            if (E.isKeyDown())
            {
                record_step("E", 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, -15.0f);
                rotateCameraDelta(0.0f, 0.0f, -15.0f);
            }
            if (J.isKeyDown())
            {
                start_recording_session();
            }
            if (K.isKeyDown())
            {
                stop_recording_session();
            }
            if (F5.isKeyDown())
            {
                run_auto_collect();
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
        if (!try_dequeue_command(cmd)) { WAIT(0); continue; }
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
        else if (cmd == "CREATE_ACCIDENT")
        {
            LOGI("script", "Creating accident... (outside check)");
            if (scriptStatus != cameraMode) {
                 LOGW("script", "CREATE_ACCIDENT received but not in camera mode");
            }
            Any cam = CAM::GET_RENDERING_CAM();
            Vector3 camPos = CAM::GET_CAM_COORD(cam);
            Vector3 nodePos; float nodeHeading;
            if (PATHFIND::GET_CLOSEST_VEHICLE_NODE_WITH_HEADING(camPos.x, camPos.y, camPos.z, &nodePos, &nodeHeading, 1, 3.0, 0)) {
                Hash h1 = GAMEPLAY::GET_HASH_KEY("adder");
                Hash h2 = GAMEPLAY::GET_HASH_KEY("zentorno");
                bool ok1 = STREAMING::IS_MODEL_VALID(h1) && STREAMING::IS_MODEL_A_VEHICLE(h1);
                bool ok2 = STREAMING::IS_MODEL_VALID(h2) && STREAMING::IS_MODEL_A_VEHICLE(h2);
                Vehicle v1, v2;
                bool proceed = ok1 && ok2;
                if (proceed && !STREAMING::HAS_MODEL_LOADED(h1)) {
                    STREAMING::REQUEST_MODEL(h1);
                    int tries = 0;
                    while (!STREAMING::HAS_MODEL_LOADED(h1) && tries < 200) { WAIT(0); tries++; }
                }
                if (proceed && !STREAMING::HAS_MODEL_LOADED(h1)) {
                    proceed = false;
                    LOGW("script", "Model h1 not loaded");
                }
                if (proceed && !STREAMING::HAS_MODEL_LOADED(h2)) {
                    STREAMING::REQUEST_MODEL(h2);
                    int tries2 = 0;
                    while (!STREAMING::HAS_MODEL_LOADED(h2) && tries2 < 200) { WAIT(0); tries2++; }
                }
                if (proceed && !STREAMING::HAS_MODEL_LOADED(h2)) {
                    proceed = false;
                    LOGW("script", "Model h2 not loaded");
                }
                
                if (!proceed) {
                    g_accidentPos[0] = nodePos.x; g_accidentPos[1] = nodePos.y; g_accidentPos[2] = nodePos.z;
                    g_accidentReady = true;
                    LOGW("script", "Model invalid or not loaded, reporting node position");
                } else {
                    float gz = nodePos.z;
                    GAMEPLAY::GET_GROUND_Z_FOR_3D_COORD(nodePos.x, nodePos.y, nodePos.z, &gz, false);
                    float offset = 80.0f;
                    v1 = VEHICLE::CREATE_VEHICLE(h1, nodePos.x, nodePos.y, gz, nodeHeading, true, false);
                    STREAMING::SET_MODEL_AS_NO_LONGER_NEEDED(h1);
                    VEHICLE::SET_VEHICLE_ON_GROUND_PROPERLY(v1);
                    VEHICLE::SET_VEHICLE_HANDBRAKE(v1, true);
                    LOGI("script", "Target vehicle created and braked");
                    float xr = nodePos.x - sin(nodeHeading * 3.14159f/180.0f) * offset;
                    float yr = nodePos.y + cos(nodeHeading * 3.14159f/180.0f) * offset;
                    v2 = VEHICLE::CREATE_VEHICLE(h2, xr, yr, gz, nodeHeading + 180.0f, true, false);
                    STREAMING::SET_MODEL_AS_NO_LONGER_NEEDED(h2);
                    VEHICLE::SET_VEHICLE_ON_GROUND_PROPERLY(v2);
                    LOGI("script", "Runner vehicle created");
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
                        LOGI("script", "Runner tasked to drive to target");
                    } else {
                        VEHICLE::SET_VEHICLE_FORWARD_SPEED(v2, 70.0f);
                        LOGI("script", "Ped model unavailable, runner forced forward");
                    }
                }
                
                // Collision detection loop
                int frames = 0;
                bool collided = false;
                while (frames < 300) {
                    if (ENTITY::IS_ENTITY_TOUCHING_ENTITY(v1, v2) || ENTITY::GET_ENTITY_HEALTH(v1) < 900 || ENTITY::GET_ENTITY_HEALTH(v2) < 900) {
                        collided = true;
                        break;
                    }
                    // Also check distance
                    Vector3 p1 = ENTITY::GET_ENTITY_COORDS(v1, true);
                    Vector3 p2 = ENTITY::GET_ENTITY_COORDS(v2, true);
                    float dist = sqrt(pow(p1.x-p2.x,2) + pow(p1.y-p2.y,2) + pow(p1.z-p2.z,2));
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
                    LOGI("script", "Accident detected and reported");
                } else {
                    LOGW("script", "Accident setup but collision not detected");
                    // Fallback: report node pos
                    g_accidentPos[0] = nodePos.x; g_accidentPos[1] = nodePos.y; g_accidentPos[2] = nodePos.z;
                    g_accidentReady = true;
                }
            } else {
                LOGE("script", "Could not find closest vehicle node");
                // Fallback to camera pos
                    g_accidentPos[0] = camPos.x; g_accidentPos[1] = camPos.y; g_accidentPos[2] = camPos.z;
                    g_accidentReady = true;
            }
        }
        else if (scriptStatus == cameraMode) {
            if (cmd == "REQUEST")
            {
                LOGD("script", "start capture");
                makeCmdStart();
            }
            else if (cmd == "GET_POSE")
            {
                Vector3 pos{}; Vector3 rot{};
                Any cam = CAM::GET_RENDERING_CAM();
                pos = CAM::GET_CAM_COORD(cam);
                rot = CAM::GET_CAM_ROT(cam, 2);
                g_pose[0]=pos.x; g_pose[1]=pos.y; g_pose[2]=pos.z;
                g_pose[3]=rot.x; g_pose[4]=rot.y; g_pose[5]=rot.z;
                g_poseReady = true;
                LOGD("script", std::string("GET_POSE: ") + std::to_string(g_pose[0]) + " " + std::to_string(g_pose[1]) + " " + std::to_string(g_pose[2]) + " " + std::to_string(g_pose[3]) + " " + std::to_string(g_pose[4]) + " " + std::to_string(g_pose[5]));
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
        }
		WAIT(0);
	}
}
