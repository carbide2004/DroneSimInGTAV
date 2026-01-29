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

scriptStatusEnum scriptStatus = scriptStop;

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
                moveCameraDelta(1.0f, 0.0f, 0.0f);
            }
            if (S.isKeyDown())
            {
                moveCameraDelta(-1.0f, 0.0f, 0.0f);
            }
            if (A.isKeyDown())
            {
                moveCameraDelta(0.0f, -1.0f, 0.0f);
            }
            if (D.isKeyDown())
            {
                moveCameraDelta(0.0f, 1.0f, 0.0f);
            }
            if (shift.isKeyDown())
            {
                moveCameraDelta(0.0f, 0.0f, 1.0f);
            }
            if (ctrl.isKeyDown())
            {
                moveCameraDelta(0.0f, 0.0f, -1.0f);
            }
            if (Q.isKeyDown())
            {
                rotateCameraDelta(0.0f, 0.0f, 45.0f);
            }
            if (E.isKeyDown())
            {
                rotateCameraDelta(0.0f, 0.0f, -45.0f);
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
            // setStatusText("Camera mode disabled.");
            LOGI("script", "Camera stopped and returned to player view");
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
