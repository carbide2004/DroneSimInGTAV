#include "script.h"
#include "export.h"
#include "main.h"
#include "utils.h"
#include "camera.h"
#include "server.h"
#include "server_v2.h"
#include "logging.h"
#include <string>
#include <fstream>
#include <algorithm>
#include <set>
#include <atlimage.h>
#include <time.h>
#include <chrono>
#include <cmath>

 

extern std::queue<std::string> g_cmdQueue;
static volatile bool g_poseReady = false;
static float g_pose[6] = {0};

scriptStatusEnum scriptStatus = scriptStop;

void scriptMain()
{

	int sleepTime = 0;
	setStatusText("DroneSim start fine!!!");
    
    InitializeServerV2();
    LOGI("script", "DroneSim script started");

    setStatusText("Awaiting client commands.");
    LOGI("script", "Awaiting client commands");

	while (true)
	{
        if (g_cmdQueue.size() > 0)
        {
            std::string cmd = g_cmdQueue.front();
            g_cmdQueue.pop();
            if (cmd == "CREATE_CAMERA")
            {
                startNewCamera();
                scriptStatus = cameraMode;
                setStatusText("Camera mode enabled.");
                LOGI("script", "Camera created and mode enabled");
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
                    if (CameraMode) {
                        Any cam = CAM::GET_RENDERING_CAM();
                        pos = CAM::GET_CAM_COORD(cam);
                        rot = CAM::GET_CAM_ROT(cam, 2);
                    } else {
                        Ped ped = PLAYER::PLAYER_PED_ID();
                        pos = ENTITY::GET_ENTITY_COORDS(ped, true);
                        float heading = ENTITY::GET_ENTITY_HEADING(ped);
                        rot = {0.0f, 0.0f, heading};
                    }
                    g_pose[0]=pos.x; g_pose[1]=pos.y; g_pose[2]=pos.z;
                    g_pose[3]=rot.x; g_pose[4]=rot.y; g_pose[5]=rot.z;
                    g_poseReady = true;
                }
            else if (cmd.rfind("MOVE ", 0) == 0)
            {
                if (CameraMode) {
                    auto s = cmd.substr(5);
                    std::stringstream ss(s);
                    float dx=0,dy=0,dz=0; ss >> dx >> dy >> dz;
                    moveCameraDelta(dx,dy,dz);
                    LOGD("script", std::string("MOVE ") + std::to_string(dx) + "," + std::to_string(dy) + "," + std::to_string(dz));
                }
            }
            else if (cmd.rfind("ROTATE ", 0) == 0)
            {
                if (CameraMode) {
                    auto s = cmd.substr(7);
                    std::stringstream ss(s);
                    float rx=0,ry=0,rz=0; ss >> rx >> ry >> rz;
                    rotateCameraDelta(rx,ry,rz);
                    LOGD("script", std::string("ROTATE ") + std::to_string(rx) + "," + std::to_string(ry) + "," + std::to_string(rz));
                }
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
                    GAMEPLAY::SET_WEATHER_TYPE_NOW_PERSIST((char*)name.c_str());
                    LOGI("script", std::string("Set weather to ")+name);
                }
            }
        }
		WAIT(0);
	}
}
