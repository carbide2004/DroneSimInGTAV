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

scriptStatusEnum scriptStatus = scriptStop;

void scriptMain()
{

	int sleepTime = 0;
	setStatusText("DroneSim start fine!!!");
    InitializeModServer();
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
            if (cmd == "REQUEST")
            {
                makeCmdStart();
            }
            else if (cmd == "CREATE_CAMERA")
            {
                startNewCamera();
                scriptStatus = cameraMode;
                setStatusText("Camera mode enabled.");
                LOGI("script", "Camera created and mode enabled");
            }
            else if (cmd == "FORWARD" || cmd == "BACKWARD" || cmd == "LEFT" || cmd == "RIGHT" || cmd == "UP" || cmd == "DOWN" || cmd == "LEFTROTATE" || cmd == "RIGHTROTATE")
            {
                if (CameraMode) {
                    LOGI("script", std::string("Processing queued camera movement command: ") + cmd);
                    adjustCamera(cmd);
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
        }
		WAIT(0);
	}
}
