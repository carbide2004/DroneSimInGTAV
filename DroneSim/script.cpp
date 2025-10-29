#include "script.h"
#include "export.h"
#include "main.h"
#include "utils.h"
#include "camera.h"
#include "server.h"
#include <string>
#include <fstream>
#include <algorithm>
#include <set>
#include <atlimage.h>
#include <time.h>
#include <chrono>
#include <cmath>

char* logFilePathScript = "logs\\script.log";

scriptStatusEnum scriptStatus = scriptStop;

void scriptMain()
{

	int sleepTime = 0;
	setStatusText("DroneSim start fine!!!");
	InitializeModServer();

	setStatusText("Start camera mode in 5 seconds.");
	WAIT(5000);
	
	scriptStatus = cameraMode;

	while (true)
	{
		if (scriptStatus == cameraMode) {
			if (CameraMode == false) {
				startNewCamera();
				setStatusText("Now you can move the camera.");
			}
			else {
				if (g_cmdQueue.size() > 0)
				{
					std::string cmd = g_cmdQueue.front();
					g_cmdQueue.pop();
					
					// 检查是否为 REQUEST 命令
					if (cmd == "REQUEST")
					{
						// 在游戏脚本线程中调用 makeCmdStart() 触发 D3D 渲染线程的捕获
						log_to_pedTxt("Processing queued command: REQUEST. Triggering D3D capture.", logFilePathScript);
						makeCmdStart(); 
					}
					else if (cmd == "FORWARD" || cmd == "BACKWARD" || cmd == "LEFT" || cmd == "RIGHT" || cmd == "UP" || cmd == "DOWN" || cmd == "LEFTROTATE" || cmd == "RIGHTROTATE")
					{
						log_to_pedTxt("Processing queued camera movement command: " + cmd, logFilePathScript);
						adjustCamera(cmd);
					}
				}
			}
		}
		WAIT(0);
	}
}
