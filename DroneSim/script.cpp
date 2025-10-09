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

static const char* logFilePathScript = "logs\\script.log";

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
				adjustCamera();
			}
		}
		WAIT(0);
	}
}
