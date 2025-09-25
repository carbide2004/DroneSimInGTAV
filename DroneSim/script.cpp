#include "script.h"
#include "export.h"
#include "main.h"
#include <string>
#include <fstream>
#include <algorithm>
#include <set>
#include <atlimage.h>
#include <time.h>
#include <chrono>
#include <cmath>

scriptStatusEnum scriptStatus = scriptStop;

void writeLog(std::string msg)
{
	std::chrono::milliseconds ms = std::chrono::duration_cast<std::chrono::milliseconds>(
		std::chrono::system_clock::now().time_since_epoch()
	);
	std::ofstream log("script.log", std::ios_base::app);
	log << ms.count() << " : " << msg << std::endl;
	log.close();
}

void scriptMain()
{
	initDataDir();

	int sleepTime = 0;
	set_status_text("GCC-CL start fine!!!");
	while (true)
	{
		if (scriptStatus == cameraMode) {
			if (CameraMode == false) {
				startNewCamera();
				set_status_text("Now you can move the camera.");
			}
			else {
				adjustCamera();
			}
		}
		else if (scriptStatus == scriptStop && CameraMode == true) {
			StopCamera();
		}
		else {
			if (++sleepTime == 2000) {
				log_to_file("sleep 2000 game ms...");
				log_to_file(std::to_string(scriptStatus));
				sleepTime = 0;
			}
			WAIT(1);
		}
		WAIT(0);
		
	}
}
