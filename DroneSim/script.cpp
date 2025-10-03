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

static const char* logFilePathScript = "logs\\script.log";

scriptStatusEnum scriptStatus = scriptStop;

void setStatusText(std::string text)
{
	UI::_SET_NOTIFICATION_TEXT_ENTRY("STRING");
	UI::_ADD_TEXT_COMPONENT_STRING((LPSTR)text.c_str());
	UI::_DRAW_NOTIFICATION(1, 1);
}

void scriptMain()
{

	int sleepTime = 0;
	setStatusText("DroneSim start fine!!!");
	
	while (true)
	{
		auto f = fopen(logFilePathScript, "a");
		std::chrono::milliseconds ms = std::chrono::duration_cast< std::chrono::milliseconds >(
				std::chrono::system_clock::now().time_since_epoch()
				);
		fprintf(f, "[%I64d] : Catch RGBD\n", ms.count());
		fclose(f);
		makeCmdStart();
		WAIT(10000);
		// if (scriptStatus == cameraMode) {
		// 	if (CameraMode == false) {
		// 		startNewCamera();
		// 		setStatusText("Now you can move the camera.");
		// 	}
		// 	else {
		// 		adjustCamera();
		// 	}
		// }
		// else if (scriptStatus == scriptStop && CameraMode == true) {
		// 	StopCamera();
		// }
		// else {
		// 	if (++sleepTime == 2000) {
		// 		log_to_file("sleep 2000 game ms...");
		// 		log_to_file(std::to_string(scriptStatus));
		// 		sleepTime = 0;
		// 	}
		// 	WAIT(1);
		// }
		// WAIT(0);
		
	}
}
