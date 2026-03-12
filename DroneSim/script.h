#pragma once
#include "natives.h"
#include "types.h"
#include "enums.h"
#include "main.h"

enum scriptStatusEnum {
	scriptStart,
	scriptStop,
	scriptReady,
	scriptReadyCamera,
	cameraMode,
	cameraModeEnd,
	scriptReadyDefineArea,
	defineArea,
	defineAreaEnd,
	scriptReadySetLevel,
	setLevel,
	setLevelEnd,
	scriptEndReady
};
extern scriptStatusEnum scriptStatus;
extern volatile bool g_accidentReady;
extern float g_accidentPos[3];

extern volatile bool g_recordingEnabled;
extern volatile int g_recordingStep;
extern char g_recordingSessionDir[260];
extern char g_recordingRequestedSession[128];
extern char g_recordingRequestedTask[256];

extern volatile bool g_fireReady;
extern float g_firePos[3];
extern int g_fireId;

extern volatile bool g_fightReady;
extern float g_fightPos[3];
void scriptMain();
