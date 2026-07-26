#pragma once
#include "natives.h"
#include "types.h"
#include "enums.h"
#include "main.h"
#include <atomic>

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
extern std::atomic<bool> g_cameraStateReady;
extern std::atomic<bool> g_cameraActive;
extern std::atomic<bool> g_accidentReady;
extern float g_accidentPos[3];

extern std::atomic<bool> g_recordingEnabled;
extern std::atomic<int> g_recordingStep;
extern char g_recordingSessionDir[260];
extern char g_recordingRequestedSession[128];
extern char g_recordingRequestedTask[256];

extern std::atomic<bool> g_fireReady;
extern float g_firePos[3];
extern int g_fireId;

extern std::atomic<bool> g_arrestReady;
extern float g_arrestPos[3];
void scriptMain();
