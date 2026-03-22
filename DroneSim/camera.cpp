#include "main.h"
#include "camera.h"
#include "script.h"
#include "keyboard.h"
#include "natives.h"
#include "utils.h"
#include <string>
#include <vector>
#include <chrono>
#include "logging.h"

 

int adjustCameraFinished = 0;

static Any cameraHandle;

void startNewCamera()
{
	//Find the location of our camera based on the current actor
	Ped actorPed = PLAYER::PLAYER_PED_ID();
	Vector3 startLocation = ENTITY::GET_ENTITY_COORDS(actorPed, true);
	float startHeading = ENTITY::GET_ENTITY_HEADING(actorPed);
    
	std::chrono::milliseconds ms = std::chrono::duration_cast< std::chrono::milliseconds >(
				std::chrono::system_clock::now().time_since_epoch()
				);
	
	Vector3 camOffset;
	camOffset.x = 0.0;
	camOffset.y = 0.0;
	camOffset.z = 10;

	Vector3 camLocation = ENTITY::GET_OFFSET_FROM_ENTITY_IN_WORLD_COORDS(actorPed, camOffset.x, camOffset.y, camOffset.z);
    LOGI("camera", std::string("Camera location (") + std::to_string(camLocation.x) + ", " + std::to_string(camLocation.y) + ", " + std::to_string(camLocation.z) + ")");
	cameraHandle = CAM::CREATE_CAM_WITH_PARAMS("DEFAULT_SCRIPTED_CAMERA", camLocation.x, camLocation.y, camLocation.z, 0.0, 0.0, 0.0, 40.0, 1, 2);

	CAM::RENDER_SCRIPT_CAMS(true, 1, 1800, 1, 0);
    WAIT(2000);
}

void moveCameraDelta(float dx, float dy, float dz)
{
    Vector3 camNewPos = CAM::GET_CAM_COORD(cameraHandle);
    Vector3 camRot = CAM::GET_CAM_ROT(cameraHandle, 2);
    Vector3 direction = MathUtils::rotationToDirection(camRot);
    if (dx != 0.0f) {
        camNewPos.x += direction.x * dx * cameraSpeedFactor;
        camNewPos.y += direction.y * dx * cameraSpeedFactor;
        camNewPos.z += direction.z * dx * cameraSpeedFactor;
    }
    if (dy != 0.0f) {
        Vector3 b{}; b.z = 1.0f;
        Vector3 sideWays = MathUtils::crossProduct(direction, b);
        camNewPos.x += sideWays.x * dy * cameraSpeedFactor;
        camNewPos.y += sideWays.y * dy * cameraSpeedFactor;
    }
    if (dz != 0.0f) {
        camNewPos.z += dz * cameraSpeedFactor;
    }
    CAM::SET_CAM_COORD(cameraHandle, camNewPos.x, camNewPos.y, camNewPos.z);
	CAM::SET_CAM_ROT(cameraHandle, camRot.x, camRot.y, camRot.z, 2);
}

void rotateCameraDelta(float rx, float ry, float rz)
{
    Vector3 currentRotation = CAM::GET_CAM_ROT(cameraHandle, 2);
    currentRotation.x += rx;
    currentRotation.y += ry;
    currentRotation.z += rz;
    CAM::SET_CAM_ROT(cameraHandle, currentRotation.x, currentRotation.y, currentRotation.z, 2);
}

void setCameraFov(float fov)
{
    Any cam = CAM::GET_RENDERING_CAM();
    if (cam) {
        CAM::SET_CAM_FOV(cam, fov);
    }
}

void StopCamera(int foldNo)
{
    CAM::RENDER_SCRIPT_CAMS(false, 1, 0, 1, 0);
    Any cam = CAM::GET_RENDERING_CAM();
    if (cam) {
        CAM::DESTROY_CAM(cam, 0);
    }
    cameraHandle = 0;
}
