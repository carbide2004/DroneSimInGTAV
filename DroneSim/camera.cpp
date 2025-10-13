#include "main.h"
#include "camera.h"
#include "script.h"
#include "keyboard.h"
#include "natives.h"
#include "utils.h"
#include <string>
#include <vector>
#include <chrono>

char* logFilePathCamera = "logs\\camera.log";

int adjustCameraFinished = 0;
bool CameraMode = false;

static Any cameraHandle;

void startNewCamera()
{
	//Find the location of our camera based on the current actor
	Ped actorPed = PLAYER::PLAYER_PED_ID();
	Vector3 startLocation = ENTITY::GET_ENTITY_COORDS(actorPed, true);
	float startHeading = ENTITY::GET_ENTITY_HEADING(actorPed);
	auto f = fopen(logFilePathCamera, "a");
	std::chrono::milliseconds ms = std::chrono::duration_cast< std::chrono::milliseconds >(
				std::chrono::system_clock::now().time_since_epoch()
				);
	
	Vector3 camOffset;
	camOffset.x = 0.0;
	camOffset.y = 0.0;
	camOffset.z = 10;

	Vector3 camLocation = ENTITY::GET_OFFSET_FROM_ENTITY_IN_WORLD_COORDS(actorPed, camOffset.x, camOffset.y, camOffset.z);
	fprintf(f, "[%I64d] : Camera location (%f, %f, %f)\n", ms.count(), camLocation.x, camLocation.y, camLocation.z);
	cameraHandle = CAM::CREATE_CAM_WITH_PARAMS("DEFAULT_SCRIPTED_CAMERA", camLocation.x, camLocation.y, camLocation.z, 0.0, 0.0, 0.0, 40.0, 1, 2);

	CAM::RENDER_SCRIPT_CAMS(true, 1, 1800, 1, 0);
	WAIT(2000);

	CameraMode = true;
}

void adjustCamera()
{

	// movement
	Vector3 camDelta = {};
	float nfov = 0.0;
	bool isMovement = false;
	if (g_cmdQueue.size() > 0) {
		std::string cmd = g_cmdQueue.front();
		g_cmdQueue.pop();
		log_to_pedTxt("front cmd is: " + cmd, logFilePathCamera);
		if (cmd == "FORWARD") {
			camDelta.x = STEPSIZE;
			isMovement = true;
			setStatusText("Camera moving forward.");
		}
		else if (cmd == "BACKWARD") {
			camDelta.x = -STEPSIZE;
			isMovement = true;
			setStatusText("Camera moving backward.");
		}
		else if (cmd == "LEFT") {
			camDelta.y = -STEPSIZE;
			isMovement = true;
			setStatusText("Camera moving left.");
		}
		else if (cmd == "RIGHT") {
			camDelta.y = STEPSIZE;	
			isMovement = true;
			setStatusText("Camera moving right.");
		}
		else if (cmd == "UP") {
			camDelta.z = STEPSIZE;
			isMovement = true;
			setStatusText("Camera moving up.");
		}
		else if (cmd == "DOWN") {
			camDelta.z = -STEPSIZE;
			isMovement = true;
			setStatusText("Camera moving down.");
		}
		else if (cmd == "LEFTROTATE") {
			Vector3 currentRotation = CAM::GET_CAM_ROT(cameraHandle, 2);
			currentRotation.z += 45.0f;
			CAM::SET_CAM_ROT(cameraHandle, currentRotation.x, currentRotation.y, currentRotation.z, 2);
			setStatusText("Camera rotating left.");
		}
		else if (cmd == "RIGHTROTATE") {
			Vector3 currentRotation = CAM::GET_CAM_ROT(cameraHandle, 2);
			currentRotation.z += -45.0f;
			CAM::SET_CAM_ROT(cameraHandle, currentRotation.x, currentRotation.y, currentRotation.z, 2);
			setStatusText("Camera rotating right.");
		}
	}
	if (isMovement) {
		Vector3 camNewPos = CAM::GET_CAM_COORD(cameraHandle);
		float fov = CAM::GET_CAM_FOV(cameraHandle);
		/*camLastPos.x = camNewPos.x;
		camLastPos.y = camNewPos.y;
		camLastPos.z = camNewPos.z;*/

		Vector3 camRot = {};
		camRot = CAM::GET_CAM_ROT(cameraHandle, 2);
		//camera rotation is not as expected. .x value is rotation in the z-plane (view up/down) and third paramter is the rotation in the x,y plane.

		Vector3 direction = {};
		direction = MathUtils::rotationToDirection(camRot);

		//forward motion
		if (camDelta.x != 0.0) {
			camNewPos.x += direction.x * camDelta.x * cameraSpeedFactor;
			camNewPos.y += direction.y * camDelta.x * cameraSpeedFactor;
			camNewPos.z += direction.z * camDelta.x * cameraSpeedFactor;
		}

		//sideways motion
		if (camDelta.y != 0.0) {
			//straight up
			Vector3 b = {};
			b.z = 1.0;

			Vector3 sideWays = {};
			sideWays = MathUtils::crossProduct(direction, b);

			camNewPos.x += sideWays.x * camDelta.y * cameraSpeedFactor;
			camNewPos.y += sideWays.y * camDelta.y * cameraSpeedFactor;
		}

		//up/down
		if (camDelta.z != 0.0) {
			camNewPos.z += camDelta.z * cameraSpeedFactor;
		}

		if (nfov != 0.0) {
			fov += nfov;
		}

		CAM::SET_CAM_COORD(cameraHandle, camNewPos.x, camNewPos.y, camNewPos.z);
		CAM::SET_CAM_FOV(cameraHandle, fov);
	}
	
	// rotation
	float rightAxisX = CONTROLS::GET_DISABLED_CONTROL_NORMAL(0, 220);
	float rightAxisY = CONTROLS::GET_DISABLED_CONTROL_NORMAL(0, 221);

	if (rightAxisX != 0.0 || rightAxisY != 0.0) {
		//Rotate camera - Multiply by sensitivity settings
		Vector3 currentRotation = CAM::GET_CAM_ROT(cameraHandle, 2);
		currentRotation.x += rightAxisY * -5.0f;
		currentRotation.z += rightAxisX * -10.0f;
		CAM::SET_CAM_ROT(cameraHandle, currentRotation.x, currentRotation.y, currentRotation.z, 2);
	}
}
