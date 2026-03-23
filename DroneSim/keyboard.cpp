#include "server_v2.h"
#include "keyboard.h"
#include "types.h"
#include "script.h"
#include "main.h"
#include "camera.h"
#include "utils.h"
#include "logging.h"

const int keyInfo::MAX_DOWN = 500; //ms

keyInfo::keyInfo() {
	time = 0;
	isConsumed = TRUE;
}
bool keyInfo::isKeyDown() {
	if (isConsumed) return false;
	isConsumed = TRUE;
	return ((GetTickCount() < time + MAX_DOWN) && !isUpNow);
}
void keyInfo::pushDown(BOOL _isUpNow, BOOL _isWithAlt, BOOL _wasDownBefore) {
	time = GetTickCount();
	isConsumed = FALSE;
	isWithAlt = _isWithAlt;
	wasDownBefore = _wasDownBefore;
	isUpNow = _isUpNow;
}
keyInfo W, A, S, D, Q, E, V, shift, ctrl, tab, oemPlus, oemMinus, F12, F5, F6, F7, F8, F9, F10, I, F11, numKey[10];

void OnKeyboardMessage(DWORD key, WORD repeats, BYTE scanCode, BOOL isExtended, BOOL isWithAlt, BOOL wasDownBefore, BOOL isUpNow)
{
	bool updown = !wasDownBefore && !isUpNow;
	auto push = [&key, &updown](char mykey) {
		return key == mykey && updown;
	};
	
	if (scriptStatus == scriptStop && push(VK_F10)) {
		F10.pushDown(isUpNow, isWithAlt, wasDownBefore);
	}
	if (scriptStatus == cameraMode) {
		if (key == 'W') W.pushDown(isUpNow, isWithAlt, wasDownBefore);
		if (key == 'A') A.pushDown(isUpNow, isWithAlt, wasDownBefore);
		if (key == 'S') S.pushDown(isUpNow, isWithAlt, wasDownBefore);
		if (key == 'D') D.pushDown(isUpNow, isWithAlt, wasDownBefore);
		if (key == 'V') V.pushDown(isUpNow, isWithAlt, wasDownBefore);
		if (key == 'Q') Q.pushDown(isUpNow, isWithAlt, wasDownBefore);
		if (key == 'E') E.pushDown(isUpNow, isWithAlt, wasDownBefore);

		if (key == VK_SHIFT) shift.pushDown(isUpNow, isWithAlt, wasDownBefore);
		if (key == VK_CONTROL) ctrl.pushDown(isUpNow, isWithAlt, wasDownBefore);
		if (push(VK_F12)) F12.pushDown(isUpNow, isWithAlt, wasDownBefore);
		if (push(VK_F5)) F5.pushDown(isUpNow, isWithAlt, wasDownBefore);
		if (push(VK_F6)) F6.pushDown(isUpNow, isWithAlt, wasDownBefore);
		if (push(VK_F7)) F7.pushDown(isUpNow, isWithAlt, wasDownBefore);
		if (push(VK_F8)) F8.pushDown(isUpNow, isWithAlt, wasDownBefore);
		if (push(VK_F9)) F9.pushDown(isUpNow, isWithAlt, wasDownBefore);
		if (push(VK_F11)) F11.pushDown(isUpNow, isWithAlt, wasDownBefore);
		if (key == VK_TAB) tab.pushDown(isUpNow, isWithAlt, wasDownBefore);
	}
}
