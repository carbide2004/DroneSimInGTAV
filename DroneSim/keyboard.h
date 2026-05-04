#pragma once
#include "types.h"
#include <WinUser.h>
#include <map>
struct keyInfo {
	const static int MAX_DOWN;
	DWORD time;
	BOOL isWithAlt;
	BOOL wasDownBefore;
	BOOL isUpNow;
	BOOL isConsumed;

	keyInfo();
	bool isKeyDown();
	void pushDown(BOOL _isUpNow, BOOL _isWithAlt, BOOL _wasDownBefore);
};
extern keyInfo W, A, S, D, Q, E, V, shift, ctrl, tab, oemPlus, oemMinus, F12, F5, F6, F7, F10, I, F11, numKey[10];

void OnKeyboardMessage(DWORD key, WORD repeats, BYTE scanCode, BOOL isExtended, BOOL isWithAlt, BOOL wasDownBefore, BOOL isUpNow);
