#pragma once

#include "types.h"

#include <WinUser.h>

#include <atomic>

struct KeyState {
    bool consume_press();
    void push();

private:
    std::atomic<bool> pending_{false};
};

extern KeyState F9;
extern KeyState F10;
extern KeyState F11;
extern KeyState MoveForward;
extern KeyState MoveBackward;
extern KeyState StrafeLeft;
extern KeyState StrafeRight;
extern KeyState YawLeft;
extern KeyState YawRight;
extern KeyState MoveUp;
extern KeyState MoveDown;

void OnKeyboardMessage(
    DWORD key,
    WORD repeats,
    BYTE scan_code,
    BOOL is_extended,
    BOOL is_with_alt,
    BOOL was_down_before,
    BOOL is_up_now);
