#include "keyboard.h"

KeyState F9;
KeyState F10;
KeyState F11;
KeyState MoveForward;
KeyState MoveBackward;
KeyState StrafeLeft;
KeyState StrafeRight;
KeyState YawLeft;
KeyState YawRight;
KeyState MoveUp;
KeyState MoveDown;

bool KeyState::consume_press() {
    if (consumed) {
        return false;
    }
    consumed = TRUE;
    return !is_up && GetTickCount() < time + kMaxDownMilliseconds;
}

void KeyState::push(BOOL is_up_now) {
    time = GetTickCount();
    is_up = is_up_now;
    consumed = FALSE;
}

void OnKeyboardMessage(
    DWORD key,
    WORD,
    BYTE,
    BOOL,
    BOOL,
    BOOL was_down_before,
    BOOL is_up_now) {
    const bool first_press = !was_down_before && !is_up_now;
    if (!first_press) {
        return;
    }
    if (key == VK_F9) {
        F9.push(is_up_now);
    } else if (key == VK_F10) {
        F10.push(is_up_now);
    } else if (key == VK_F11) {
        F11.push(is_up_now);
    } else if (key == 'W') {
        MoveForward.push(is_up_now);
    } else if (key == 'S') {
        MoveBackward.push(is_up_now);
    } else if (key == 'A') {
        StrafeLeft.push(is_up_now);
    } else if (key == 'D') {
        StrafeRight.push(is_up_now);
    } else if (key == 'Q') {
        YawLeft.push(is_up_now);
    } else if (key == 'E') {
        YawRight.push(is_up_now);
    } else if (key == 'Z') {
        MoveUp.push(is_up_now);
    } else if (key == 'C') {
        MoveDown.push(is_up_now);
    }
}
