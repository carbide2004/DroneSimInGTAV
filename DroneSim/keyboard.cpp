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
    return pending_.exchange(false, std::memory_order_acq_rel);
}

void KeyState::push() {
    pending_.store(true, std::memory_order_release);
}

void OnKeyboardMessage(
    DWORD key,
    WORD,
    BYTE,
    BOOL,
    BOOL,
    BOOL was_down_before,
    BOOL is_up_now) {
    if (was_down_before || is_up_now) {
        return;
    }
    if (key == VK_F9) {
        F9.push();
    } else if (key == VK_F10) {
        F10.push();
    } else if (key == VK_F11) {
        F11.push();
    } else if (key == 'W') {
        MoveForward.push();
    } else if (key == 'S') {
        MoveBackward.push();
    } else if (key == 'A') {
        StrafeLeft.push();
    } else if (key == 'D') {
        StrafeRight.push();
    } else if (key == 'Q') {
        YawLeft.push();
    } else if (key == 'E') {
        YawRight.push();
    } else if (key == 'Z') {
        MoveUp.push();
    } else if (key == 'C') {
        MoveDown.push();
    }
}
