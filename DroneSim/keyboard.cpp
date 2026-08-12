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

bool KeyState::is_down() const {
    return down_.load(std::memory_order_acquire);
}

void KeyState::set_down(bool down) {
    down_.store(down, std::memory_order_release);
}

void KeyState::reset() {
    pending_.store(false, std::memory_order_release);
    down_.store(false, std::memory_order_release);
}

void OnKeyboardMessage(
    DWORD key,
    WORD,
    BYTE,
    BOOL,
    BOOL,
    BOOL was_down_before,
    BOOL is_up_now) {
    if (key == 'W') {
        MoveForward.set_down(!is_up_now);
    } else if (key == 'S') {
        MoveBackward.set_down(!is_up_now);
    } else if (key == 'A') {
        StrafeLeft.set_down(!is_up_now);
    } else if (key == 'D') {
        StrafeRight.set_down(!is_up_now);
    } else if (key == 'Q') {
        YawLeft.set_down(!is_up_now);
    } else if (key == 'E') {
        YawRight.set_down(!is_up_now);
    } else if (key == 'Z') {
        MoveUp.set_down(!is_up_now);
    } else if (key == 'C') {
        MoveDown.set_down(!is_up_now);
    } else if (was_down_before || is_up_now) {
        return;
    } else if (key == VK_F9) {
        F9.push();
    } else if (key == VK_F10) {
        F10.push();
    } else if (key == VK_F11) {
        F11.push();
    }
}
