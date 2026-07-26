#pragma once
#include <cstdint>

enum MsgType : uint8_t {
    MSG_CREATE_CAMERA = 1,
    MSG_MOVE = 2,
    MSG_ROTATE = 3,
    MSG_SET_FOV = 4,
    MSG_CAPTURE = 5,
    MSG_PING = 6,
    MSG_GET_POSE = 7,
    MSG_SET_TIME = 8,
    MSG_SET_WEATHER = 9,
    MSG_STOP_CAMERA = 10,
    MSG_CREATE_ACCIDENT = 11,
    MSG_GET_RECORDING_INFO = 12,
    MSG_SET_RECORDING_SESSION = 13,
    MSG_CREATE_FIRE = 14,
    MSG_CREATE_ARREST = 15,
    MSG_SET_POSTURE = 16,
    MSG_TELEPORT_PLAYER = 17,
    MSG_RESTORE_PLAYER = 18,
    MSG_CLEAR_SCENE = 19,
    MSG_GET_CAMERA_STATE = 20
};

struct MsgHeader {
    char magic[4];
    uint8_t version;
    uint8_t type;
    uint8_t flags;
    uint8_t reserved;
    uint64_t request_id;
    uint32_t length;
};
