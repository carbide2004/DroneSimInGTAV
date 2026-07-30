#pragma once

#include <cstdint>

enum MsgType : std::uint8_t {
    MSG_CREATE_CAMERA = 1,
    MSG_SET_FOV = 4,
    MSG_CAPTURE = 5,
    MSG_PING = 6,
    MSG_GET_POSE = 7,
    MSG_SET_TIME = 8,
    MSG_SET_WEATHER = 9,
    MSG_STOP_CAMERA = 10,
    MSG_TELEPORT_PLAYER = 17,
    MSG_RESTORE_PLAYER = 18,
    MSG_GET_CAMERA_STATE = 20,
    MSG_SET_CAMERA_POSE = 21,
    MSG_PREPARE_FIRE_SCENARIO = 22,
    MSG_GET_SCENARIO_STATE = 23,
    MSG_START_SCENARIO = 24,
    MSG_RESET_SCENARIO = 25,
    MSG_SET_CAMERA_PITCH = 26,
    MSG_ENTER_LOCKSTEP = 27,
    MSG_GET_LOCKSTEP_STATE = 28,
    MSG_ADVANCE_LOCKSTEP = 29,
    MSG_EXIT_LOCKSTEP = 30,
    MSG_QUERY_VISIBILITY = 31,
    MSG_PROBE_CAMERA_START = 32,
    MSG_PROBE_CAMERA_GEOMETRY_BATCH = 33,
    MSG_QUERY_TARGET_VISIBILITY_BATCH = 34,
};

struct MsgHeader {
    char magic[4];
    std::uint8_t version;
    std::uint8_t type;
    std::uint8_t flags;
    std::uint8_t reserved;
    std::uint64_t request_id;
    std::uint32_t length;
};
