#pragma once

#include "command_queue.h"

#include <filesystem>
#include <string>

// Append one camera position to the GTA installation's data directory as a
// JSONL record. The executable location, rather than the process working
// directory, is used so the output is stable regardless of how GTA was
// launched.
bool append_camera_anchor(
    const RuntimePose& pose,
    std::filesystem::path& output_path,
    std::string& error);
