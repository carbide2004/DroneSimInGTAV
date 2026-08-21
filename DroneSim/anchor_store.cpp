#include "anchor_store.h"

#include <Windows.h>

#include <cmath>
#include <fstream>
#include <iomanip>

namespace {

constexpr wchar_t kAnchorFileName[] = L"DroneSim_anchors.jsonl";

bool executable_directory(
    std::filesystem::path& directory,
    std::string& error) {
    std::wstring buffer(32768, L'\0');
    const DWORD length = GetModuleFileNameW(
        nullptr,
        buffer.data(),
        static_cast<DWORD>(buffer.size()));
    if (length == 0 || length >= buffer.size()) {
        error = "Windows did not return the GTA executable path";
        return false;
    }
    buffer.resize(length);
    directory = std::filesystem::path(buffer).parent_path();
    if (directory.empty()) {
        error = "The GTA executable directory is empty";
        return false;
    }
    return true;
}

}  // namespace

bool append_camera_anchor(
    const RuntimePose& pose,
    std::filesystem::path& output_path,
    std::string& error) {
    if (!std::isfinite(pose.x) ||
        !std::isfinite(pose.y) ||
        !std::isfinite(pose.z)) {
        error = "Camera position contains a non-finite coordinate";
        return false;
    }

    std::filesystem::path game_directory;
    if (!executable_directory(game_directory, error)) {
        return false;
    }
    const std::filesystem::path data_directory = game_directory / L"data";
    output_path = data_directory / kAnchorFileName;

    std::error_code filesystem_error;
    std::filesystem::create_directories(
        data_directory,
        filesystem_error);
    if (filesystem_error) {
        error =
            "Could not create the GTA data directory: " +
            filesystem_error.message();
        return false;
    }

    std::ofstream stream(
        output_path,
        std::ios::out | std::ios::app | std::ios::binary);
    if (!stream.is_open()) {
        error = "Could not open the anchor list for append";
        return false;
    }
    stream << std::fixed << std::setprecision(6)
           << "{\"x\":" << pose.x
           << ",\"y\":" << pose.y
           << ",\"z\":" << pose.z
           << "}\n";
    stream.flush();
    if (!stream.good()) {
        error = "Writing the anchor list failed";
        return false;
    }
    return true;
}
