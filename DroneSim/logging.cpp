#include "logging.h"
#include <filesystem>
#include <chrono>
#include <iomanip>

std::mutex Logger::mtx;
std::ofstream Logger::ofs;
std::string Logger::path;
size_t Logger::max_size = 10485760;
LogLevel Logger::cur_level = LOG_INFO;

void Logger::init(const char* dir, size_t max_bytes) {
    std::lock_guard<std::mutex> lk(mtx);
    max_size = max_bytes;
    try {
        std::filesystem::create_directories(dir);
        path = std::string(dir) + std::string("\\DroneSim.log");
    } catch (...) {
        path = std::string("DroneSim.log");
    }
    ofs.open(path, std::ios::out | std::ios::app);
}

void Logger::shutdown() {
    std::lock_guard<std::mutex> lk(mtx);
    if (ofs.is_open()) ofs.flush();
    ofs.close();
}

void Logger::set_level(LogLevel level) {
    std::lock_guard<std::mutex> lk(mtx);
    cur_level = level;
}

std::string Logger::now_ts() {
    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() % 1000;
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    tm = *std::localtime(&t);
#endif
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%04d-%02d-%02dT%02d:%02d:%02d.%03lld",
        tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
        tm.tm_hour, tm.tm_min, tm.tm_sec, (long long)ms);
    return std::string(buf);
}

void Logger::ensure_open() {
    if (!ofs.is_open()) ofs.open(path, std::ios::out | std::ios::app);
}

void Logger::rotate_if_needed() {
    try {
        if (std::filesystem::exists(path)) {
            auto sz = std::filesystem::file_size(path);
            if (sz >= max_size) {
                ofs.flush();
                ofs.close();
                for (int i = 2; i >= 0; --i) {
                    std::string src = path + (i == 0 ? std::string("") : std::string(".") + std::to_string(i));
                    std::string dst = path + std::string(".") + std::to_string(i + 1);
                    if (std::filesystem::exists(src)) {
                        std::error_code ec;
                        std::filesystem::rename(src, dst, ec);
                    }
                }
                std::filesystem::rename(path, path + std::string(".1"));
                ofs.open(path, std::ios::out | std::ios::app);
            }
        }
    } catch (...) {}
}

void Logger::log(LogLevel level, const char* module, const std::string& msg) noexcept {
    if (level < cur_level) return;
    std::lock_guard<std::mutex> lk(mtx);
    ensure_open();
    rotate_if_needed();
    const char* lv = level == LOG_TRACE ? "TRACE" : level == LOG_DEBUG ? "DEBUG" : level == LOG_INFO ? "INFO" : level == LOG_WARN ? "WARN" : "ERROR";
    ofs << now_ts() << " " << lv << " [" << module << "] " << msg << std::endl;
}
