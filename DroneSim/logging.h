#pragma once
#include <string>
#include <mutex>
#include <fstream>

enum LogLevel { LOG_TRACE = 0, LOG_DEBUG = 1, LOG_INFO = 2, LOG_WARN = 3, LOG_ERROR = 4 };

class Logger {
public:
    static void init(const char* dir = "logs", size_t max_bytes = 10485760);
    static void shutdown();
    static void set_level(LogLevel level);
    static void log(LogLevel level, const char* module, const std::string& msg) noexcept;
private:
    static void ensure_open();
    static void rotate_if_needed();
    static std::string now_ts();
    static std::mutex mtx;
    static std::ofstream ofs;
    static std::string path;
    static size_t max_size;
    static LogLevel cur_level;
};

#define LOGT(module, msg) Logger::log(LOG_TRACE, module, msg)
#define LOGD(module, msg) Logger::log(LOG_DEBUG, module, msg)
#define LOGI(module, msg) Logger::log(LOG_INFO, module, msg)
#define LOGW(module, msg) Logger::log(LOG_WARN, module, msg)
#define LOGE(module, msg) Logger::log(LOG_ERROR, module, msg)
