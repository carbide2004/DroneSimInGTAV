#include "command_queue.h"
#include "logging.h"
#include <queue>
#include <mutex>

static std::queue<std::string> g_queue;
static std::mutex g_mtx;

void enqueue_command(const std::string& cmd) {
    std::lock_guard<std::mutex> lk(g_mtx);
    g_queue.push(cmd);
    LOGD("command_queue", std::string("Enqueued command: ") + cmd + ", queue size: " + std::to_string(g_queue.size()));
}

bool try_dequeue_command(std::string& cmd) {
    std::lock_guard<std::mutex> lk(g_mtx);
    if (g_queue.empty()) return false;
    cmd = g_queue.front();
    g_queue.pop();
    LOGD("command_queue", std::string("Dequeued command: ") + cmd + ", remaining queue size: " + std::to_string(g_queue.size()));
    return true;
}

size_t command_queue_size() {
    std::lock_guard<std::mutex> lk(g_mtx);
    return g_queue.size();
}
