#include "command_queue.h"

#include "logging.h"

#include <queue>
#include <utility>

namespace {

std::queue<RuntimeCommandPtr> g_queue;
std::mutex g_queue_mutex;

}  // namespace

RuntimeCommandPtr make_runtime_command(
    RuntimeCommandType type,
    std::uint64_t request_id) {
    return std::make_shared<RuntimeCommand>(type, request_id);
}

void enqueue_command(const RuntimeCommandPtr& command) {
    if (!command) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_queue_mutex);
    g_queue.push(command);
    LOGD(
        "command_queue",
        "Enqueued request " + std::to_string(command->request_id) +
            ", queue size " + std::to_string(g_queue.size()));
}

bool try_dequeue_command(RuntimeCommandPtr& command) {
    std::lock_guard<std::mutex> lock(g_queue_mutex);
    if (g_queue.empty()) {
        command.reset();
        return false;
    }
    command = g_queue.front();
    g_queue.pop();
    return true;
}

bool wait_for_command(
    const RuntimeCommandPtr& command,
    std::chrono::milliseconds timeout,
    RuntimeCommandResult& result) {
    if (!command) {
        return false;
    }
    std::unique_lock<std::mutex> lock(command->completion_mutex);
    if (!command->completion_cv.wait_for(
            lock,
            timeout,
            [&command]() { return command->completed; })) {
        command->cancelled.store(true, std::memory_order_release);
        return false;
    }
    result = command->result;
    return true;
}

void complete_command(
    const RuntimeCommandPtr& command,
    RuntimeCommandResult result) {
    if (!command) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(command->completion_mutex);
        if (command->completed) {
            return;
        }
        command->result = std::move(result);
        command->completed = true;
    }
    command->completion_cv.notify_all();
}

std::size_t command_queue_size() {
    std::lock_guard<std::mutex> lock(g_queue_mutex);
    return g_queue.size();
}
