#pragma once
#include <string>

void enqueue_command(const std::string& cmd);
bool try_dequeue_command(std::string& cmd);
size_t command_queue_size();
