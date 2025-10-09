#pragma once
#define BOOST_ASIO_NO_WIN32_LEAN_AND_MEAN
#define _WIN32_WINNT 0x0601 // 确保在 Windows.h 之前定义
#define WINVER 0x0601
#include <boost/asio.hpp>
#include <iostream>
#include <thread>
#include <vector>
#include <fstream>
#include "utils.h"

extern char* SERVER_LOG_FILE;


// ====================================================================
// ModServer 类声明
// ====================================================================
class ModServer
{
public:
    // 构造函数
    ModServer(boost::asio::io_context& io_context, unsigned short port);

private:
    // 异步接受新连接的逻辑
    void start_accept();

    // 处理单个客户端连接的逻辑
    void handle_client_connection();

    // 异步发送数据辅助函数
    void send_data_async(std::vector<unsigned char> data);

    // 获取文件字节数据的函数
    std::vector<unsigned char> GetBytes(std::string filePath);

    // 成员变量
    boost::asio::ip::tcp::acceptor acceptor_;
    boost::asio::ip::tcp::socket socket_; // 用于接受新连接，其所有权会转移
};


// ====================================================================
// 全局服务器控制函数声明
// 这些函数将用于在你的 Mod 生命周期中启动和停止服务器
// ====================================================================
void InitializeModServer();
void ShutdownModServer();