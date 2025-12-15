#include "server.h"
#include "logging.h"

namespace ba = boost::asio;
namespace bap = boost::asio::ip;

// 用于管理 ModServer 实例的全局指针
static std::unique_ptr<class ModServer> g_modServerInstance;
// 用于管理 io_context 的全局实例
static ba::io_context g_ioContext;
// 用于管理服务器线程的全局指针
static std::unique_ptr<std::thread> g_serverThread;

extern catchState cmdToCatch;
extern std::string g_rgbCapturedFilePath;
extern std::string g_depthCapturedFilePath;
extern std::string g_stencilCapturedFilePath;
extern std::string g_matrixCapturedFilePath;
extern std::queue<std::string> g_cmdQueue;

 

// ====================================================================
// ModServer 类成员函数的实现
// 记住使用 ModServer:: 前缀
// ====================================================================
ModServer::ModServer(boost::asio::io_context& io_context, unsigned short port)
    : acceptor_(io_context, boost::asio::ip::tcp::endpoint(boost::asio::ip::tcp::v4(), port)),
      socket_(io_context)
{
    LOGI("server", std::string("Mod Server listening on port ") + std::to_string(port));
    start_accept();
}

void ModServer::start_accept()
{
    LOGI("server", "Waiting for new client connection...");
    
    acceptor_.async_accept(socket_,
        [this](const boost::system::error_code& error)
        {
            if (!error)
            {
                try {
                    std::string client_info = socket_.remote_endpoint().address().to_string() + ":" + 
                                            std::to_string(socket_.remote_endpoint().port());
                    LOGI("server", std::string("Client connected from: ") + client_info);
                    
                    // 立即开始处理客户端连接
                    handle_client_connection();
                } catch (const std::exception& e) {
                    LOGE("server", std::string("Error getting client endpoint: ") + e.what());
                    handle_client_connection(); // 仍然尝试处理连接
                }
            }
            else
            {
                LOGE("server", std::string("Error accepting connection: ") + error.message());
                // 继续监听新的连接
                start_accept();
            }
        });
}

std::vector<unsigned char> ModServer::GetBytes(std::string filePath)
{
    std::vector<unsigned char> image_data;

    std::ifstream file(filePath, std::ios::binary | std::ios::ate);

    if (file.is_open()) {
        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);

        image_data.resize(size);
        if (file.read(reinterpret_cast<char*>(image_data.data()), size)) {
            // 读取成功
        } else {
            // 读取失败，清空vector
            image_data.clear();
        }
        file.close();
    }
    
    return image_data;
}

void ModServer::handle_client_connection()
{
    // 检查socket状态
    if (!socket_.is_open())
    {
        LOGW("server", "Socket is not open when trying to read");
        start_accept(); // 重新开始接受连接
        return;
    }
    
    LOGD("server", "Starting async_read_some operation");
    
    // 使用shared_ptr管理buffer生命周期
    auto buffer = std::make_shared<std::vector<char>>(128);

    socket_.async_read_some(boost::asio::buffer(*buffer),
        [this, buffer](const boost::system::error_code& error, size_t bytes_transferred)
        {
            LOGD("server", "async_read_some callback triggered");
            
            if (!error)
            {
                LOGD("server", std::string("Async_read_some completed. Bytes transferred: ") + std::to_string(bytes_transferred));

                if (bytes_transferred == 0) {
                    LOGW("server", "Client sent 0 bytes - client disconnected before sending data");
                    
                    if (socket_.is_open()) {
                        socket_.close();
                    }
                    
                    // 重新开始接受新连接
                    start_accept();
                    return;
                }

                // 在转换为字符串之前，打印原始字节内容 (ASCII表示)
                std::string raw_command_bytes(buffer->begin(), buffer->begin() + bytes_transferred);
                std::string debug_output = "Raw bytes received (ASCII): ";
                for (char c : raw_command_bytes) {
                    if (c >= 32 && c <= 126) { // 可打印ASCII字符
                        debug_output += c;
                    } else {
                        debug_output += "\\x" + std::to_string(static_cast<unsigned int>(static_cast<unsigned char>(c))); // 非可打印字符显示十六进制
                    }
                }
                LOGD("server", debug_output);

                std::string command(buffer->begin(), buffer->begin() + bytes_transferred);
                
                // 去除字符串末尾的空白字符
                command.erase(command.find_last_not_of(" \t\n\r\f\v") + 1);
                
                LOGD("server", std::string("Received command (string conversion): '") + command + "'");

                if (command == "REQUEST")
                {
                    // REQUEST：将命令推入队列，由 GTAV 脚本线程（script.cpp）处理
                    g_cmdQueue.push(command); 
                    LOGI("server", "Command added to queue: 'REQUEST'");

                    // 立即关闭连接并重新开始接受新连接
                    if (socket_.is_open()) {
                        socket_.close(); 
                    }
                    start_accept(); 
                }
                else if (command == "CHECK") {
                    // 检查是否捕获RGBD完成
					std::string catch_flag = (cmdToCatch == catchStop) ? "READY" : "NOTREADY";
                    LOGI("server", std::string("Command recognized: CHECK. Capture status: ") + catch_flag);
                    if (cmdToCatch == catchStop) {
                        send_data_async(std::vector<unsigned char>{'R','E','A','D','Y'});
                    } else {
                        send_data_async(std::vector<unsigned char>{'N','O','T','R','E','A','D','Y'});
                    }
                }
                else if (command == "CAPTURE")
                {
                    // CAPTURE：立即发送上次捕获的文件
                    LOGI("server", "Command recognized: CAPTURE. Sending last captured data.");
                    
                    // 从全局路径变量中获取数据（这些路径是在 REQUEST 期间设置的）
                    std::vector<unsigned char> rgb_data = GetBytes(g_rgbCapturedFilePath);
                    std::vector<unsigned char> depth_data = GetBytes(g_depthCapturedFilePath);
                    std::vector<unsigned char> stencil_data = GetBytes(g_stencilCapturedFilePath);
                    std::vector<unsigned char> matrix_data = GetBytes(g_matrixCapturedFilePath);

                    // 检查数据是否有效，如果无效（例如大小为0），则发送错误或空数据
                    if (rgb_data.empty() || depth_data.empty()) {
                        LOGE("server", "Error: RGB or Depth data is empty. Was REQUEST command sent?");
                        std::string error_resp = "ERROR: Last capture data not ready.";
                        boost::asio::async_write(socket_, boost::asio::buffer(error_resp), [this](const boost::system::error_code& write_error, size_t) {
                            if (write_error) LOGE("server", std::string("Error sending error response: ") + write_error.message());
                            if (socket_.is_open()) socket_.close();
                            start_accept();
                        });
                        return; // 结束处理
                    }
                    
                    // 准备组合数据 
                    std::vector<unsigned char> combined_data;
                    uint32_t rgb_size = static_cast<uint32_t>(rgb_data.size());
                    combined_data.insert(combined_data.end(), reinterpret_cast<unsigned char*>(&rgb_size), reinterpret_cast<unsigned char*>(&rgb_size) + sizeof(uint32_t));
                    uint32_t depth_size = static_cast<uint32_t>(depth_data.size());
                    combined_data.insert(combined_data.end(), reinterpret_cast<unsigned char*>(&depth_size), reinterpret_cast<unsigned char*>(&depth_size) + sizeof(uint32_t));
                    // uint32_t stencil_size = static_cast<uint32_t>(stencil_data.size());
                    // combined_data.insert(combined_data.end(), reinterpret_cast<unsigned char*>(&stencil_size), reinterpret_cast<unsigned char*>(&stencil_size) + sizeof(uint32_t));
                    // uint32_t matrix_size = static_cast<uint32_t>(g_matrixCapturedFilePath.size());
                    // combined_data.insert(combined_data.end(), reinterpret_cast<unsigned char*>(&matrix_size), reinterpret_cast<unsigned char*>(&matrix_size) + sizeof(uint32_t));
                    
                    combined_data.insert(combined_data.end(), rgb_data.begin(), rgb_data.end());
                    combined_data.insert(combined_data.end(), depth_data.begin(), depth_data.end());
                    // combined_data.insert(combined_data.end(), stencil_data.begin(), stencil_data.end());
                    // combined_data.insert(combined_data.end(), matrix_data.begin(), matrix_data.end());
                    
                    LOGI("server", std::string("Combined image data prepared with total size: ") + std::to_string(combined_data.size()) + " bytes");
                    send_data_async(std::move(combined_data)); // 异步发送数据
                    
                    // 注意：send_data_async 会在发送完成后关闭连接并调用 start_accept()
                }
                else
                {
                    g_cmdQueue.push(command); // 将命令添加到队列中
                    LOGI("server", std::string("Command added to queue: '") + command + "'");
                    if (socket_.is_open()) {
                        socket_.close(); // 关闭当前连接
                    }
                    start_accept(); // 重新开始接受新连接
                }
                // set_status_text("Command received: " + command);
                // else
                // {
                //     log_to_pedTxt("Unknown command received: '" + command + "'. Sending error response.", SERVER_LOG_FILE);
                //     std::string unknown_resp = "ERROR: Unknown command.";
                //     boost::asio::async_write(socket_, boost::asio::buffer(unknown_resp),
                //         [this](const boost::system::error_code& write_error, size_t){
                //         if (write_error) log_to_pedTxt("Error sending unknown command response: " + write_error.message(), SERVER_LOG_FILE);
                //         if (socket_.is_open()) {
                //             socket_.close();
                //         }
                //         start_accept(); // 重新开始接受连接
                //     });
                // }
            }
            else
            {
                LOGE("server", std::string("Error receiving command (read_some): ") + error.message());
                LOGE("server", std::string("Error value: ") + std::to_string(error.value()));
                
                // 检查是否是连接重置错误
                if (error == boost::asio::error::connection_reset || 
                    error == boost::asio::error::eof ||
                    error.value() == 10054) // Windows连接重置错误
                {
                    LOGW("server", "Client disconnected before sending data");
                }
                
                if (socket_.is_open()) {
                    socket_.close();
                }
                LOGW("server", "Connection closed due to read error.");
                
                // 重新开始接受新连接
                start_accept();
            }
        });
}

void ModServer::send_data_async(std::vector<unsigned char> data) {
    // 检查socket状态
    if (!socket_.is_open()) {
        LOGW("server", "Attempted to send data on a closed or invalid socket.");
        start_accept(); // 重新开始接受连接
        return;
    }

    LOGD("server", std::string("Preparing to send data of size: ") + std::to_string(data.size()) + " bytes");

    // 使用shared_ptr管理数据生命周期
    auto shared_data = std::make_shared<std::vector<unsigned char>>(std::move(data));
    
    // 准备长度数据 (小端序)
    auto length_bytes = std::make_shared<std::vector<unsigned char>>(4);
    uint32_t data_len = static_cast<uint32_t>(shared_data->size());
    std::memcpy(length_bytes->data(), &data_len, sizeof(data_len));

    LOGD("server", std::string("Sending data length: ") + std::to_string(data_len) + " bytes");

    // 先发送长度
    boost::asio::async_write(socket_, boost::asio::buffer(*length_bytes),
        [this, shared_data, length_bytes](const boost::system::error_code& error_len, size_t bytes_transferred_len) {
        if (!error_len) {
            LOGD("server", std::string("Successfully sent length header: ") + std::to_string(bytes_transferred_len) + " bytes");
            
            // 发送实际数据
            boost::asio::async_write(socket_, boost::asio::buffer(*shared_data),
                [this, shared_data](const boost::system::error_code& error_data, size_t bytes_transferred_data) {
                if (!error_data) {
                    LOGI("server", std::string("Image data sent successfully: ") + std::to_string(bytes_transferred_data) + " bytes");
                } else {
                    LOGE("server", std::string("Error sending image data: ") + error_data.message());
                }
                
                // 发送完成后关闭连接并重新开始接受新连接
                if (socket_.is_open()) {
                    socket_.close();
                }
                LOGI("server", "Connection closed after sending image data.");
                
                // 重新开始接受新连接
                start_accept();
            });
        } else {
            LOGE("server", std::string("Error sending image data length: ") + error_len.message());
            if (socket_.is_open()) {
                socket_.close();
            }
            LOGW("server", "Connection closed due to image data length send error.");
            
            // 重新开始接受新连接
            start_accept();
        }
    });
}

static bool g_winsock_initialized = false;

void InitializeModServer()
{
    // 确保只初始化一次 Winsock
    if (!g_winsock_initialized) {
        WSADATA wsaData;
        int result = WSAStartup(MAKEWORD(2, 2), &wsaData); // 请求 Winsock 2.2 版本
        if (result != 0) {
            LOGE("server", std::string("WSAStartup failed with error: ") + std::to_string(result));
            return; // WSAStartup 失败，不继续初始化服务器
        }
        LOGI("server", "WSAStartup successfully called.");
        g_winsock_initialized = true;
    }
    
    // 检查是否已经初始化过，避免重复启动
    if (g_modServerInstance) {
        LOGW("server", "Mod Server already initialized. Skipping.");
        return;
    }

    try
    {
        // 在新线程中运行 io_context
        g_serverThread = std::make_unique<std::thread>([]() {
            try {
                g_modServerInstance = std::make_unique<ModServer>(g_ioContext, 12345);
                LOGI("server", "Starting io_context.run()...");
                g_ioContext.run(); // 运行 io_context，它会阻塞直到所有任务完成或 stop() 被调用
                LOGI("server", "io_context stopped running.");
            } catch (const std::exception& e) {
                LOGE("server", std::string("Server thread exception caught: ") + e.what());
            }
        });
        g_serverThread->detach(); // 分离线程，让它独立运行

        LOGI("server", "Mod Server initialization sequence started.");
    }
    catch (const std::exception& e)
    {
        LOGE("server", std::string("Mod Server Initialization Exception: ") + e.what());
    }
}

// 在你的Mod卸载时调用此函数，用于清理资源
void ShutdownModServer()
{
    LOGI("server", "Mod Server shutdown sequence initiated.");
    
    // 停止 io_context，这将导致 g_ioContext.run() 返回，从而结束服务器线程
    g_ioContext.stop();

    // 清理 ModServer 实例和线程指针
    g_modServerInstance.reset();
    g_serverThread.reset();

    LOGI("server", "Mod Server resources cleaned up.");
}
