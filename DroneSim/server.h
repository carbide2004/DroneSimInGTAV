#pragma once

#include "proto.h"

#include <boost/asio.hpp>

#include <string>
#include <vector>

class Server {
public:
    Server(boost::asio::io_context& io, unsigned short port);

private:
    void start_accept();
    void handle_client();
    void write_response(const std::vector<unsigned char>& data);
    bool read_exact(
        boost::asio::ip::tcp::socket& socket,
        void* buffer,
        std::size_t length);

    boost::asio::ip::tcp::acceptor acceptor_;
    boost::asio::ip::tcp::socket socket_;
};

void InitializeServer();
void ShutdownServer();
