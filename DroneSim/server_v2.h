#pragma once
#include <boost/asio.hpp>
#include <vector>
#include <string>
#include "proto.h"

class ServerV2 {
public:
    ServerV2(boost::asio::io_context& io, unsigned short port);
private:
    void start_accept();
    void handle_client();
    void write_response(const std::vector<unsigned char>& data);
    bool read_exact(boost::asio::ip::tcp::socket& s, void* buf, size_t len);
    boost::asio::ip::tcp::acceptor acceptor_;
    boost::asio::ip::tcp::socket socket_;
};

void InitializeServerV2();
void ShutdownServerV2();
