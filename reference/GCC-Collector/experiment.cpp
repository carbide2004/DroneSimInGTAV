#include <iostream>
#include <vector>
#include <string>
#include <windows.h> // 需要包含 Windows.h


// 模拟一个资源密集型类
class MyResource {
public:
    std::vector<int> data;
    std::string name;

    // 构造函数
    MyResource(size_t size, const std::string& n) : data(size), name(n) {
        std::cout << "MyResource(" << name << ") - 构造函数 (size=" << size << ")" << std::endl;
        for (size_t i = 0; i < size; ++i) {
            data[i] = i;
        }
    }

    // 拷贝构造函数
    MyResource(const MyResource& other) : data(other.data), name(other.name) {
        std::cout << "MyResource(" << name << ") - 拷贝构造函数" << std::endl;
    }

    // 移动构造函数
    MyResource(MyResource&& other) noexcept : data(std::move(other.data)), name(std::move(other.name)) {
        std::cout << "MyResource(" << name << ") - 移动构造函数" << std::endl;
        // 确保源对象处于有效但空的状态
        // other.data.clear(); // 这一步通常由 std::vector::operator= 内部完成，此处仅为演示
        // other.name.clear(); // 同上
    }

    // 拷贝赋值运算符
    MyResource& operator=(const MyResource& other) {
        if (this != &other) {
            data = other.data;
            name = other.name;
            std::cout << "MyResource(" << name << ") - 拷贝赋值运算符" << std::endl;
        }
        return *this;
    }

    // 移动赋值运算符
    MyResource& operator=(MyResource&& other) noexcept {
        if (this != &other) {
            data = std::move(other.data);
            name = std::move(other.name);
            std::cout << "MyResource(" << name << ") - 移动赋值运算符" << std::endl;
            // 确保源对象处于有效但空的状态
        }
        return *this;
    }

    // 析构函数
    ~MyResource() {
        std::cout << "MyResource(" << name << ") - 析构函数" << std::endl;
    }
};

// 接受 MyResource 对象的函数
void processResource(MyResource res) {
    std::cout << "进入 processResource，处理资源: " << res.name << std::endl;
}

int main() {
    std::ios_base::sync_with_stdio(false);
    SetConsoleOutputCP(CP_UTF8);
    std::cout << "--- 场景1: 复制 ---" << std::endl;
    MyResource res1(1000, "Original"); // 构造函数
    MyResource res2 = res1;            // 拷贝构造函数
    std::cout << "res1 名称: " << res1.name << std::endl; // Original
    std::cout << "res2 名称: " << res2.name << std::endl; // Original

    std::cout << "\n--- 场景2: 移动 (使用 std::move) ---" << std::endl;
    MyResource res3(500, "Source"); // 构造函数
    MyResource res4 = std::move(res3); // 移动构造函数被调用
    std::cout << "res3 名称 (移动后): " << res3.name << std::endl; // 通常为空或未定义，取决于具体实现，例如空字符串 ""
    std::cout << "res4 名称 (移动后): " << res4.name << std::endl; // Source

    std::cout << "\n--- 场景3: 函数参数传递 (复制 vs 移动) ---" << std::endl;
    MyResource res5(200, "ForCopy");
    std::cout << "调用 processResource (复制传入):" << std::endl;
    processResource(res5); // 调用拷贝构造函数
    std::cout << "res5 名称 (复制传入后): " << res5.name << std::endl; // ForCopy

    MyResource res6(200, "ForMove");
    std::cout << "调用 processResource (移动传入):" << std::endl;
    processResource(std::move(res6)); // 调用移动构造函数
    std::cout << "res6 名称 (移动传入后): " << res6.name << std::endl; // 通常为空或未定义

    std::cout << "\n--- Main 函数结束 ---" << std::endl;
    return 0;
}