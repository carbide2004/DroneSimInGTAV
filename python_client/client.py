import socket
import struct
import numpy as np
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import time
import os
from datetime import datetime

HOST = '127.0.0.1'
PORT = 12345
WIDTH = 2560
HEIGHT = 1440

# 确保 'record' 文件夹存在
def ensure_record_dir_exists():
    if not os.path.exists("record"):
        os.makedirs("record")
        print("已创建 'record' 文件夹。")

def send_camera_command(command: str):
    """
    连接服务器，发送单个相机控制指令，然后关闭连接。
    此函数仅用于发送控制指令，不接收数据。
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            print(f"正在发送相机控制指令: '{command}'")
            s.sendall(command.encode('utf-8'))
            print(f"指令 '{command}' 已发送。")
            
    except ConnectionRefusedError:
        print("连接失败。请确保C++服务器正在运行并监听正确的IP和端口。")
    except Exception as e:
        print(f"发送命令时发生错误: {e}")

def get_data_from_server(command: str):
    """
    连接服务器，发送指令，接收合并的RGB和深度数据。
    数据格式为：TOTAL_LENGTH(4字节) + RGB_SIZE(4字节) + DEPTH_SIZE(4字节) + RGB_DATA + DEPTH_DATA。
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            print(f"\n正在发送数据获取指令: '{command}'")

            s.sendall(command.encode('utf-8'))
            
            # 首先接收总数据长度
            total_length_bytes = s.recv(4)
            if not total_length_bytes:
                print("未接收到总数据长度信息，服务器可能已关闭连接。")
                return None, None
            total_data_length = struct.unpack('<I', total_length_bytes)[0]
            print(f"接收到总数据长度信息: {total_data_length} 字节。")

            if total_data_length == 0:
                print("接收到的总数据长度为0。")
                return None, None

            # 接收所有合并的数据
            combined_data = b""
            while len(combined_data) < total_data_length:
                remaining_bytes = total_data_length - len(combined_data)
                chunk = s.recv(min(4096, remaining_bytes))
                if not chunk:
                    print("服务器在数据传输完成前断开连接。")
                    return None, None
                combined_data += chunk
            
            print(f"成功接收到所有合并数据，共 {len(combined_data)} 字节。")

            # 从合并数据中解析RGB和深度数据
            offset = 0
            rgb_data_length = struct.unpack('<I', combined_data[offset:offset+4])[0]
            offset += 4
            depth_data_length = struct.unpack('<I', combined_data[offset:offset+4])[0]
            offset += 4

            rgb_data = combined_data[offset : offset + rgb_data_length]
            offset += rgb_data_length
            depth_data = combined_data[offset : offset + depth_data_length]

            print(f"解析出RGB数据长度: {len(rgb_data)} 字节，深度数据长度: {len(depth_data)} 字节。")
            
            return rgb_data, depth_data

    except ConnectionRefusedError:
        print("连接失败。请确保C++服务器正在运行并监听正确的IP和端口。")
    except Exception as e:
        print(f"发生错误: {e}")
    return None, None

def save_rgb_image(image_data):
    """
    将字节数据转换为RGB图片并保存到文件。
    """
    if not image_data:
        print("没有RGB图片数据可保存。")
        return
    try:
        img = Image.open(BytesIO(image_data))
        print(f"RGB图片成功接收。格式: {img.format}, 大小: {img.size}, 模式: {img.mode}")
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
        filename = f"record/rgb_{timestamp}.png"
        img.save(filename)
        print(f"RGB图片已保存为 {filename}")
    except IOError as e:
        print(f"无法将接收到的数据转换为图片: {e}")

def save_depth_image(image_data):
    """
    将字节数据转换为深度图片并保存到文件。
    """
    if not image_data:
        print("没有深度图片数据可保存。")
        return
    try:
        expected_size = WIDTH * HEIGHT * 4
        if len(image_data) != expected_size:
            print(f"警告: 深度数据大小不匹配。预期 {expected_size} 字节，实际 {len(image_data)} 字节。")
        
        depth_array = np.frombuffer(image_data, dtype=np.float32)
        depth_array = depth_array.reshape((HEIGHT, WIDTH))
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
        filename = f"record/depth_{timestamp}.png"
        plt.imsave(filename, depth_array, cmap='gray')
        print(f"深度图片已保存为 {filename}")

    except ValueError as e:
        print(f"无法将数据转换为深度图: {e}")
    except Exception as e:
        print(f"保存深度图时发生错误: {e}")

if __name__ == "__main__":
    ensure_record_dir_exists()
    
    for i in range(3):
        print(f"\n等待 {3 - i} 秒后开始发送命令...")
        time.sleep(1)
    
    commands = [
        "FORWARD", 
        "RIGHTROTATE", 
        "BACKWARD", 
        "LEFTROTATE",
        "UP",
        "DOWN"
    ]
    
    for cmd in commands:
        send_camera_command(cmd)
        time.sleep(0.1)
        
        rgb_data, depth_data = get_data_from_server("CAPTURE")
        if rgb_data:
            save_rgb_image(rgb_data)
        if depth_data:
            save_depth_image(depth_data)

        time.sleep(0.1)
    
    print("\n所有命令及其数据获取任务已完成。")