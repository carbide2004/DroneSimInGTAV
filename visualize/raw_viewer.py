import numpy as np
import matplotlib.pyplot as plt
import os

def visualize_raw_file(file_path, width, height, dtype=np.uint8):
    """
    可视化 .raw 文件。

    Args:
        file_path (str): .raw 文件的路径。
        width (int): 图像的宽度。
        height (int): 图像的高度。
        dtype (numpy.dtype): 数据的类型，例如 np.uint8, np.float32。
    """
    if not os.path.exists(file_path):
        print(f"错误：文件 '{file_path}' 不存在。")
        return

    try:
        # 读取原始数据
        data = np.fromfile(file_path, dtype=dtype)

        # 检查数据大小是否与预期相符
        expected_size = width * height
        if data.size != expected_size:
            print(f"警告：文件 '{file_path}' 的数据大小 ({data.size}) 与预期大小 ({expected_size}) 不符。")
            # 尝试根据实际数据大小调整形状，如果可能的话
            if data.size > expected_size and data.size % expected_size == 0:
                print(f"尝试将数据重塑为 {height}x{width}x{data.size // expected_size}。")
                image = data.reshape((height, width, data.size // expected_size))
            elif data.size < expected_size:
                print("数据不足以形成完整的图像。")
                return
            else:
                print("无法将数据重塑为预期的图像尺寸。")
                return
        else:
            # 将一维数据重塑为二维图像
            image = data.reshape((height, width))

        # 显示图像
        plt.imshow(image, cmap='gray') # 假设是灰度图像，如果需要彩色，可能需要调整
        plt.title(f"可视化: {os.path.basename(file_path)}")
        plt.colorbar(label='像素值')
        plt.show()

    except Exception as e:
        print(f"处理文件 '{file_path}' 时发生错误：{e}")

if __name__ == "__main__":
    # 示例用法：
    # 假设你有一个名为 'depth.raw' 的文件，宽度为 2560，高度为 1440，数据类型为 np.float32
    # 请根据你的实际文件路径进行修改
    raw_file_path = os.path.join(os.getcwd(), "temp_capture_depth_1759459869618.raw") # 假设 raw 文件在当前工作目录
    image_width = 2560  # 图像宽度
    image_height = 1440 # 图像高度
    data_type = np.float32 # 数据类型为单通道浮点数 (32位)

    print(f"尝试可视化文件：{raw_file_path}")
    visualize_raw_file(raw_file_path, image_width, image_height, data_type)

    # 如果你的 .raw 文件是其他数据类型，例如 float32，可以这样调用：
    # visualize_raw_file("path/to/your/float_data.raw", 1024, 768, np.float32)