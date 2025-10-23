import os
import time
import numpy as np
from PIL import Image
from client import send_camera_command, get_data_from_server, save_rgb_image, save_depth_image, ensure_record_dir_exists

ROOT_DATA_FOLDER = "record_data_" + time.strftime("%Y%m%d_%H%M%S", time.localtime())

# 无人机移动和旋转的步长
MOVE_STEP = 5.0  # 每次 FORWARD/BACKWARD 移动的距离 (米)
ROTATE_STEP = 45.0 # 每次 RIGHTROTATE/LEFTROTATE 旋转的角度 (度)

# 无人机当前估计位置和朝向 (x, y, z, yaw)
# 初始位置假设为 (0, 0, 0)，初始朝向为 0 度 (沿X轴正方向)
current_drone_x = 0.0
current_drone_y = 0.0
current_drone_z = 0.0
current_drone_yaw = 0.0 # 0-360度

# 定义遍历范围
X_MIN, X_MAX, X_STEP = -50.0, 50.0, 10.0
Y_MIN, Y_MAX, Y_STEP = -50.0, 50.0, 10.0
Z_MIN, Z_MAX, Z_STEP = 0.0, 10.0, 5.0 

# 定义水平八个朝向 (yaw)
HORIZONTAL_ORIENTATIONS = [
        0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0
    ]

def rotate_to_yaw(target_yaw):
    global current_drone_yaw
    # 将目标偏航角规范化到 [0, 360) 范围
    target_yaw = target_yaw % 360
    if target_yaw < 0: 
        target_yaw += 360

    # 计算最短旋转角度
    delta_yaw = target_yaw - current_drone_yaw
    if delta_yaw > 180:
        delta_yaw -= 360
    elif delta_yaw < -180:
        delta_yaw += 360

    # 根据 delta_yaw 发送旋转指令
    while abs(delta_yaw) > ROTATE_STEP / 2: # 允许一定的误差范围
        if delta_yaw > 0:
            send_camera_command("RIGHTROTATE")
            current_drone_yaw = (current_drone_yaw + ROTATE_STEP) % 360
            delta_yaw -= ROTATE_STEP
        else:
            send_camera_command("LEFTROTATE")
            current_drone_yaw = (current_drone_yaw - ROTATE_STEP + 360) % 360
            delta_yaw += ROTATE_STEP
        time.sleep(0.01)
    print(f"无人机已旋转到朝向: {current_drone_yaw:.2f} 度")

def move_to_position(target_x, target_y, target_z):
    global current_drone_x, current_drone_y, current_drone_z, current_drone_yaw

    # 移动到目标Z坐标 (高度)
    while abs(target_z - current_drone_z) > MOVE_STEP / 2:
        if target_z > current_drone_z:
            send_camera_command("UP")
            current_drone_z += MOVE_STEP
        else:
            send_camera_command("DOWN")
            current_drone_z -= MOVE_STEP
        time.sleep(0.01)

    # 调整X坐标
    rotate_to_yaw(0) # 旋转到0度，假设0度是X轴正方向
    while abs(target_x - current_drone_x) > MOVE_STEP / 2:
        if target_x > current_drone_x:
            send_camera_command("FORWARD") # 沿X轴正方向移动
            current_drone_x += MOVE_STEP
        else:
            send_camera_command("BACKWARD") # 沿X轴负方向移动
            current_drone_x -= MOVE_STEP
        time.sleep(0.01)

    # 调整Y坐标
    rotate_to_yaw(90) # 旋转到90度，假设90度是Y轴正方向
    while abs(target_y - current_drone_y) > MOVE_STEP / 2:
        if target_y > current_drone_y:
            send_camera_command("FORWARD") # 沿Y轴正方向移动
            current_drone_y += MOVE_STEP
        else:
            send_camera_command("BACKWARD") # 沿Y轴负方向移动
            current_drone_y -= MOVE_STEP
        time.sleep(0.01)

    print(f"无人机已移动到位置: ({current_drone_x:.2f}, {current_drone_y:.2f}, {current_drone_z:.2f})")

def capture_and_save_data(position, orientation):
    """
    在指定位置和方向拍摄RGB和深度图像，并结构化存储。
    """
    pos_str = f"pos_{position[0]:.2f}_{position[1]:.2f}_{position[2]:.2f}".replace('.', '_')
    ori_str = f"ori_{orientation[0]:.2f}".replace('.', '_')

    save_dir = os.path.join(ROOT_DATA_FOLDER, pos_str, ori_str)
    os.makedirs(save_dir, exist_ok=True)
    print(f"创建保存目录: {save_dir}")

    rgb_data, depth_data = get_data_from_server("CAPTURE")
    if not rgb_data or not depth_data:
        print("未获取到有效数据，跳过保存。")
        return False
    if rgb_data:
        save_rgb_image(rgb_data, os.path.join(save_dir, "rgb.png"))
    if depth_data:
        save_depth_image(depth_data, os.path.join(save_dir, "depth.png"))
    return True

# 确保 'record' 文件夹存在
def ensure_record_dir_exists():
    if not os.path.exists(ROOT_DATA_FOLDER):
        os.makedirs(ROOT_DATA_FOLDER)
        print(f"已创建 '{ROOT_DATA_FOLDER}' 文件夹。")

if __name__ == "__main__":
    ensure_record_dir_exists()

    print("在10秒后开始无人机数据采集...")
    time.sleep(10)

    for x in np.arange(X_MIN, X_MAX + X_STEP, X_STEP):
        for y in np.arange(Y_MIN, Y_MAX + Y_STEP, Y_STEP):
            for z in np.arange(Z_MIN, Z_MAX + Z_STEP, Z_STEP):
                target_position = (x, y, z)
                print(f"\n--- 正在移动到位置: {target_position} ---")
                move_to_position(x, y, z)
                # time.sleep(0.5) # 给予无人机一些时间到达指定位置

                for yaw in HORIZONTAL_ORIENTATIONS:
                    print(f"--- 正在旋转到朝向: {yaw} 度 ---")
                    rotate_to_yaw(yaw)
                    # time.sleep(0.1) # 给予无人机一些时间旋转

                    # 捕获并保存数据
                    success = capture_and_save_data(target_position, (yaw, 0.0, 0.0)) # pitch 和 roll 暂时设为0
                    time.sleep(0.5) # 每次拍摄后稍作等待
                    while not success:
                        print("数据采集失败，重试...")
                        success = capture_and_save_data(target_position, (yaw, 0.0, 0.0))
                        time.sleep(0.5) # 每次重试后稍作等待

    print("\n所有数据采集任务已完成。")