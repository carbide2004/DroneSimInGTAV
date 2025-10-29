import os
import time
import numpy as np
from PIL import Image
import open3d as o3d
import math
from io import BytesIO
import matplotlib.pyplot as plt
from client import send_camera_command, get_data_from_server, save_rgb_image, save_depth_image, ensure_record_dir_exists, WIDTH, HEIGHT, FOV

ROOT_DATA_FOLDER = "record\\record_data_cloud_" + time.strftime("%Y%m%d_%H%M%S", time.localtime())

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
X_MIN, X_MAX, X_STEP = 0, 0, 10.0
Y_MIN, Y_MAX, Y_STEP = 0, 0, 10.0
Z_MIN, Z_MAX, Z_STEP = 0.0, 0.0, 5.0 

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

# 确保 'record' 文件夹存在
def ensure_record_dir_exists():
    if not os.path.exists(ROOT_DATA_FOLDER):
        os.makedirs(ROOT_DATA_FOLDER)
        print(f"已创建 '{ROOT_DATA_FOLDER}' 文件夹。")

# 相机内参 (根据实际相机参数调整)

def calc_camera_intrinsics(width, height, fov):
    fx = width / (2 * math.tan(fov / 2))
    fy = height / (2 * math.tan(fov / 2))
    cx = width / 2
    cy = height / 2
    return o3d.camera.PinholeCameraIntrinsic(
        width, height, # 图像宽度和高度
        fx, fy, # 焦距 fx (像素)
        cx, cy  # 主点 cx (像素)
    )

INTRINSICS = calc_camera_intrinsics(WIDTH, HEIGHT, FOV)

def capture_rgbd_data():
    """
    获取RGB和深度图像数据，并返回NumPy数组。
    """
    while True:
        send_camera_command("REQUEST")
        time.sleep(1) # 等待服务器处理请求
        rgb_data_bytes, depth_data_bytes = get_data_from_server("CAPTURE")
        if not rgb_data_bytes or not depth_data_bytes:
            print("未获取到有效数据，重试...")
            continue

        expected_depth_size = WIDTH * HEIGHT * 4 # float32 is 4 bytes
        if len(depth_data_bytes) != expected_depth_size:
            print(f"警告: 深度数据大小不匹配。预期 {expected_depth_size} 字节，实际 {len(depth_data_bytes)} 字节。重试...")
            continue
        break
    
    # 将RGB字节数据转换为PIL Image，再转换为NumPy数组
    rgb_image = Image.open(BytesIO(rgb_data_bytes)).convert("RGB")
    rgb_array = np.array(rgb_image)

    # 将深度字节数据转换为NumPy数组
    depth_array = np.frombuffer(depth_data_bytes, dtype=np.float32).reshape((HEIGHT, WIDTH)).copy()

    # 避免除以零或非常小的值，设置一个阈值
    depth_array[depth_array < 1e-8] = 1e-8 # 避免除以零
    depth_array = 1.0 / depth_array

    # 可视化深度图像（旁边带有颜色条）
    # plt.imshow(depth_array, cmap='gray')
    # plt.title("Depth Image")
    # plt.colorbar(label="Distance (m)")
    # plt.show()
    
    return rgb_array, depth_array

def rgbd_to_pointcloud(rgb_image_np, depth_image_np, camera_intrinsics, drone_position, drone_yaw):
    """
    将RGB图像和深度图像转换为彩色点云，并进行坐标变换。
    """
    # 创建Open3D的RGBDImage对象
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(rgb_image_np), 
        o3d.geometry.Image(depth_image_np), 
        depth_scale=1.0, # 深度值单位为米
        depth_trunc=1000.0, # 最大深度值
        convert_rgb_to_intensity=False
    )

    # 使用相机内参创建点云
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, 
        camera_intrinsics
    )

    # --- 关键的坐标变换部分 ---
    
    # 1. 定义相机坐标系到无人机(世界)坐标系的固定转换 R_fixed
    # 假设：
    # 相机坐标系(cam): X-右, Y-下, Z-前 (Open3D默认)
    # 世界坐标系(world): X-前(0度yaw), Y-左(90度yaw), Z-上
    # 
    # 转换目标：Z_cam -> X_world, -X_cam -> Y_world, -Y_cam -> Z_world
    
    R_fixed = np.array([
        [ 0.,  0.,  1.],
        [-1.,  0.,  0.],
        [ 0., -1.,  0.]
    ])

    # 2. 定义无人机姿态旋转 R_pose (绕世界Z轴旋转 yaw 角度)
    yaw_rad = np.deg2rad(drone_yaw)
    cos_yaw = np.cos(yaw_rad)
    sin_yaw = np.sin(yaw_rad)

    # R_pose 绕Z轴旋转 (世界坐标系):
    #   [ cos, -sin, 0 ]
    #   [ sin, cos,  0 ]
    #   [ 0,   0,    1 ]
    R_pose = np.array([
        [ cos_yaw, -sin_yaw, 0.],
        [ sin_yaw,  cos_yaw, 0.],
        [ 0.,       0.,      1.]
    ])

    # 3. 计算最终的旋转矩阵 R_world = R_pose * R_fixed
    R_world = R_pose @ R_fixed
    
    # 4. 构建 4x4 齐次变换矩阵 T
    t_world = np.array(drone_position).reshape((3, 1))
    
    T_world = np.eye(4)
    T_world[:3, :3] = R_world
    T_world[:3, 3] = t_world.flatten()
    
    # 5. 应用变换
    # Open3D的 transform 方法接受 4x4 矩阵
    pcd.transform(T_world)

    return pcd

if __name__ == "__main__":
    ensure_record_dir_exists()

    print("在5秒后开始无人机数据采集...")
    time.sleep(5)

    # 初始化一个空的全局点云
    global_point_cloud = o3d.geometry.PointCloud()

    for x in np.arange(X_MIN, X_MAX + X_STEP, X_STEP):
        for y in np.arange(Y_MIN, Y_MAX + Y_STEP, Y_STEP):
            for z in np.arange(Z_MIN, Z_MAX + Z_STEP, Z_STEP):
                target_position = (x, y, z)
                print(f"\n--- 正在移动到位置: {target_position} ---")
                move_to_position(x, y, z)

                for yaw in HORIZONTAL_ORIENTATIONS:
                    print(f"--- 正在旋转到朝向: {yaw} 度 ---")
                    rotate_to_yaw(yaw)

                    rgb_array, depth_array = capture_rgbd_data()
                    if rgb_array is not None and depth_array is not None:
                        # 将RGBD数据转换为点云
                        current_pcd = rgbd_to_pointcloud(
                            rgb_array, depth_array, INTRINSICS, 
                            (current_drone_x, current_drone_y, current_drone_z), 
                            current_drone_yaw
                        )
                        
                        # 将当前点云合并到全局点云中
                        global_point_cloud += current_pcd
                        # 可视化全局点云
                        # print("正在可视化全局点云...")
                        # o3d.visualization.draw_geometries([global_point_cloud])

                        print(f"已获取并合并位置 ({x:.2f}, {y:.2f}, {z:.2f}), 朝向 {yaw:.2f} 度的点云。")
                    else:
                        print(f"未能获取位置 ({x:.2f}, {y:.2f}, {z:.2f}), 朝向 {yaw:.2f} 度的RGBD数据。")
                    # time.sleep(0.5) # 每次拍摄后稍作等待

    print("\n所有数据采集任务已完成。")
    print(f"全局点云包含 {len(global_point_cloud.points)} 个点。")

    # 进行体素降采样
    # voxel_size = 0.1 
    # print(f"正在对点云进行体素下采样，体素大小为 {voxel_size}...")
    # downsampled_pcd = global_point_cloud.voxel_down_sample(voxel_size=voxel_size)
    # print(f"下采样后点云包含 {len(downsampled_pcd.points)} 个点。")

    # 保存点云
    output_filename = os.path.join(ROOT_DATA_FOLDER, "scene_point_cloud.pcd")
    os.makedirs(ROOT_DATA_FOLDER, exist_ok=True)
    o3d.io.write_point_cloud(output_filename, global_point_cloud)
    print(f"彩色点云已保存到: {output_filename}")

    # 可视化全局点云
    print("正在可视化全局点云...")
    o3d.visualization.draw_geometries([global_point_cloud])
