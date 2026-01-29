from dronesim_client import DroneSimClient
import time
from datetime import datetime

def move_to_pose(cli, target_pose):
    cur = cli.get_pose()
    if not cur:
        return False
    cx, cy, cz, crx, cry, crz = cur
    tx, ty, tz, trx, try_, trz = target_pose
    cli.move(tx - cx, ty - cy, tz - cz)
    cli.rotate(trx - crx, try_ - cry, trz - crz)
    return True

def wait_recording(cli):
    print("等待你按 J 开始录制（按 K 结束）...")
    last_step = -1
    session_dir = ""
    while True:
        info = cli.get_recording_info()
        if not info:
            time.sleep(0.5)
            continue
        if info["enabled"] and not session_dir:
            session_dir = info["session_dir"]
            print(f"录制已开始，输出目录: {session_dir}")
        if info["enabled"]:
            if info["step"] != last_step:
                last_step = info["step"]
            time.sleep(1.0)
            continue
        if session_dir:
            print(f"录制已结束，最终step数: {info['step']}")
            return session_dir
        time.sleep(0.5)

def main():
    print("等待5秒...")
    time.sleep(5)

    cli = DroneSimClient()
    session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    cli.set_recording_session(session_name)

    print("进入相机模式...")
    cli.create_camera()

    print("设置时间为正午12点...")
    cli.set_time(12, 0, 0)

    print("创建车祸...")
    acc = cli.create_accident()
    if not acc:
        print("创建车祸失败")
        cli.stop_camera()
        return
    print(f"车祸坐标: x={acc[0]:.2f}, y={acc[1]:.2f}, z={acc[2]:.2f}")

    print("获取建议初始位姿并移动到位...")
    pose = cli.get_suggested_start_pose()
    if not pose:
        print("获取建议初始位姿失败")
    else:
        ok = move_to_pose(cli, pose)
        if not ok:
            print("移动到建议位姿失败")

    print("现在可以用键盘控制无人机：W/A/S/D/Shift/Ctrl/Q/E")
    print("J 开始录制，K 结束录制；每次离散动作会保存一个step（动作前RGBD+pose+action）")

    try:
        session_dir = wait_recording(cli)
        print(f"本次采集目录: {session_dir}")
    finally:
        print("退出相机模式...")
        cli.stop_camera()

if __name__ == "__main__":
    main()

