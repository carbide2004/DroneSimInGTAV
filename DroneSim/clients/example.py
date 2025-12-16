from dronesim_client import DroneSimClient, visualize
import sys
import time

def main():
    print("5s后开始测试")
    time.sleep(5)

    cli = DroneSimClient()
    cam_id = cli.create_camera()
    cli.set_fov(60.0)
    cli.set_time(12, 0, 0)
    cli.set_weather("RAIN")
    pose = cli.get_pose()
    print("pose:", pose)
    cli.move(5.0, 0.0, 0.0)
    cli.rotate(0.0, 0.0, 45.0)
    time.sleep(5)
    cap = cli.capture()
    if not cap:
        print("capture failed")
        sys.exit(1)
    w, h, rgb, depth = cap
    ok = visualize(rgb, depth, w, h)
    if not ok:
        with open("rgb.bin", "wb") as f:
            f.write(rgb)
        with open("depth.bin", "wb") as f:
            f.write(depth)
        print("saved rgb.bin and depth.bin")
    cli.stop_camera()

if __name__ == "__main__":
    main()
