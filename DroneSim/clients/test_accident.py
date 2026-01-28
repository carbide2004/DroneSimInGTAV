from dronesim_client import DroneSimClient
import time

def main():
    print("Waiting 5 seconds...")
    time.sleep(5)
    
    client = DroneSimClient()
    print("Entering camera mode...")
    client.create_camera()
    
    print("Setting time to 12:00:00...")
    client.set_time(12, 0, 0)
    
    print("Creating accident...")
    pos = client.create_accident()
    if pos:
        print(f"Accident created at: x={pos[0]:.2f}, y={pos[1]:.2f}, z={pos[2]:.2f}")
        pose = client.get_pose()
        if pose:
            x,y,z,rx,ry,rz = pose
            tx,ty,tz = pos[0], pos[1], pos[2] + 50.0
            dx,dy,dz = tx - x, ty - y, tz - z
            print(f"Moving by: dx={dx:.2f}, dy={dy:.2f}, dz={dz:.2f}")
            client.move(dx,dy,dz)
            delta_rx = -90.0 - rx
            print(f"Rotating pitch by: {delta_rx:.2f}")
            client.rotate(delta_rx, 0.0, 0.0)
            print("Waiting 30 seconds...")
            time.sleep(30)
        else:
            print("Failed to get current pose")
    else:
        print("Failed to create accident (timeout or error)")
        
    print("Exiting camera mode...")
    client.stop_camera()

if __name__ == "__main__":
    main()
