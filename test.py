#!/usr/bin/env python

# CARLA Interactive Traffic Manager Script with Video
#
# This script adds video streaming and saving capabilities to the interactive TM script.
#
# New Features:
# - Spawns a dedicated camera to replicate the spectator view.
# - Saves the camera feed to a local video file ('output.mp4') using OpenCV.
# - Streams the camera feed over the network to a separate client for live viewing.
# - The camera now dynamically follows the user-selected vehicle.

import glob
import os
import sys
import random
import time
import threading
import socket
import struct
import cv2
import numpy as np

try:
    # This path needs to be correct for your setup
    sys.path.append(glob.glob('/home/cc/c_sim/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    print("Error: CARLA egg file not found. Please verify the path in the script.")
    sys.exit()

import carla
# This path needs to be correct for your setup
sys.path.append('/home/cc/c_sim/PythonAPI/carla/')
from agents.navigation.global_route_planner import GlobalRoutePlanner

# --- Shared State for Threads ---
shared_state = {
    'selected_vehicle': None,
    'command': None,
    'simulation_running': True,
    'manual_control_vehicles': set(),
    'new_frame': None, # For passing frames from CARLA to the streaming thread
    'lock': threading.Lock() # To prevent race conditions on the new_frame
}

# --- User Input Thread ---
def user_input_thread():
    global shared_state
    while shared_state['simulation_running']:
        try:
            command = input("Enter command > ")
            shared_state['command'] = command.lower().strip()
        except EOFError:
            shared_state['simulation_running'] = False
        except Exception as e:
            print(f"Input Error: {e}")
            shared_state['simulation_running'] = False

# --- NEW: Video Streaming Thread ---
def video_streaming_thread(host='0.0.0.0', port=6969):
    global shared_state
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # This allows reusing the address, helpful for quick restarts
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host, port))
    server_socket.listen(1)
    print(f"[STREAM] Listening for a viewer client on {host}:{port}")
    
    conn = None
    try:
        conn, addr = server_socket.accept()
        print(f"[STREAM] Connection from: {addr}")

        while shared_state['simulation_running']:
            frame = None
            with shared_state['lock']:
                if shared_state['new_frame'] is not None:
                    frame = shared_state['new_frame']
                    # Don't consume the frame, just copy it, so the video writer can also use it
            
            if frame is not None:
                try:
                    # Encode the frame as JPEG for efficient network transfer
                    result, encoded_frame = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                    if result:
                        data = encoded_frame.tobytes()
                        size = len(data)
                        # Pack the size and send it, then send the data
                        conn.sendall(struct.pack('>L', size) + data)
                except (ConnectionResetError, BrokenPipeError):
                    print("[STREAM] Client disconnected.")
                    break
                except Exception as e:
                    print(f"[STREAM] Error: {e}")
                    break
            
            # Sleep briefly to control the streaming frame rate and reduce CPU usage
            time.sleep(1/30) # Aim for ~30 FPS stream

    finally:
        if conn:
            conn.close()
        server_socket.close()
        print("[STREAM] Streaming thread stopped.")


def print_menu(selected_vehicle):
    """Prints the command menu to the console."""
    print("\n" + "="*30)
    print("      COMMAND MENU")
    print("="*30)
    print("Current Focus: {}".format(f"Vehicle ID {selected_vehicle.id}" if selected_vehicle else "None"))
    print("---")
    print("select <id>   - Focus on vehicle by ID")
    print("speed <km/h>  - Set target speed")
    print("lane <l/r>    - Force lane change")
    print("brake         - Apply full brakes")
    print("release       - Release manual brake")
    print("help          - Show this menu")
    print("exit          - Quit simulation")
    print("="*30)

def main():
    global shared_state
    actor_list = []
    vehicles_list = []
    client = None
    video_writer = None
    camera = None

    try:
        # --- 1. Connect and set up the world ---
        client = carla.Client('localhost', 2000)
        client.set_timeout(20.0)
        world = client.get_world()
        spectator = world.get_spectator()
        
        # --- 2. Set up Traffic Manager and Synchronous Mode ---
        tm = client.get_trafficmanager(8005)
        tm.set_global_distance_to_leading_vehicle(2.5)
        tm.set_respawn_dormant_vehicles(True)
        
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05 # 20 FPS
        world.apply_settings(settings)
        tm.set_synchronous_mode(True)
        
        # --- 3. Spawn Vehicles ---
        blueprint_library = world.get_blueprint_library()
        spawn_points = world.get_map().get_spawn_points()
        
        vehicle_bp1 = blueprint_library.find('vehicle.tesla.model3')
        vehicle_bp1.set_attribute('color', '255,0,0')
        vehicle1 = world.spawn_actor(vehicle_bp1, spawn_points[50])
        vehicles_list.append(vehicle1)
        actor_list.append(vehicle1)

        vehicle_bp2 = blueprint_library.find('vehicle.audi.etron')
        vehicle_bp2.set_attribute('color', '0,0,255')
        vehicle2 = world.spawn_actor(vehicle_bp2, spawn_points[85])
        vehicles_list.append(vehicle2)
        actor_list.append(vehicle2)
        
        print(f"Spawned Vehicle {vehicle1.id} (Red) and Vehicle {vehicle2.id} (Blue)")

        # --- 4. Set up the spectator camera ---
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '1280')
        camera_bp.set_attribute('image_size_y', '720')
        camera = world.spawn_actor(camera_bp, carla.Transform(), attach_to=spectator)
        actor_list.append(camera)
        
        # --- 5. Set up Video Saving ---
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter('output.mp4', fourcc, 20.0, (1280, 720))

        # --- 6. Camera callback to process images ---
        def process_camera_image(image):
            global shared_state
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = np.reshape(array, (image.height, image.width, 4))
            bgr_image = array[:, :, :3]

            if video_writer:
                video_writer.write(bgr_image)
            
            with shared_state['lock']:
                shared_state['new_frame'] = bgr_image

        camera.listen(process_camera_image)

        # --- 7. Generate Route and Configure Vehicles ---
        grp = GlobalRoutePlanner(world.get_map(), 1.0)
        route = grp.trace_route(spawn_points[50].location, spawn_points[150].location)
        path_locations = [rt[0].transform.location for rt in route]
        for vehicle in vehicles_list:
            tm.set_path(vehicle, path_locations)
            vehicle.set_autopilot(True, tm.get_port())
            # For faster testing, let's hardcode the speed for now
            tm.set_desired_speed(vehicle, 50.0)

        # --- 8. Start Threads ---
        input_thread = threading.Thread(target=user_input_thread)
        input_thread.start()
        
        stream_thread = threading.Thread(target=video_streaming_thread)
        stream_thread.start()
        
        shared_state['selected_vehicle'] = vehicle1
        print_menu(shared_state['selected_vehicle'])
        
        # --- 9. Main simulation loop ---
        while shared_state['simulation_running']:
            world.tick()
            
            focus_transform = spectator.get_transform() 
            if shared_state['selected_vehicle']:
                vehicle_loc = shared_state['selected_vehicle'].get_location()
                focus_transform = carla.Transform(
                    vehicle_loc + carla.Location(z=40, x=-25), # Better camera angle
                    carla.Rotation(pitch=-30)
                )
            spectator.set_transform(focus_transform)
            
            # Command processing...
            if shared_state['command']:
                # (Command processing logic is the same)
                cmd_parts = shared_state['command'].split()
                cmd = cmd_parts[0]
                
                if cmd == 'exit':
                    shared_state['simulation_running'] = False
                elif cmd == 'help':
                    print_menu(shared_state['selected_vehicle'])
                elif cmd == 'select' and len(cmd_parts) > 1:
                    try:
                        veh_id = int(cmd_parts[1])
                        found = False
                        for v in vehicles_list:
                            if v.id == veh_id:
                                shared_state['selected_vehicle'] = v
                                print(f"Focused on Vehicle {v.id}")
                                found = True
                                break
                        if not found:
                            print(f"Vehicle with ID {veh_id} not found.")
                    except ValueError:
                        print("Invalid vehicle ID.")
                
                elif cmd == 'brake':
                    if shared_state['selected_vehicle']:
                        shared_state['manual_control_vehicles'].add(shared_state['selected_vehicle'].id)
                        print(f"Applying brakes to Vehicle {shared_state['selected_vehicle'].id}. Use 'release' to stop.")
                
                elif cmd == 'release':
                     if shared_state['selected_vehicle']:
                        shared_state['manual_control_vehicles'].discard(shared_state['selected_vehicle'].id)
                        shared_state['selected_vehicle'].set_autopilot(True, tm.get_port())
                        print(f"Released manual control for Vehicle {shared_state['selected_vehicle'].id}")

                elif cmd == 'speed' and len(cmd_parts) > 1:
                    if shared_state['selected_vehicle']:
                        try:
                            speed_val = float(cmd_parts[1])
                            tm.set_desired_speed(shared_state['selected_vehicle'], speed_val)
                            print(f"Set desired speed for Vehicle {shared_state['selected_vehicle'].id} to {speed_val} km/h")
                        except ValueError:
                            print("Invalid speed value.")
                
                elif cmd == 'lane' and len(cmd_parts) > 1:
                    if shared_state['selected_vehicle']:
                        direction = cmd_parts[1].lower()
                        if direction == 'l':
                            tm.force_lane_change(shared_state['selected_vehicle'], False)
                            print(f"Requesting Vehicle {shared_state['selected_vehicle'].id} to change lane left.")
                        elif direction == 'r':
                            tm.force_lane_change(shared_state['selected_vehicle'], True)
                            print(f"Requesting Vehicle {shared_state['selected_vehicle'].id} to change lane right.")
                
                shared_state['command'] = None
            
            for v_id in list(shared_state['manual_control_vehicles']):
                for v in vehicles_list:
                    if v.id == v_id:
                        v.apply_control(carla.VehicleControl(brake=1.0))
                        break

    finally:
        shared_state['simulation_running'] = False
        if video_writer:
            video_writer.release()
            print("Video file 'output.mp4' saved.")
        if client:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)
            tm.set_synchronous_mode(False)
            
            print('\nDestroying actors and cleaning up...')
            # It's safer to destroy the camera first
            if camera and camera.is_alive:
                camera.destroy()
            
            alive_actors = [actor for actor in actor_list if actor.is_alive]
            if alive_actors:
                client.apply_batch([carla.command.DestroyActor(x) for x in alive_actors])
            print('Actors destroyed. Exiting.')

if __name__ == '__main__':
    main()
