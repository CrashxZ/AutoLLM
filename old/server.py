#!/usr/bin/env python

# CARLA Simulation Server
#
# This script runs the CARLA simulation, saves a video file, streams the
# video feed, and listens for remote commands. It now waits for initial
# speed commands from the client before starting the interactive loop.

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

# Adjust this path to your CARLA directory on the server
carla_dir = '/home/labsdr/carla_simulator'

try:
    sys.path.append(glob.glob(f'{carla_dir}/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    print(f"Error: CARLA egg file not found in '{carla_dir}/PythonAPI/carla/dist/'. Please check the 'carla_dir' variable.")
    sys.exit()

import carla
sys.path.append(f'{carla_dir}/PythonAPI/carla')
from agents.navigation.global_route_planner import GlobalRoutePlanner

# --- Shared State for Threads ---
shared_state = {
    'selected_vehicle': None,
    'command': None,
    'simulation_running': True,
    'manual_control_vehicles': set(),
    'new_frame': None,
    'lock': threading.Lock()
}

# --- Command Receiving Thread ---
def command_receiving_thread(conn):
    """Listens for commands from the connected client."""
    global shared_state
    while shared_state['simulation_running']:
        try:
            command_data = conn.recv(1024)
            if not command_data:
                break
            command = command_data.decode('utf-8')
            with shared_state['lock']:
                shared_state['command'] = command.lower().strip()
        except (ConnectionResetError, BrokenPipeError):
            print("[COMMAND] Client disconnected.")
            shared_state['simulation_running'] = False
            break
        except Exception:
            shared_state['simulation_running'] = False
            break
    print("[COMMAND] Command receiving thread stopped.")

# --- Video Streaming Thread ---
def video_streaming_thread(conn, addr):
    """Streams video frames to the client."""
    global shared_state
    print(f"[STREAM] Connection from: {addr}")
    
    # Start a thread to listen for commands from this client
    cmd_thread = threading.Thread(target=command_receiving_thread, args=(conn,))
    cmd_thread.start()

    while shared_state['simulation_running']:
        frame = None
        with shared_state['lock']:
            if shared_state['new_frame'] is not None:
                frame = shared_state['new_frame']
        
        if frame is not None:
            try:
                result, encoded_frame = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                if result:
                    data = encoded_frame.tobytes()
                    size = len(data)
                    conn.sendall(struct.pack('>L', size) + data)
            except (ConnectionResetError, BrokenPipeError):
                print("[STREAM] Client disconnected during streaming.")
                shared_state['simulation_running'] = False
                break
            except Exception as e:
                print(f"[STREAM] Streaming Error: {e}")
                shared_state['simulation_running'] = False
                break
        
        time.sleep(1/30) # Limit streaming FPS
        
    print("[STREAM] Streaming logic stopped.")

def main():
    global shared_state
    actor_list = []
    vehicles_list = []
    client = None
    video_writer = None
    camera = None
    tm = None
    
    server_socket = None
    conn = None

    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.load_world('Town04_Opt', map_layers=carla.MapLayer.Ground)
        spectator = world.get_spectator()
        
        tm = client.get_trafficmanager(8005) # Use a non-default TM port
        tm.set_global_distance_to_leading_vehicle(2.5)
        tm.set_respawn_dormant_vehicles(True)
        
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)
        tm.set_synchronous_mode(True)
        
        blueprint_library = world.get_blueprint_library()
        spawn_points = world.get_map().get_spawn_points()
        
        vehicle_bp1 = blueprint_library.find('vehicle.audi.tt')
        vehicle_bp1.set_attribute('color', '255,0,0')
        vehicle1 = world.spawn_actor(vehicle_bp1, spawn_points[100])
        vehicles_list.append(vehicle1)
        actor_list.append(vehicle1)

        vehicle_bp2 = blueprint_library.find('vehicle.audi.etron')
        vehicle_bp2.set_attribute('color', '0,0,255')
        vehicle2 = world.spawn_actor(vehicle_bp2, spawn_points[102])
        vehicles_list.append(vehicle2)
        actor_list.append(vehicle2)
        
        print(f"Spawned Vehicle {vehicle1.id} (Red) and Vehicle {vehicle2.id} (Blue)")

        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '1280')
        camera_bp.set_attribute('image_size_y', '720')
        camera = world.spawn_actor(camera_bp, carla.Transform(), attach_to=spectator)
        actor_list.append(camera)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter('output.mp4', fourcc, 20.0, (1280, 720))

        def process_camera_image(image):
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = np.reshape(array, (image.height, image.width, 4))
            bgr_image = array[:, :, :3]

            if video_writer:
                video_writer.write(bgr_image)
            with shared_state['lock']:
                shared_state['new_frame'] = bgr_image

        camera.listen(process_camera_image)

        # --- Wait for client connection and perform initial setup ---
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind(('0.0.0.0', 6969))
        server_socket.listen(1)
        print("[SETUP] Waiting for client to connect for initial setup...")
        conn, addr = server_socket.accept()
        
        # Send vehicle IDs to the client
        id_str = "ids:" + ",".join([str(v.id) for v in vehicles_list])
        conn.sendall(id_str.encode('utf-8'))
        print(f"[SETUP] Sent vehicle IDs to client: {id_str}")

        # Wait for initial speed commands from the client
        configured_vehicles = set()
        while len(configured_vehicles) < len(vehicles_list):
            data = conn.recv(1024)
            if not data:
                raise ConnectionError("Client disconnected during setup.")
            command = data.decode('utf-8').lower().strip()
            cmd_parts = command.split()
            if len(cmd_parts) == 3 and cmd_parts[0] == 'initspeed':
                try:
                    veh_id = int(cmd_parts[1])
                    speed = float(cmd_parts[2])
                    for v in vehicles_list:
                        if v.id == veh_id:
                            tm.set_desired_speed(v, speed)
                            print(f"[SETUP] Set initial speed for Vehicle {veh_id} to {speed} km/h")
                            configured_vehicles.add(veh_id)
                            break
                except (ValueError, IndexError):
                    print(f"[SETUP] Received invalid initial speed command: {command}")
        
        print("[SETUP] All vehicles configured. Starting simulation.")
        
        # --- Start Thread for streaming and command listening ---
        stream_thread = threading.Thread(target=video_streaming_thread, args=(conn, addr))
        stream_thread.start()
        
        grp = GlobalRoutePlanner(world.get_map(), 1.0)
        route = grp.trace_route(spawn_points[100].location, spawn_points[150].location)
        path_locations = [rt[0].transform.location for rt in route]
        for vehicle in vehicles_list:
            tm.set_path(vehicle, path_locations)
            vehicle.set_autopilot(True, tm.get_port())

        with shared_state['lock']:
            shared_state['selected_vehicle'] = vehicle1
        
        while shared_state['simulation_running']:
            world.tick()
            
            with shared_state['lock']:
                selected_v = shared_state['selected_vehicle']
                
            focus_transform = spectator.get_transform() 
            if selected_v:
                vehicle_loc = selected_v.get_location()
                focus_transform = carla.Transform(
                    vehicle_loc + carla.Location(z=70),
                    carla.Rotation(pitch=-90)
                )
            spectator.set_transform(focus_transform)
            
            with shared_state['lock']:
                command = shared_state.get('command')
                if command:
                    shared_state['command'] = None # Consume command

            if command and not command.startswith('initspeed'):
                cmd_parts = command.split()
                cmd = cmd_parts[0]
                
                with shared_state['lock']:
                    selected_v = shared_state['selected_vehicle']

                if cmd == 'exit':
                    shared_state['simulation_running'] = False
                elif cmd == 'select' and len(cmd_parts) > 1:
                    try:
                        veh_id = int(cmd_parts[1])
                        found = False
                        for v in vehicles_list:
                            if v.id == veh_id:
                                with shared_state['lock']:
                                    shared_state['selected_vehicle'] = v
                                found = True
                                break
                    except ValueError: pass
                
                elif selected_v: # Commands that require a selected vehicle
                    if cmd == 'brake':
                        with shared_state['lock']:
                            shared_state['manual_control_vehicles'].add(selected_v.id)
                    elif cmd == 'release':
                        with shared_state['lock']:
                            shared_state['manual_control_vehicles'].discard(selected_v.id)
                        selected_v.set_autopilot(True, tm.get_port())
                    elif cmd == 'speed' and len(cmd_parts) > 1:
                        try:
                            speed = float(cmd_parts[1])
                            tm.set_desired_speed(selected_v, speed)
                        except ValueError: pass
                    elif cmd == 'lane' and len(cmd_parts) > 1:
                        direction = cmd_parts[1]
                        if direction == 'l': tm.force_lane_change(selected_v, False)
                        elif direction == 'r': tm.force_lane_change(selected_v, True)

            with shared_state['lock']:
                manual_ids = list(shared_state['manual_control_vehicles'])
            for v_id in manual_ids:
                for v in vehicles_list:
                    if v.id == v_id:
                        v.apply_control(carla.VehicleControl(brake=1.0))
                        break

    finally:
        shared_state['simulation_running'] = False
        if conn: conn.close()
        if server_socket: server_socket.close()
        if video_writer:
            video_writer.release()
            print("Video file 'output.mp4' saved.")
        if client:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)
            if tm:
                tm.set_synchronous_mode(False)
            print('\nDestroying actors...')
            client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
            print('Actors destroyed.')

if __name__ == '__main__':
    main()
