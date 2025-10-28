#!/usr/bin/env python

# CARLA Interactive Client
#
# This script connects to the remote CARLA server to:
# 1. Receive vehicle IDs and prompt for initial speeds.
# 2. Display the live video stream in a window.
# 3. Provide a command-line interface to send control commands to the server.

import socket
import struct
import cv2
import numpy as np
import threading

# --- Shared State for Threads ---
shared_state = {
    'command_to_send': None,
    'lock': threading.Lock(),
    'simulation_running': True # Flag to sync threads
}

# --- User Input Thread ---
def user_input_thread():
    """A separate thread to handle blocking user input without freezing the video."""
    global shared_state
    
    print("\n" + "="*30)
    print("      COMMAND MENU (Client)")
    print("="*30)
    print("select <id>   - Focus on vehicle by ID")
    print("speed <km/h>  - Set target speed")
    print("lane <l/r>    - Force lane change")
    print("brake         - Apply full brakes")
    print("release       - Release manual brake")
    print("exit          - Quit simulation")
    print("="*30)

    # Loop now checks the global running flag
    while shared_state['simulation_running']:
        try:
            command = input("Enter command > ")
            if not shared_state['simulation_running']:
                break
            with shared_state['lock']:
                shared_state['command_to_send'] = command
            if command.lower().strip() == 'exit':
                # Signal the main thread to exit as well
                shared_state['simulation_running'] = False
                break
        except EOFError:
            shared_state['simulation_running'] = False
            break
    print("Input thread finished.")

def main():
    global shared_state
    # --- Configuration ---
    # IMPORTANT: Replace with your CARLA server's IP address
    SERVER_HOST = '172.28.251.74' 
    SERVER_PORT = 6969
    
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    input_thread = None # Initialize to None to prevent UnboundLocalError
    
    try:
        print(f"Connecting to {SERVER_HOST}:{SERVER_PORT}...")
        client_socket.connect((SERVER_HOST, SERVER_PORT))
        print("Connection successful!")
        
        # --- NEW: Initial Setup Phase ---
        print("Waiting to receive vehicle IDs from server...")
        id_data = client_socket.recv(1024).decode('utf-8')
        if id_data.startswith('ids:'):
            vehicle_ids_str = id_data.split(':')[1]
            vehicle_ids = [int(id_str) for id_str in vehicle_ids_str.split(',')]
            print(f"Server spawned vehicles with IDs: {vehicle_ids}")

            for veh_id in vehicle_ids:
                while True:
                    try:
                        speed = float(input(f"Enter initial speed (km/h) for Vehicle {veh_id}: "))
                        command = f"initspeed {veh_id} {speed}"
                        client_socket.sendall(command.encode('utf-8'))
                        break
                    except ValueError:
                        print("Invalid input. Please enter a number.")
            print("Initial setup complete. Starting video stream...")
        else:
            print("Error: Did not receive valid vehicle IDs from server.")
            return

        # --- Start the command input thread ---
        input_thread = threading.Thread(target=user_input_thread)
        input_thread.start()
        
        data = b""
        payload_size = struct.calcsize(">L")
        last_command_sent_for_display = "None"

        while shared_state['simulation_running']:
            # --- Check for and send commands ---
            command = None
            with shared_state['lock']:
                if shared_state['command_to_send']:
                    command = shared_state['command_to_send']
                    shared_state['command_to_send'] = None
            
            if command:
                last_command_sent_for_display = command
                client_socket.sendall(command.encode('utf-8'))
                if command.lower().strip() == 'exit':
                    print("Exit command sent. Closing client.")
                    break

            # --- Receive video frames (non-blocking) ---
            client_socket.settimeout(0.01)
            try:
                packet = client_socket.recv(4096)
                if not packet:
                    print("Server closed the connection.")
                    shared_state['simulation_running'] = False
                    break
                data += packet
            except socket.timeout:
                pass # This is expected if no new frame has arrived
            
            # Process buffer only if we have enough data for the size header
            if len(data) >= payload_size:
                packed_msg_size = data[:payload_size]
                data = data[payload_size:]
                msg_size = struct.unpack(">L", packed_msg_size)[0]

                if msg_size == 0:
                    continue 

                while len(data) < msg_size:
                    data += client_socket.recv(4096)

                frame_data = data[:msg_size]
                data = data[msg_size:]

                frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)
                
                if frame is not None:
                    cv2.putText(frame, f"Last Sent: {last_command_sent_for_display}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow('CARLA Stream', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                client_socket.sendall('exit'.encode('utf-8'))
                print("'q' pressed. Closing client.")
                shared_state['simulation_running'] = False
                break

    except ConnectionRefusedError:
        print(f"Connection refused. Is the server script running and port {SERVER_PORT} open?")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Closing connection...")
        shared_state['simulation_running'] = False # Ensure input thread exits
        if input_thread is not None:
            input_thread.join()
        client_socket.close()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
