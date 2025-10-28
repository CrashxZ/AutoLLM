#!/usr/bin/env python

# CARLA Multi-Vehicle Script with Labels
#
# This script loads a layered map, spawns two vehicles near each other,
# sets them both to autopilot, and follows the first vehicle with a
# top-down camera view. It also draws text labels above each vehicle.

import glob
import os
import sys
import random
import time
import math

try:
    # Find the CARLA egg file and add it to the Python path
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    print("Error: CARLA egg file not found. Make sure you are in the 'PythonAPI/examples' directory.")
    sys.exit()

import carla

# List to keep track of actors spawned in the simulation
actor_list = []

def find_nearby_spawn_point(start_point, all_spawn_points, max_dist=15.0):
    """
    Finds a spawn point from the list that is close to the start_point.
    """
    for spawn_point in all_spawn_points:
        # Calculate distance between the two points
        dist = start_point.location.distance(spawn_point.location)
        if dist < max_dist and dist > 1.0: # Ensure it's not the same point
            return spawn_point
    return None # Return None if no suitable point is found

def main():
    """
    Main function to connect to CARLA, load the map, spawn actors, and run.
    """
    try:
        # 1. Connect to the CARLA server
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)

        # 2. Load the optimized map with only the ground layer
        map_name = 'Town04_Opt'
        layers_to_load = carla.MapLayer.Ground
        
        print(f"Loading map '{map_name}' with layers: {layers_to_load}...")
        world = client.load_world(map_name, map_layers=layers_to_load)
        print("Map loaded successfully.")

        blueprint_library = world.get_blueprint_library()
        
        # 3. Define vehicle blueprints
        vehicle_bp1 = blueprint_library.find('vehicle.tesla.model3')
        vehicle_bp1.set_attribute('color', '255,0,0') # Red

        vehicle_bp2 = blueprint_library.find('vehicle.audi.etron')
        vehicle_bp2.set_attribute('color', '0,0,255') # Blue

        # 4. Find spawn points for two vehicles near each other
        all_spawn_points = world.get_map().get_spawn_points()
        spawn_point1 = random.choice(all_spawn_points)
        spawn_point2 = find_nearby_spawn_point(spawn_point1, all_spawn_points)

        if spawn_point2 is None:
            print("Warning: Could not find a nearby spawn point. Spawning at a random location.")
            spawn_point2 = random.choice(all_spawn_points)

        # 5. Spawn the vehicles
        vehicle1 = world.spawn_actor(vehicle_bp1, spawn_point1)
        actor_list.append(vehicle1)
        print(f'Spawned Vehicle 1: {vehicle1.type_id} at {spawn_point1.location}')

        vehicle2 = world.spawn_actor(vehicle_bp2, spawn_point2)
        actor_list.append(vehicle2)
        print(f'Spawned Vehicle 2: {vehicle2.type_id} at {spawn_point2.location}')

        # 6. Set both vehicles to autopilot
        vehicle1.set_autopilot(True)
        vehicle2.set_autopilot(True)
        print("Both vehicles are now on autopilot.")

        # 7. Set up the spectator camera for a top-down view
        spectator = world.get_spectator()
        
        print("\nPress Ctrl+C to exit.")
        
        # Main loop to keep the camera and labels following the vehicles
        while True:
            # --- Update spectator camera ---
            vehicle1_location = vehicle1.get_location()
            spectator.set_transform(carla.Transform(
                vehicle1_location + carla.Location(z=70),
                carla.Rotation(pitch=-90)
            ))
            
            # --- Draw labels above the vehicles ---
            # The life_time parameter makes the text disappear after a short time,
            # so we must redraw it in every frame to make it follow the vehicle.
            world.debug.draw_string(vehicle1.get_location() + carla.Location(z=2),
                                      f'V0',
                                      draw_shadow=False,
                                      color=carla.Color(r=255, g=0, b=0),
                                      life_time=0.03, # Text lives for ~1 frame
                                      persistent_lines=True)

            world.debug.draw_string(vehicle2.get_location() + carla.Location(z=2),
                                      f'V1',
                                      draw_shadow=False,
                                      color=carla.Color(r=0, g=0, b=255),
                                      life_time=0.03,
                                      persistent_lines=True)

            time.sleep(0.02)

    finally:
        # Clean up all spawned actors before exiting
        print('\nDestroying actors and cleaning up...')
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        print('Actors destroyed. Exiting.')

if __name__ == '__main__':
    main()
