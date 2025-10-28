# CARLA Interactive Client

This Python script acts as a client to connect to a remote CARLA autonomous driving simulator. It allows a user to view the live simulation feed from a vehicle's camera and send real-time commands to control the vehicles within the simulation.

## What It Does

1.  **Connects to Server**: Establishes a socket connection to the CARLA server script.
2.  **Initial Setup**: Receives a list of vehicle IDs from the server and prompts the user to set an initial speed for each one.
3.  **Live Video Stream**: Opens a window (using OpenCV) to display the real-time camera feed from the currently selected vehicle.
4.  **Command Interface**: Provides a command-line interface in the terminal for sending control commands to the server.

## 1. Installation (Pip Requirements)

You will need Python 3 and the following libraries. You can install them using pip:

```bash
pip install opencv-python numpy
