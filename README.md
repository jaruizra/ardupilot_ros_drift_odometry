# ROS Visual Odometry

## Overview
This project implements visual odometry using ROS (Robot Operating System). Visual odometry determines the position and orientation of a robot by analyzing input from its camera sensors. This enables the robot to navigate and understand its environment without relying on external systems like GPS.

## Project Structure
The project is organized as follows:

- `src/`: Contains the source code for the ROS nodes.
- `launch/`: Includes launch files to start the ROS nodes.
- `config/`: Holds configuration files such as camera parameters.
- `CMakeLists.txt`: Build configuration file for the ROS package.
- `package.xml`: Package manifest file with project metadata and dependencies.

## How It Works
The core of the project is a ROS node that performs visual odometry by processing image data from a camera. The node follows these main steps:

1. **Image Acquisition**: Subscribes to a camera topic to receive image data in real-time.
2. **Feature Extraction**: Detects and extracts key features from each image frame.
3. **Feature Matching**: Matches features between consecutive frames to find correspondences.
4. **Motion Estimation**: Estimates the relative motion between frames based on the matched features.
5. **Pose Update**: Updates the robot's pose by integrating the estimated motion over time.
6. **Publishing Results**: Publishes the estimated pose to a ROS topic for use by other nodes.

## File Descriptions
The `yolo_detection` directory contains all the code related to the node:
- `src/yolo_node_og.py`: Initial test using the YOLOv8 vision model. The YOLOv8 models and the YOLOv8-buildings model (a fine-tuned model to detect roofs) are from Hugging Face: https://huggingface.co/keremberke/yolov8n-building-segmentation
- `src/yolo_detection/yolo_detection/yolo_node_dtrllm.py`: An improved version of the YOLOv8 model for roof detection, which requires more compute time.
- `src/yolo_detection/yolo_detection/orb_node.py`: Switches strategy to point cloud using the ORB algorithm for feature detection.
- `src/yolo_detection/yolo_detection/swift.py`: An improvement over the ORB node, using the Swift detector for better performance with similar compute requirements.
- `src/yolo_detection/yolo_detection/swift_detect.py`: Initial attempt at feature matching across frames, which did not work as expected.
- `src/yolo_detection/yolo_detection/image_publiser_node.py`: Publishes images from a folder to the `/camera/image/raw` topic, simulating a camera for the ROS2 visual odometry node.
- `src/yolo_detection/setup.py`: Sets up the ROS2 package.
- `src/yolo_detection/package.xml`: Defines the package information, dependencies, and other metadata required by ROS.
- `download_model.py`: A simple Python script to download models.

## Installation

### Prerequisites
- **ROS**: Install the appropriate version of ROS for your system (e.g., ROS Noetic).
- **OpenCV**: Ensure OpenCV is installed for image processing capabilities.
- **C++ Compiler**: A compiler that supports C++11 or later.

### Building the Package
1. Clone the repository:
2. Navigate to your ROS workspace: `cd ~/yolo_ws`
3. Build the workspace: `colcon build`
4. Source the workspace: `source install/setup.bash`

## Usage
Refer to `instructions.txt` for ROS commands to launch the nodes and instructions on building and sourcing the workspace.

## ROS Nodes

### Node: visual_odometry_node

#### Subscribed Topics:
- **/camera/image_raw (sensor_msgs/Image)**: Receives raw image data from the camera.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.