import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os


class ImagePublisherNode(Node):
    def __init__(self):
        super().__init__('image_publisher_node')

        # Declare a parameter for the folder path
        self.declare_parameter('image_folder', 'path/to/images')

        # Get the folder path from the parameter
        self.image_folder = self.get_parameter('image_folder').get_parameter_value().string_value
        self.get_logger().info(f"Using image folder: {self.image_folder}")

        # Validate the folder path
        if not os.path.exists(self.image_folder) or not os.path.isdir(self.image_folder):
            self.get_logger().error(f"Invalid folder path: {self.image_folder}")
            rclpy.shutdown()
            return

        # Get list of image files in the folder
        self.image_files = [f for f in os.listdir(self.image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.image_files.sort()  # Optional: sort alphabetically
        if not self.image_files:
            self.get_logger().error("No valid image files found in the folder.")
            rclpy.shutdown()
            return

        self.get_logger().info(f"Found {len(self.image_files)} images in the folder.")

        # Initialize CvBridge
        self.bridge = CvBridge()

        # Create a publisher for the images
        self.image_pub = self.create_publisher(Image, 'camera/image_raw', 10)

        # Initialize timer to publish images every 5 seconds
        self.timer = self.create_timer(0.5, self.timer_callback)

        # Initialize image index
        self.current_index = 0

    def timer_callback(self):
        if self.current_index >= len(self.image_files):
            # All images have been published; stop the timer and shut down the node
            self.get_logger().info("All images published. Shutting down...")
            self.timer.cancel()
            rclpy.shutdown()
            return

        # Get the next image file
        image_path = os.path.join(self.image_folder, self.image_files[self.current_index])
        self.get_logger().info(f"Publishing image: {image_path}")

        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            self.get_logger().error(f"Failed to read image: {image_path}")
            self.current_index += 1  # Skip to the next image
            return

        # Convert to ROS Image message
        ros_image = self.bridge.cv2_to_imgmsg(image, encoding='bgr8')

        # Publish the image
        self.image_pub.publish(ros_image)

        # Log confirmation
        self.get_logger().info(f"Image published successfully: {image_path}")

        # Move to the next image
        self.current_index += 1


def main(args=None):
    rclpy.init(args=args)
    node = ImagePublisherNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
