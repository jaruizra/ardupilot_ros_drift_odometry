import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import time
from ultralytics import YOLO


class YOLODetectionNode(Node):
    def __init__(self):
        super().__init__('yolo_detection_node')

        # Initialize the CV Bridge
        self.bridge = CvBridge()

        # Create publisher
        self.image_pub = self.create_publisher(Image, 'yolo_annotated_image', 10)

        # Create subscriber to receive images from a camera topic
        self.image_sub = self.create_subscription(Image, 'camera/image_raw', self.image_callback, 10)

        # Load YOLO model
        self.get_logger().info('Loading YOLO model...')
        self.model = YOLO('yolov8n-buildings.pt')  # My model

        # Create output folder for saving annotated images
        self.output_folder = "output_images"
        os.makedirs(self.output_folder, exist_ok=True)

        # Initialize the image index for sequential naming
        self.image_index = 1

        # Timer to log "waiting for images" every second
        self.timer = self.create_timer(1.0, self.waiting_callback)
        self.image_received = False

        # Initialize timestamps for measuring processing time
        self.last_image_time = None

    def waiting_callback(self):
        if not self.image_received:
            self.get_logger().info("Waiting for images on 'camera/image_raw'...")

    def image_callback(self, msg):
        # Record start time
        start_time = time.time()

        self.image_received = True
        self.get_logger().info(" ")
        self.get_logger().info("Image received! Processing...")

        # Log time since the last image was processed
        if self.last_image_time is not None:
            time_since_last_image = start_time - self.last_image_time
            self.get_logger().info(f"Time since last image processed: {time_since_last_image:.3f} seconds.")

        # Convert ROS Image message to OpenCV image
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Process the image with YOLO
        annotated_img = self.process_image(img)

        # Save the annotated image
        self.save_image(annotated_img)

        # Publish the annotated image
        ros_image = self.bridge.cv2_to_imgmsg(annotated_img, encoding='bgr8')
        self.image_pub.publish(ros_image)
        self.get_logger().info("Annotated image published.")

        # Record end time and calculate processing duration
        end_time = time.time()
        processing_time = end_time - start_time
        self.get_logger().info(f"Processing time for this image: {processing_time:.3f} seconds.")

        # Reset the flag to continue waiting for images
        self.image_received = False

        # Update last image time
        self.last_image_time = end_time
    
    def process_image(self, img):
        # Run YOLO inference
        results = self.model(img)
        annotated_img = img.copy()

        # Initialize sums
        sum_x = 0
        sum_y = 0
        box_count = 0

        # Annotate buildings in the image and calculate center sums
        for result in results:
            for box in result.boxes:  # Iterate over detected boxes
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates

                # Calculate the center of the bounding box
                x_center = (x1 + x2) // 2
                y_center = (y1 + y2) // 2

                # Add to the sums
                sum_x += x_center
                sum_y += y_center
                box_count += 1

                # Draw the bounding box
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (255, 0, 0), 2)

                # Annotate with label and center coordinates
                label = f"Building: {float(box.conf[0]):.2f}"
                center_label = f"Center: ({x_center}, {y_center})"
                cv2.putText(annotated_img, label, (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.putText(annotated_img, center_label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(annotated_img, (x_center, y_center), 5, (0, 255, 0), -1)

        # Output summed position
        if box_count > 0:
            self.get_logger().info(f"Summed center position: ({sum_x}, {sum_y})")
        else:
            self.get_logger().warning("No bounding boxes detected. Summed center position: (0, 0)")

        return annotated_img

    def save_image(self, image):
        # Save the annotated image with a sequential name
        output_path = os.path.join(self.output_folder, f"{self.image_index}.jpg")
        cv2.imwrite(output_path, image)
        self.get_logger().info(f"Image saved: {output_path}")

        # Increment the image index
        self.image_index += 1


def main(args=None):
    rclpy.init(args=args)
    node = YOLODetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
