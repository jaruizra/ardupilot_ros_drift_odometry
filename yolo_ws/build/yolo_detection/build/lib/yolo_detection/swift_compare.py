import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import numpy as np


class FeatureDetectionNode(Node):
    def __init__(self):
        super().__init__('feature_detection_node')

        # Initialize the CV Bridge
        self.bridge = CvBridge()

        # Create publisher
        self.image_pub = self.create_publisher(Image, 'feature_annotated_image', 10)

        # Create subscriber to receive images from a camera topic
        self.image_sub = self.create_subscription(Image, 'camera/image_raw', self.image_callback, 10)

        # Create output folder for saving annotated images
        self.output_folder = "output_images"
        os.makedirs(self.output_folder, exist_ok=True)

        # Initialize the image index for sequential naming
        self.image_index = 1

        # Initialize SIFT detector with default settings
        self.sift_default = cv2.SIFT_create()

        # SIFT with adjusted parameters for finer detection
        self.sift_fine = cv2.SIFT_create(
            contrastThreshold=0.02,
            edgeThreshold=10,
            nOctaveLayers=5
        )

        # Initialize previous frame and keypoints for tracking
        self.prev_frame = None
        self.prev_keypoints = None

    def image_callback(self, msg):
        self.get_logger().info("Image received! Processing...")

        # Convert ROS Image message to OpenCV image
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Process the image with different techniques
        results = self.process_image(img)

        # Publish the annotated image (default SIFT as an example)
        ros_image = self.bridge.cv2_to_imgmsg(results['sift_default'], encoding='bgr8')
        self.image_pub.publish(ros_image)
        self.get_logger().info("Annotated image published.")

        # Increment the image index
        self.image_index += 1

    def process_image(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Store results for comparison
        results = {}

        # Technique 1: SIFT with default settings
        annotated_img_default, _, _ = self.detect_and_annotate(gray, self.sift_default)
        results['sift_default'] = annotated_img_default
        self.save_image(annotated_img_default, suffix='_sift_default')

        # Technique 2: SIFT with fine-tuned parameters
        annotated_img_fine, _, _ = self.detect_and_annotate(gray, self.sift_fine)
        results['sift_fine'] = annotated_img_fine
        self.save_image(annotated_img_fine, suffix='_sift_fine')

        # Technique 3: Keypoint tracking with optical flow
        annotated_img_tracked = self.track_keypoints(gray)
        results['keypoint_tracking'] = annotated_img_tracked
        self.save_image(annotated_img_tracked, suffix='_keypoint_tracking')

        return results

    def detect_and_annotate(self, gray, detector):
        # Enhance contrast with CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)

        # Detect features and compute descriptors
        keypoints, descriptors = detector.detectAndCompute(enhanced_gray, None)

        # Optionally filter top keypoints (retain more to improve results)
        if len(keypoints) > 2000:
            keypoints = sorted(keypoints, key=lambda x: x.response, reverse=True)[:2000]

        # Draw keypoints
        annotated_img = cv2.drawKeypoints(
            enhanced_gray, keypoints, None, color=(255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT
        )

        # Log results
        self.get_logger().info(f"Keypoints detected: {len(keypoints)}")

        return annotated_img, keypoints, descriptors


    def track_keypoints(self, gray):
        # Initialize keypoints on the first frame
        if self.prev_frame is None:
            self.prev_frame = gray
            self.prev_keypoints = []  # Initialize empty keypoints
            return gray

        # If no previous keypoints, reinitialize with SIFT
        if not self.prev_keypoints:
            keypoints, _ = self.sift_default.detectAndCompute(gray, None)
            self.prev_keypoints = keypoints
            self.prev_frame = gray
            return gray

        # Track keypoints using optical flow
        p0 = np.float32([kp.pt for kp in self.prev_keypoints]).reshape(-1, 1, 2)
        p1, st, _ = cv2.calcOpticalFlowPyrLK(self.prev_frame, gray, p0, None)

        # Filter valid tracked points
        tracked_keypoints = [
            cv2.KeyPoint(p[0][0], p[0][1], kp.size)
            for p, kp, valid in zip(p1, self.prev_keypoints, st) if valid
        ]

        # Update previous frame and keypoints
        self.prev_frame = gray
        self.prev_keypoints = tracked_keypoints

        # Annotate tracked keypoints
        annotated_img = cv2.drawKeypoints(
            gray, tracked_keypoints, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT
        )

        self.get_logger().info(f"Tracked keypoints: {len(tracked_keypoints)}")

        return annotated_img


    def save_image(self, image, suffix=''):
        # Save the image with a sequential name and optional suffix
        filename = f"{self.image_index}{suffix}.jpg"
        output_path = os.path.join(self.output_folder, filename)
        cv2.imwrite(output_path, image)
        self.get_logger().info(f"Image saved: {output_path}")


def main(args=None):
    rclpy.init(args=args)
    node = FeatureDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
