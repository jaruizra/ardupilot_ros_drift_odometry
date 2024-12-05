import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import numpy as np

class VisualOdometryNode(Node):
    def __init__(self):
        super().__init__('visual_odometry_node')

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

        # Initialize SIFT detector and descriptor
        self.sift = cv2.SIFT_create()

        # Initialize variables for previous frame data
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.prev_image = None

        # Camera intrinsic parameters (you need to replace these with your actual camera parameters)
        self.focal_length = 960.0  # Focal length in pixels
        self.principal_point = (960.0, 540.0)  # Principal point at the center

    def image_callback(self, msg):
        self.get_logger().info("Image received! Processing...")

        # Convert ROS Image message to OpenCV image
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect features and compute descriptors using SIFT
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)

        # Draw keypoints on the image
        annotated_img = cv2.drawKeypoints(
            img, keypoints, None, color=(255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT
        )

        # Save the annotated image
        self.save_image(annotated_img, suffix='_sift')

        # If previous descriptors exist, match features and estimate motion
        if self.prev_descriptors is not None and descriptors is not None:
            # Match descriptors between previous and current frame
            matches = self.match_features(self.prev_descriptors, descriptors)

            # Proceed if enough matches are found
            if len(matches) > 10:
                # Extract matched keypoints
                src_pts = np.float32([self.prev_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

                # Compute Essential Matrix
                E, mask = cv2.findEssentialMat(dst_pts, src_pts, focal=self.focal_length, pp=self.principal_point, method=cv2.RANSAC, prob=0.999, threshold=1.0)

                # Recover pose from Essential Matrix
                _, R, t, mask = cv2.recoverPose(E, dst_pts, src_pts, focal=self.focal_length, pp=self.principal_point)

                # Log rotation and translation vectors
                self.get_logger().info(f"Rotation matrix:\n{R}")
                self.get_logger().info(f"Translation vector:\n{t}")

                # Optionally, you can accumulate motion here to estimate the trajectory

                # Draw matches (optional)
                match_img = cv2.drawMatches(self.prev_image, self.prev_keypoints, img, keypoints, matches, None, flags=2)
                self.save_image(match_img, suffix='_matches')

            else:
                self.get_logger().warning("Not enough matches to compute motion.")

        else:
            self.get_logger().info("No previous descriptors to match with or descriptors are None.")

        # Update previous frame data
        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors
        self.prev_image = img

        # Publish the annotated image
        ros_image = self.bridge.cv2_to_imgmsg(annotated_img, encoding='bgr8')
        self.image_pub.publish(ros_image)
        self.get_logger().info("Annotated image published.")

        # Increment the image index
        self.image_index += 1

    def match_features(self, desc1, desc2):
        # Use BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(desc1, desc2, k=2)

        # Apply ratio test as per Lowe's paper
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        self.get_logger().info(f"Good matches found: {len(good_matches)}")
        return good_matches

    def save_image(self, image, suffix=''):
        # Save the image with a sequential name and optional suffix
        filename = f"{self.image_index}{suffix}.jpg"
        output_path = os.path.join(self.output_folder, filename)
        cv2.imwrite(output_path, image)
        self.get_logger().info(f"Image saved: {output_path}")

def main(args=None):
    rclpy.init(args=args)
    node = VisualOdometryNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
