import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import numpy as np
from scipy.spatial import KDTree


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

        # Initialize ORB detector with optimized parameters
        self.orb = cv2.ORB_create(
            nfeatures=15000, #number of keypoints to retain.
            scaleFactor=1.1, #how much the image is scaled down between pyramid levels.
            nlevels=30, #pyramid levels to use.
            edgeThreshold=15, #Size of the border where features are not detected.
            firstLevel=0, #Index of the first pyramid level.
            WTA_K=2, #Number of points that produce each element of the BRIEF descriptor.
            scoreType=cv2.ORB_HARRIS_SCORE, #Scoring method for keypoints
            patchSize=45, #Size of the area around each keypoint used for orientation computation.
            fastThreshold=5 #Threshold for the FAST feature detector.
        )

    def image_callback(self, msg):
        self.get_logger().info("Image received! Processing...")

        # Convert ROS Image message to OpenCV image
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Process the image with ORB + SIFT and edge preprocessing
        annotated_img, combined_keypoints, combined_descriptors = self.process_image(img)

        # Publish the annotated image
        ros_image = self.bridge.cv2_to_imgmsg(annotated_img, encoding='bgr8')
        self.image_pub.publish(ros_image)
        self.get_logger().info("Annotated image published.")

        # Increment the image index
        self.image_index += 1

    def process_image(self, img):
        # Convert image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Step 1: Preprocess with Laplacian to emphasize edges
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        enhanced_gray = cv2.convertScaleAbs(laplacian)

        # Step 2: Detect features using ORB
        keypoints_orb, descriptors_orb = self.orb.detectAndCompute(enhanced_gray, None)

        # Step 3: Detect features using SIFT
        sift = cv2.SIFT_create()
        keypoints_sift, descriptors_sift = sift.detectAndCompute(gray, None)

        # Step 4: Combine keypoints and descriptors
        combined_keypoints = keypoints_orb + keypoints_sift

        combined_descriptors = None
        if descriptors_orb is not None and descriptors_sift is not None:
            combined_descriptors = np.vstack([descriptors_orb, descriptors_sift])

        # Optional: Filter duplicates using KDTree
        unique_keypoints = self.filter_duplicate_keypoints(combined_keypoints)

        # Step 5: Draw combined keypoints on the image
        annotated_img = cv2.drawKeypoints(
            img, unique_keypoints, None, color=(255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT
        )

        # Log results
        self.get_logger().info(f"ORB keypoints: {len(keypoints_orb)}, SIFT keypoints: {len(keypoints_sift)}")
        self.get_logger().info(f"Combined keypoints: {len(combined_keypoints)}, Unique keypoints: {len(unique_keypoints)}")

        # Save annotated image
        self.save_image(annotated_img, suffix='_annotated_combined')

        return annotated_img, unique_keypoints, combined_descriptors

    def filter_duplicate_keypoints(self, keypoints):
        # Extract coordinates from keypoints
        points = np.array([kp.pt for kp in keypoints], dtype=np.float32)

        # Build a KDTree for spatial filtering
        tree = KDTree(points)
        unique_indices = []
        seen = set()

        for idx, pt in enumerate(points):
            if idx in seen:
                continue
            neighbors = tree.query_ball_point(pt, r=5)  # Radius threshold for duplicates
            unique_indices.append(idx)
            seen.update(neighbors)

        # Return unique keypoints
        return [keypoints[i] for i in unique_indices]

    def save_image(self, image, suffix=''):
        # Save the image with a sequential name and optional suffix
        filename = f"{self.image_index}{suffix}.jpg"
        output_path = os.path.join(self.output_folder, filename)
        cv2.imwrite(output_path, image)
        self.get_logger().info(f"Image saved: {output_path}")

    def save_keypoints(self, keypoints):
        # Extract keypoint coordinates
        points_2d = np.array([kp.pt for kp in keypoints], dtype=np.float32)

        # Assign z-coordinate (altitude)
        altitude = 60.0  # in meters
        z = np.full((points_2d.shape[0], 1), altitude)

        # Combine x, y, z into a single array
        points_3d = np.hstack((points_2d, z))

        # Save to a PLY file for point cloud visualization
        keypoints_path = os.path.join(self.output_folder, f"{self.image_index}_keypoints.ply")
        self.save_ply(points_3d, keypoints_path)
        self.get_logger().info(f"Keypoints saved: {keypoints_path}")

    def save_ply(self, points, filename):
        # Save points to a PLY file
        with open(filename, 'w') as f:
            f.write('ply\n')
            f.write('format ascii 1.0\n')
            f.write(f'element vertex {points.shape[0]}\n')
            f.write('property float x\n')
            f.write('property float y\n')
            f.write('property float z\n')
            f.write('end_header\n')
            for p in points:
                f.write(f"{p[0]} {p[1]} {p[2]}\n")


def main(args=None):
    rclpy.init(args=args)
    node = FeatureDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
