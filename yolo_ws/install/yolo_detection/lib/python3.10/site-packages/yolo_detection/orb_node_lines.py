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

        # Initialize ORB detector with optimized parameters
        self.orb = cv2.ORB_create(
            nfeatures=2000,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=15,
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=45,
            fastThreshold=15
        )

    def image_callback(self, msg):
        self.get_logger().info("Image received! Processing...")

        # Convert ROS Image message to OpenCV image
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Process the image with ORB
        annotated_img, keypoints, descriptors = self.process_image(img)

        # Publish the annotated image
        ros_image = self.bridge.cv2_to_imgmsg(annotated_img, encoding='bgr8')
        self.image_pub.publish(ros_image)
        self.get_logger().info("Annotated image published.")

        # Increment the image index
        self.image_index += 1

    def process_image(self, img):
        # Convert image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)

        # Apply Canny edge detection
        edges = cv2.Canny(enhanced_gray, 50, 150)

        # Detect ORB keypoints
        keypoints, descriptors = self.orb.detectAndCompute(enhanced_gray, None)

        # Collect keypoints near edges to prioritize lines/patterns
        filtered_keypoints = []
        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            if edges[y, x] > 0:  # Keep keypoints that fall on strong edges
                filtered_keypoints.append(kp)

        # Draw filtered keypoints on the image
        annotated_img = cv2.drawKeypoints(
            img, filtered_keypoints, None, color=(255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT
        )

        # Cluster keypoints to identify linear patterns (e.g., roads, tree rows)
        if len(filtered_keypoints) > 1:
            points = np.array([kp.pt for kp in filtered_keypoints], dtype=np.float32)

            # DBSCAN clustering to group keypoints into patterns
            from sklearn.cluster import DBSCAN
            clustering = DBSCAN(eps=20, min_samples=5).fit(points)

            # Draw lines for clustered points
            for cluster_id in set(clustering.labels_):
                if cluster_id == -1:  # Ignore noise
                    continue
                cluster_points = points[clustering.labels_ == cluster_id]
                for i in range(len(cluster_points) - 1):
                    pt1 = tuple(cluster_points[i].astype(int))
                    pt2 = tuple(cluster_points[i + 1].astype(int))
                    cv2.line(annotated_img, pt1, pt2, (0, 255, 0), 2)  # Green for patterns

        # Log the number of filtered keypoints and clusters
        self.get_logger().info(f"Filtered keypoints: {len(filtered_keypoints)}")
        self.get_logger().info(f"Clusters detected: {len(set(clustering.labels_)) - 1}")  # Exclude noise

        # Save annotated image
        self.save_image(annotated_img, suffix='_annotated')

        return annotated_img, filtered_keypoints, descriptors



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
