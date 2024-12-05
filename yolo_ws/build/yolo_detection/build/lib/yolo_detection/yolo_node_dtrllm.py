import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from pathlib import Path
import cv2
import torch
from transformers import AutoModelForObjectDetection, AutoImageProcessor
import supervision as sv
import os
import time


class YoloNode(Node):
    def __init__(self):
        super().__init__('yolo_node')

        # Initialize CvBridge
        self.bridge = CvBridge()

        # ROS publishers and subscribers
        self.image_pub = self.create_publisher(Image, 'annotated_roof_image', 10)
        self.image_sub = self.create_subscription(Image, 'camera/image_raw', self.image_callback, 10)

        # Load detector and processor
        self.get_logger().info('Loading RT-DETR model for roof detection...')
        self.detector = AutoModelForObjectDetection.from_pretrained(
            "Yifeng-Liu/rt-detr-finetuned-for-satellite-image-roofs-detection"
        )
        self.processor = AutoImageProcessor.from_pretrained(
            "Yifeng-Liu/rt-detr-finetuned-for-satellite-image-roofs-detection"
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.detector.to(self.device)

        # Confidence threshold for predictions
        self.confidence_threshold = 0.5

        # Initialize annotator
        self.mask_annotator = sv.MaskAnnotator()

        # Output folder for saving annotated images
        self.output_folder = "output_images"
        os.makedirs(self.output_folder, exist_ok=True)

        # Initialize image index for sequential naming
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
        self.get_logger().info("Image received! Processing...")

        # Log time since the last image was processed
        if self.last_image_time is not None:
            time_since_last_image = start_time - self.last_image_time
            self.get_logger().info(f"Time since last image processed: {time_since_last_image:.3f} seconds.")

        # Convert ROS Image message to OpenCV image
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Process the image for roof detection
        annotated_image = self.process_image(image)

        # Save the annotated image
        self.save_image(annotated_image)

        # Publish the annotated image
        ros_image = self.bridge.cv2_to_imgmsg(annotated_image, encoding='bgr8')
        self.image_pub.publish(ros_image)
        self.get_logger().info("Processed image published.")

        # Record end time and calculate processing duration
        end_time = time.time()
        processing_time = end_time - start_time
        self.get_logger().info(f"Processing time for this image: {processing_time:.3f} seconds.")

        # Reset the flag to continue waiting for images
        self.image_received = False

        # Update last image time
        self.last_image_time = end_time

    def process_image(self, image):
        # Preprocess the image for the detector
        inputs = self.processor(images=image, return_tensors='pt').to(self.device)

        # Perform inference
        with torch.no_grad():
            outputs = self.detector(**inputs)

        # Post-process detections
        target_sizes = torch.tensor([image.shape[:2]]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs=outputs,
            threshold=self.confidence_threshold,
            target_sizes=target_sizes
        )[0]

        # Annotate the image with detection results
        return self.annotate_image(image, results)

    def annotate_image(self, image, results):
        # Annotate bounding boxes
        for score, box in zip(results["scores"], results["boxes"]):
            if score >= self.confidence_threshold:
                box = [round(coord) for coord in box.tolist()]
                cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                label = f"Roof: {score:.2f}"
                cv2.putText(image, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return image

    def save_image(self, image):
        # Save the annotated image with a sequential name
        output_path = os.path.join(self.output_folder, f"{self.image_index}.jpg")
        cv2.imwrite(output_path, image)
        self.get_logger().info(f"Image saved: {output_path}")

        # Increment the image index
        self.image_index += 1


def main(args=None):
    rclpy.init(args=args)
    node = YoloNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
