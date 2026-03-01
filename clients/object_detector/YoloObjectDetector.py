import cv2
import os
import torch
from pathlib import Path
from ultralytics import YOLO

class YoloObjectDetector:
    """Ultra-Advanced Object Detection with support for latest YOLO models"""

    def __init__(
        self, device, model_name="yolo12x.pt", confidence=0.5, verbose=True
    ):
        """
        Initialize ultra-advanced detector

        Args:
            model_name: YOLO model name (see MODEL_INFO for options)
            confidence: Confidence threshold
            device: Device to use ('auto', 'cpu', 'cuda', 'mps')
            verbose: Print detailed information
        """
        self.model_name = model_name
        self.confidence = confidence
        self.device = device
        # Load model with error handling
        self.load_model()

    def _load_yolo_safe(self, model_path):
        """Load YOLO model with PyTorch 2.8+ meta tensor workaround."""
        # PyTorch 2.8+ uses lazy init with meta tensors by default.
        # Force CPU device context so all tensors are materialized on CPU,
        # then explicitly move to the target device (GPU if available).
        with torch.device('cpu'):
            model = YOLO(model_path)
        # Move model to target device (e.g., 'cuda' for RTX 5090) at init time
        # so it doesn't need to be transferred on every inference call
        if self.device and self.device != "cpu":
            model.to(self.device)
        return model

    def load_model(self):
        """Load the specified YOLO model with fallback options"""
        try:
            self.model = self._load_yolo_safe(f"models/yolo12x-object-detector/{self.model_name}")
            self.class_names = self.model.names
        except Exception as e:
            print(f"Failed to load {self.model_name}: {e}")
            fallback_models = [
                "yolo12x.pt",
                "yolo12l.pt",
                "yolo12m.pt",
                "yolo12s.pt",
                "yolo12n.pt",
            ]
            for fallback in fallback_models:
                try:
                    print(f"Trying fallback model: {fallback}")
                    self.model = self._load_yolo_safe(f"models/yolo12x-object-detector/{fallback}")
                    self.class_names = self.model.names
                    self.model_name = fallback
                    print(f"Fallback model loaded successfully!")
                    break
                except:
                    continue
            else:
                raise RuntimeError("Failed to load any YOLO model")

    def detect_object_from_image(self, input_image_path, output_directory_path):
        """
        Detect objects in a single image with advanced features
        """
        cropped_object_images = []
        if not os.path.exists(input_image_path):
            print(f"Image not found: {input_image_path}")
            return cropped_object_images

        # Validate that the image can be read
        test_image = cv2.imread(input_image_path)
        if test_image is None:
            print(f"Cannot read image file (possibly corrupted or unsupported format): {input_image_path}")
            return cropped_object_images

        if not os.path.exists(output_directory_path):
            os.makedirs(output_directory_path, exist_ok=True)

        try:
            # Run detection
            results = self.model(input_image_path, conf=self.confidence, device=self.device)
        except Exception as e:
            print(f"Error during object detection for {input_image_path}: {str(e)}")
            return cropped_object_images

        # Use the already validated image
        image = test_image.copy()

        # Create output directory
        output_folder = os.path.join(output_directory_path, "detected_objects")
        os.makedirs(output_folder, exist_ok=True)

        # Process results
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)

                for i, (box, conf, cls_id) in enumerate(
                    zip(boxes, confidences, class_ids)
                ):
                    x1, y1, x2, y2 = box.astype(int)
                    class_name = self.class_names[cls_id]

                    if class_name != "person":
                        # Draw bounding box
                        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        # Draw label with background
                        label = f"{class_name}: {conf:.2f}"
                        (label_width, label_height), _ = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                        )
                        cv2.rectangle(
                            image,
                            (x1, y1 - label_height - 10),
                            (x1 + label_width, y1),
                            (0, 255, 0),
                            -1,
                        )
                        cv2.putText(
                            image,
                            label,
                            (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            2,
                        )
                        base_filename = os.path.splitext(
                            os.path.basename(input_image_path)
                        )[0]
                        # Saving the detected cropped images
                        crop = image[y1:y2, x1:x2]
                        extension = Path(input_image_path).suffix
                        cropped_object_image_path = f"{output_folder}/{base_filename}_{class_name}_{i}{extension}"
                        cv2.imwrite(cropped_object_image_path, crop)
                        cropped_object_images.append(
                            {
                                "cropped_object_image_path": cropped_object_image_path,
                                "class_name": class_name,
                            }
                        )

        return cropped_object_images

    def detect_object_from_video(self, input_video_path, output_directory_path, frame_interval=30):
        """
        Detect objects in a video with advanced features
        
        Args:
            input_video_path (str): Path to input video
            output_directory_path (str): Directory for saving detected objects
            frame_interval (int): Process every Nth frame (default: 30)
            
        Returns:
            list: List of detected object information with paths and class names
        """
        cropped_object_images = []
        if not os.path.exists(input_video_path):
            print(f"Video not found: {input_video_path}")
            return cropped_object_images

        # Validate that the video can be read
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            print(f"Cannot read video file (possibly corrupted or unsupported format): {input_video_path}")
            return cropped_object_images

        if not os.path.exists(output_directory_path):
            os.makedirs(output_directory_path, exist_ok=True)

        # Create output directory
        output_folder = os.path.join(output_directory_path, "detected_objects")
        os.makedirs(output_folder, exist_ok=True)

        base_filename = os.path.splitext(os.path.basename(input_video_path))[0]
        frame_count = 0
        object_counter = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Process every frame_interval frames
                if frame_count % frame_interval == 0:
                    try:
                        # Run detection on current frame
                        results = self.model(frame, conf=self.confidence, device=self.device)
                    except Exception as e:
                        print(f"Error during object detection for frame {frame_count}: {str(e)}")
                        frame_count += 1
                        continue

                    # Process results
                    for result in results:
                        if result.boxes is not None:
                            boxes = result.boxes.xyxy.cpu().numpy()
                            confidences = result.boxes.conf.cpu().numpy()
                            class_ids = result.boxes.cls.cpu().numpy().astype(int)

                            for i, (box, conf, cls_id) in enumerate(
                                zip(boxes, confidences, class_ids)
                            ):
                                x1, y1, x2, y2 = box.astype(int)
                                class_name = self.class_names[cls_id]

                                if class_name != "person":
                                    # Draw bounding box
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                                    # Draw label with background
                                    label = f"{class_name}: {conf:.2f}"
                                    (label_width, label_height), _ = cv2.getTextSize(
                                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                                    )
                                    cv2.rectangle(
                                        frame,
                                        (x1, y1 - label_height - 10),
                                        (x1 + label_width, y1),
                                        (0, 255, 0),
                                        -1,
                                    )
                                    cv2.putText(
                                        frame,
                                        label,
                                        (x1, y1 - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.6,
                                        (255, 255, 255),
                                        2,
                                    )
                                    
                                    # Saving the detected cropped images
                                    crop = frame[y1:y2, x1:x2]
                                    extension = ".jpg"  # Save as jpg
                                    cropped_object_image_path = f"{output_folder}/{base_filename}_frame_{frame_count}_{class_name}_{object_counter}{extension}"
                                    
                                    if cv2.imwrite(cropped_object_image_path, crop):
                                        cropped_object_images.append(
                                            {
                                                "cropped_object_image_path": cropped_object_image_path,
                                                "class_name": class_name,
                                                "frame_number": frame_count,
                                            }
                                        )
                                        object_counter += 1
                                        print(f"Saved object from frame {frame_count}: {class_name} to {cropped_object_image_path}")
                                    else:
                                        print(f"Failed to save object from frame {frame_count}")

                frame_count += 1

        finally:
            cap.release()

        return cropped_object_images


# if __name__ == "__main__":
#     detector = YoloObjectDetector(model_name="yolo11m.pt", confidence=0.5, device="auto", verbose=True)
#     detector.detect_object_from_image("data/images/test.jpg")
#     detector.detect_object_from_video("data/videos/test.mp4", "data/output")
