import cv2
import os

class HaarFaceDetector:
    """
    A haar face detection class.
    """

    def __init__(self):
        self.detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def detect_faces_haar(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
        )
        return faces

    def process_image_with_face_detection(
        self, image_path, output_folder_path
    ):
        saved_file_paths = []
        number_of_faces_saved = 0
        # Load and validate image
        if not os.path.exists(image_path):
            print(f"Image file {image_path} not found")
            return {
                "number_of_faces_saved": number_of_faces_saved,
                "saved_file_paths": saved_file_paths,
            }
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load image from {image_path}")
            return {
                "number_of_faces_saved": number_of_faces_saved,
                "saved_file_paths": saved_file_paths,
            }
        # Detect faces
        faces = self.detect_faces_haar(image)
        # Create output directory
        output_folder = os.path.join(output_folder_path, "detected_faces")
        os.makedirs(output_folder, exist_ok=True)
        # Save each detected face
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        for i, (x, y, w, h) in enumerate(faces):
            # Ensure coordinates are within image bounds
            x = max(0, x)
            y = max(0, y)
            w = min(w, image.shape[1] - x)
            h = min(h, image.shape[0] - y)
            # Extract face region
            face_crop = image[y : y + h, x : x + w]
            # Generate unique filename
            face_filename = f"{base_filename}_face_{i+1}.jpg"
            face_path = os.path.join(output_folder, face_filename)
            # Save face image
            if cv2.imwrite(face_path, face_crop):
                number_of_faces_saved += 1
                saved_file_paths.append(face_path)
                print(f"Saved face {i+1} to {face_path}")
            else:
                print(f"Failed to save face {i+1}")
        return {
            "number_of_faces_saved": number_of_faces_saved,
            "saved_file_paths": saved_file_paths,
        }

    def process_video_with_face_detection(
        self, video_path, output_folder_path, frame_interval=30
    ):
        """
        Process a video to detect and save all faces from frames.

        Args:
            video_path (str): Path to input video
            output_folder_path (str): Directory for saving faces
            frame_interval (int): Process every Nth frame (default: 30)

        Returns:
            dict: {"number_of_faces_saved": int, "saved_file_paths": list}
        """
        
        saved_file_paths = []
        number_of_faces_saved = 0
        
        # Load and validate video
        if not os.path.exists(video_path):
            print(f"Video file {video_path} not found")
            return {
                "number_of_faces_saved": number_of_faces_saved,
                "saved_file_paths": saved_file_paths,
            }

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Could not load video from {video_path}")
            return {
                "number_of_faces_saved": number_of_faces_saved,
                "saved_file_paths": saved_file_paths,
            }

        # Create output directory
        output_folder = os.path.join(output_folder_path, "detected_faces")
        os.makedirs(output_folder, exist_ok=True)

        base_filename = os.path.splitext(os.path.basename(video_path))[0]
        frame_count = 0
        face_counter = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Process every frame_interval frames
                if frame_count % frame_interval == 0:
                    # Detect faces in current frame
                    faces = self.detect_faces_haar(frame)

                    # Save each detected face
                    for i, (x, y, w, h) in enumerate(faces):
                        # Ensure coordinates are within frame bounds
                        x = max(0, x)
                        y = max(0, y)
                        w = min(w, frame.shape[1] - x)
                        h = min(h, frame.shape[0] - y)

                        # Extract face region
                        face_crop = frame[y : y + h, x : x + w]

                        # Generate unique filename with frame info
                        face_filename = f"{base_filename}_frame_{frame_count}_face_{face_counter + 1}.jpg"
                        face_path = os.path.join(output_folder, face_filename)

                        # Save face image
                        if cv2.imwrite(face_path, face_crop):
                            number_of_faces_saved += 1
                            saved_file_paths.append(face_path)
                            face_counter += 1
                            print(f"Saved face from frame {frame_count} to {face_path}")
                        else:
                            print(f"Failed to save face from frame {frame_count}")

                frame_count += 1

        finally:
            cap.release()

        return {
            "number_of_faces_saved": number_of_faces_saved,
            "saved_file_paths": saved_file_paths,
        }

# if __name__ == "__main__":
#     detector = HaarFaceDetector()
#     print(detector.process_image_with_face_detection("./sample_3.jpg"))
#     # print(detector.process_video_with_face_detection("./sample_video.mp4", "./output"))