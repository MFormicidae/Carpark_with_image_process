import cv2

class Multimedia:
    def __init__(self):
        self.image_name = r"media\master_frame.jpg"
        self.video_name = "morning.mp4"

    def get_image(self):
     
        try:
            img = cv2.imread(self.image_name)
            if img is None:
                raise FileNotFoundError(f"Image file '{self.image_name}' not found.")
            return img, self.image_name
        except Exception as e:
            print(f"Error occurred: {e}")
            return None

    def get_video(self):
        try:
            video_capture = cv2.VideoCapture(self.video_name)
            if not video_capture.isOpened():
                raise FileNotFoundError(f"Video file '{self.video_name}' not found or cannot be opened.")
            return video_capture
        except Exception as e:
            print(f"Error occurred: {e}")
            return None
