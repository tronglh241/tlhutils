import cv2


class VideoReader:
    def __init__(self, video_path):
        """
        Initializes the VideoReader class.

        Args:
            video_path (str): Path to the video file.
        """
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise ValueError(f"Error: Unable to open video file {video_path}")

        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def get_frame_dimensions(self):
        """
        Returns the dimensions of the video frames.

        Returns:
            tuple: (frame_width, frame_height)
        """
        return self.frame_width, self.frame_height

    def get_fps(self):
        """
        Returns the frames per second (FPS) of the video.

        Returns:
            float: Frames per second.
        """
        return self.fps

    def get_frame_count(self):
        """
        Returns the total number of frames in the video.

        Returns:
            int: Total number of frames.
        """
        return self.frame_count

    def read_frame(self):
        """
        Reads the next frame from the video.

        Returns:
            tuple: (ret, frame), where ret is a boolean indicating success,
                   and frame is the video frame (or None if the video ends).
        """
        ret, frame = self.cap.read()
        if not ret:
            return False, None
        return True, frame

    def release(self):
        """
        Releases the video capture object and frees resources.
        """
        self.cap.release()
