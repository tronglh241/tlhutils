import cv2


class VideoWriter:
    def __init__(self, output_path, fps, frame_size, codec='mp4v'):
        """
        Initializes the VideoWriter class.

        Args:
            output_path (str): Path to the output video file.
            fps (float): Frames per second for the output video.
            frame_size (tuple): Size of each frame (width, height).
            codec (str): FourCC code for the video codec (default is 'mp4v').
        """
        self.output_path = output_path
        self.fps = fps
        self.frame_size = frame_size
        self.codec = codec

        # Convert codec to four-character code
        fourcc = cv2.VideoWriter_fourcc(*self.codec)  # type: ignore

        # Initialize VideoWriter object
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

        if not self.writer.isOpened():
            raise ValueError(f"Error: Unable to open video writer for {output_path}")

    def write_frame(self, frame):
        """
        Writes a frame to the video file.

        Args:
            frame (ndarray): The frame to be written.
        """
        self.writer.write(frame)

    def release(self):
        """
        Releases the video writer object and finalizes the output video file.
        """
        self.writer.release()
