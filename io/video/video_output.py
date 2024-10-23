from abc import ABC, abstractmethod

import cv2


class AbstractVideoHandler(ABC):
    @abstractmethod
    def handle_frame(self, frame):
        '''
        Abstract method to handle a video frame.

        Args:
            frame (ndarray): The frame to be handled (displayed or saved).
        '''
        pass

    @abstractmethod
    def release(self):
        '''
        Abstract method to release resources.
        '''
        pass


class VideoWriter(AbstractVideoHandler):
    def __init__(self, output_path, fps, frame_size, codec='mp4v'):
        '''
        Initializes the VideoWriter class.

        Args:
            output_path (str): Path to the output video file.
            fps (float): Frames per second for the output video.
            frame_size (tuple): Size of each frame (width, height).
            codec (str): FourCC code for the video codec (default is 'mp4v').
        '''
        self.output_path = output_path
        self.fps = fps
        self.frame_size = frame_size
        self.codec = codec

        # Convert codec to four-character code
        fourcc = cv2.VideoWriter_fourcc(*self.codec)  # type: ignore

        # Initialize VideoWriter object
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

        if not self.writer.isOpened():
            raise ValueError(f'Error: Unable to open video writer for {output_path}')

    def handle_frame(self, frame):
        '''
        Writes a frame to the video file.

        Args:
            frame (ndarray): The frame to be written.
        '''
        self.writer.write(frame)

    def release(self):
        '''
        Releases the video writer object and finalizes the output video file.
        '''
        self.writer.release()


class VideoDisplay(AbstractVideoHandler):
    def __init__(self, window_name: str = 'Video', delay: int = 10):
        '''
        Initializes the ScreenVideoDisplay class for showing video frames on the screen.

        Args:
            window_name (str): Name of the display window (default is 'Video').
        '''
        self.window_name = window_name
        self.delay = delay

    def handle_frame(self, frame):
        '''
        Displays a frame on the screen.

        Args:
            frame (ndarray): The frame to be displayed.
        '''
        cv2.imshow(self.window_name, frame)
        return self.wait_key()

    def release(self):
        '''
        Closes the display window.
        '''
        cv2.destroyAllWindows()

    def wait_key(self):
        '''
        Waits for a key press for a given amount of time.

        Args:
            delay (int): Delay in milliseconds for key waiting (default is 1ms).

        Returns:
            int: ASCII code of the pressed key, or -1 if no key was pressed.
        '''
        return cv2.waitKey(self.delay)
