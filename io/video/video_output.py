from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

import cv2
import numpy.typing as npt


class AbstractVideoHandler(ABC):
    @abstractmethod
    def handle_frame(self, frame: npt.NDArray[Any]) -> Optional[int]:
        '''
        Abstract method to handle a video frame.

        Args:
            frame (npt.NDArray[Any]): The frame to be handled (displayed or saved).

        Returns:
            Optional[int]: Used for return codes, e.g., ASCII code of a pressed key (for display), or None.
        '''
        pass

    @abstractmethod
    def release(self) -> None:
        '''
        Abstract method to release resources.
        '''
        pass


class VideoWriter(AbstractVideoHandler):
    def __init__(self, output_path: str, fps: float, frame_size: Tuple[int, int], codec: str = 'mp4v') -> None:
        '''
        Initializes the VideoWriter class.

        Args:
            output_path (str): Path to the output video file.
            fps (float): Frames per second for the output video.
            frame_size (Tuple[int, int]): Size of each frame (width, height).
            codec (str): FourCC code for the video codec (default is 'mp4v').
        '''
        self.output_path: str = output_path
        self.fps: float = fps
        self.frame_size: Tuple[int, int] = frame_size
        self.codec: str = codec

        # Convert codec to four-character code
        fourcc: int = cv2.VideoWriter_fourcc(*self.codec)  # type: ignore

        # Initialize VideoWriter object
        self.writer: cv2.VideoWriter = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

        if not self.writer.isOpened():
            raise ValueError(f'Error: Unable to open video writer for {output_path}')

    def handle_frame(self, frame: npt.NDArray[Any]) -> None:
        '''
        Writes a frame to the video file.

        Args:
            frame (npt.NDArray[Any]): The frame to be written.
        '''
        self.writer.write(frame)

    def release(self) -> None:
        '''
        Releases the video writer object and finalizes the output video file.
        '''
        self.writer.release()


class VideoDisplay(AbstractVideoHandler):
    def __init__(self, window_name: str = 'Video', delay: int = 10) -> None:
        '''
        Initializes the ScreenVideoDisplay class for showing video frames on the screen.

        Args:
            window_name (str): Name of the display window (default is 'Video').
            delay (int): Delay in milliseconds for key waiting (default is 10ms).
        '''
        self.window_name: str = window_name
        self.delay: int = delay

    def handle_frame(self, frame: npt.NDArray[Any]) -> Optional[int]:
        '''
        Displays a frame on the screen.

        Args:
            frame (npt.NDArray[Any]): The frame to be displayed.

        Returns:
            Optional[int]: ASCII code of the pressed key if any, otherwise None.
        '''
        cv2.imshow(self.window_name, frame)
        return self.wait_key()

    def release(self) -> None:
        '''
        Closes the display window.
        '''
        cv2.destroyAllWindows()

    def wait_key(self) -> int:
        '''
        Waits for a key press for a given amount of time.

        Returns:
            int: ASCII code of the pressed key, or -1 if no key was pressed.
        '''
        return cv2.waitKey(self.delay)
