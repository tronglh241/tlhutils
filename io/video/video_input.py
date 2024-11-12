from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

import cv2
import numpy.typing as npt


class AbstractVideoReader(ABC):
    @abstractmethod
    def read_frame(self) -> Tuple[bool, Optional[npt.NDArray[Any]]]:
        '''
        Abstract method to read the next frame from the video source.

        Returns:
            tuple: (ret, frame), where ret is a boolean indicating success,
                   and frame is the video frame (or None if the video ends).
        '''
        pass

    @abstractmethod
    def release(self) -> None:
        '''
        Abstract method to release resources associated with the video source.
        '''
        pass

    @abstractmethod
    def get_frame_dimensions(self) -> Tuple[int, int]:
        '''
        Abstract method to return the frame dimensions of the video source.

        Returns:
            tuple: (frame_width, frame_height)
        '''
        pass

    @abstractmethod
    def get_fps(self) -> float:
        '''
        Abstract method to return the frames per second (FPS) of the video source.

        Returns:
            float: FPS of the video source.
        '''
        pass

    @abstractmethod
    def get_frame_count(self) -> int:
        '''
        Abstract method to return the total number of frames in the video source.

        Returns:
            int: Total number of frames.
        '''
        pass


class VideoReader(AbstractVideoReader):
    def __init__(self, video_path: str) -> None:
        '''
        Initializes the VideoReader class.

        Args:
            video_path (str): Path to the video file.
        '''
        self.video_path: str = video_path
        self.cap: cv2.VideoCapture = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise ValueError(f'Error: Unable to open video file {video_path}')

        self.frame_width: int = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height: int = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps: float = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count: int = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def read_frame(self) -> Tuple[bool, Optional[npt.NDArray[Any]]]:
        '''
        Reads the next frame from the video file.

        Returns:
            tuple: (ret, frame), where ret is a boolean indicating success,
                   and frame is the video frame (or None if the video ends).
        '''
        ret: bool
        frame: Optional[npt.NDArray[Any]]
        ret, frame = self.cap.read()
        return ret, frame

    def release(self) -> None:
        '''
        Releases the video capture object and frees resources.
        '''
        self.cap.release()

    def get_frame_dimensions(self) -> Tuple[int, int]:
        '''
        Returns the dimensions of the video frames.

        Returns:
            tuple: (frame_width, frame_height)
        '''
        return self.frame_width, self.frame_height

    def get_fps(self) -> float:
        '''
        Returns the frames per second (FPS) of the video file.

        Returns:
            float: Frames per second.
        '''
        return self.fps

    def get_frame_count(self) -> int:
        '''
        Returns the total number of frames in the video file.

        Returns:
            int: Total number of frames.
        '''
        return self.frame_count
