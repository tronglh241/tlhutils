from typing import List, Optional, Tuple

import cv2
import numpy as np
from tqdm import trange

from .video_input import VideoReader
from .video_output import VideoWriter


class VideoMerger:
    def __init__(
        self,
        video_paths: List[str],
        output_path: str,
        orientation: str = 'horizontal',
        fps: Optional[float] = None,
        frame_size: Optional[Tuple[int, int]] = None,
        codec: str = 'mp4v',
    ):
        '''
        Initialize the VideoMerger.

        Args:
            video_paths (List[str]): Paths to the video files to be merged.
            output_path (str): Path for the output merged video.
            orientation (str): 'horizontal' or 'vertical' for concatenation.
            fps (Optional[float]): FPS for the output video. Defaults to FPS of the first video if None.
            frame_size (Optional[Tuple[int, int]]): Frame size for output video.
                If None, calculates compatible frame size.
        '''
        self.video_paths = video_paths
        self.output_path = output_path
        self.orientation = orientation
        self.fps = fps
        self.frame_size = frame_size
        self.codec = codec

    def merge_videos(self) -> None:  # noqa: C901
        # Initialize VideoReader for each video path
        readers = [VideoReader(path) for path in self.video_paths]

        n_frames = min(reader.get_frame_count() for reader in readers)
        frame_sizes = [reader.get_frame_dimensions() for reader in readers]

        # Use FPS of the first video if FPS is not specified
        if self.fps is None:
            fps = readers[0].get_fps()

        # Calculate frame size if not provided
        if self.frame_size is None:
            frame_size, frame_sizes = self.calculate_output_frame_size(frame_sizes)

        # Initialize VideoWriter with specified output FPS and frame size
        writer = VideoWriter(
            output_path=self.output_path,
            fps=fps,
            frame_size=frame_size,
            codec=self.codec,
        )

        for _ in trange(n_frames):
            frames = []
            ret_all = True

            # Read frames from each video
            for reader in readers:
                ret, frame = reader.read_frame()

                if not ret:
                    ret_all = False
                    break

                assert frame is not None
                frames.append(frame)

            if not ret_all:
                break

            # Resize frames to match output frame size
            for i, frame in enumerate(frames):
                if frame.shape[1::-1] != frame_sizes[i]:
                    frames[i] = cv2.resize(frame, frame_sizes[i])

            # Concatenate frames horizontally or vertically
            if self.orientation == 'horizontal':
                merged_frame = np.hstack(frames)
            else:
                merged_frame = np.vstack(frames)

            # Write the merged frame to output video
            writer.handle_frame(merged_frame)

        # Release resources
        for reader in readers:
            reader.release()

        writer.release()

    def calculate_output_frame_size(
        self,
        frame_sizes: List[Tuple[int, int]],
    ) -> Tuple[Tuple[int, int], List[Tuple[int, int]]]:
        '''
        Calculate the output frame size to ensure compatibility for stacking.

        Args:
            frame_sizes (List[Tuple[int, int]]): List of (width, height) for each video.

        Returns:
            Tuple[int, int]: Calculated frame size for stacking compatibility.
        '''
        if self.orientation == 'horizontal':
            total_width = sum([size[0] for size in frame_sizes])
            max_height = max([size[1] for size in frame_sizes])

            out_frame_size = (total_width, max_height)
            out_frame_sizes = [(size[0], max_height) for size in frame_sizes]
        else:
            max_width = max([size[0] for size in frame_sizes])
            total_height = sum([size[1] for size in frame_sizes])

            out_frame_size = (max_width, total_height)
            out_frame_sizes = [(max_width, size[1]) for size in frame_sizes]

        return out_frame_size, out_frame_sizes
