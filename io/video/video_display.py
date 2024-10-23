import cv2


class VideoDisplay:
    def __init__(self, window_name="Video"):
        """
        Initializes the VideoDisplay class for showing video frames on the screen.

        Args:
            window_name (str): Name of the display window (default is "Video").
        """
        self.window_name = window_name
        # Create a window for displaying frames
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def show_frame(self, frame):
        """
        Displays a frame on the screen.

        Args:
            frame (ndarray): The frame to be displayed.
        """
        cv2.imshow(self.window_name, frame)

    def wait_key(self, delay=1):
        """
        Waits for a key press for a given amount of time.

        Args:
            delay (int): Delay in milliseconds for key waiting (default is 1ms).

        Returns:
            int: ASCII code of the pressed key, or -1 if no key was pressed.
        """
        return cv2.waitKey(delay)

    def release(self):
        """
        Closes the display window.
        """
        cv2.destroyAllWindows()
