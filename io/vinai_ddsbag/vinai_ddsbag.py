import bisect
import sqlite3
from collections import defaultdict
from enum import Enum
from typing import Any, Dict, List, Optional

import cv2
import numpy as np


class SQLiteCursor:
    def __init__(self, connection):
        """
        Initializes the SQLiteCursor class.

        Args:
            connection (sqlite3.Connection): SQLite connection object.
        """
        self.connection = connection
        self.cursor = self.connection.cursor()

    def execute(self, query, parameters=None):
        """
        Executes a given SQL query.

        Args:
            query (str): The SQL query to execute.
            parameters (tuple or dict, optional): Parameters for the query.
        """
        if parameters:
            self.cursor.execute(query, parameters)
        else:
            self.cursor.execute(query)

    def fetchall(self):
        """Fetches all rows of the query result."""
        return self.cursor.fetchall()

    def fetchone(self):
        """Fetches the next row of a query result."""
        return self.cursor.fetchone()

    def commit(self):
        """Commits the current transaction."""
        self.connection.commit()

    def close(self):
        """Closes the cursor."""
        self.cursor.close()


class SQLiteDatabase:
    def __init__(self, db_path):
        """
        Initializes the SQLiteDatabase class.

        Args:
            db_path (str): Path to the SQLite database file.
        """
        self.connection = sqlite3.connect(db_path)

    def table_exists(self, table_name):
        """
        Checks if a table exists in the database.

        Args:
            table_name (str): Name of the table to check.

        Returns:
            bool: True if the table exists, False otherwise.
        """
        query = "SELECT name FROM sqlite_master WHERE type='table' AND name=?;"
        cursor = self.query(query, (table_name,))
        result = cursor.fetchone() is not None
        cursor.close()
        return result

    def query(self, query, parameters=None):
        """
        Executes a given SQL query and returns the results.

        Args:
            query (str): The SQL query to execute.
            parameters (tuple or dict, optional): Parameters for the query.

        Returns:
            list: A list of rows returned by the query.
        """
        cursor = SQLiteCursor(self.connection)
        cursor.execute(query, parameters)
        return cursor

    def close(self):
        """Closes the database connection."""
        self.connection.close()


class Gear(Enum):
    VF_GEAR_PARK = 'VF_GEAR_PARK'
    VF_GEAR_REVERSE = 'VF_GEAR_REVERSE'
    VF_GEAR_NEUTRAL = 'VF_GEAR_NEUTRAL'
    VF_GEAR_DRIVE = 'VF_GEAR_DRIVE'


class FrameItem:
    def __init__(
        self,
        front,
        left,
        rear,
        right,
    ):
        self.front = front
        self.left = left
        self.rear = rear
        self.right = right


class VinAIDDSBag:
    def __init__(self, db_path: str, compressed: bool = False, buffer_size: int = 20):
        self.db_path = db_path
        self.compressed = compressed
        self.ref_cam = [cam_name for cam_name in self.cam_topics][0]
        self.db: Optional[SQLiteDatabase] = None
        self.buffer_size = buffer_size
        self.frame_buffers: Dict[str, List[Any]] = defaultdict(list)

    def __iter__(self):
        return self

    def __next__(self):
        next_item = self.next()

        if next_item is None:
            raise StopIteration

        return next_item

    @property
    def cam_topics(self) -> Dict[str, str]:
        if self.compressed:
            return {
                'front': '/adas/sensors/camera/fisheye_front@0',
                'left': '/adas/sensors/camera/fisheye_left@0',
                'rear': '/adas/sensors/camera/fisheye_rear@0',
                'right': '/adas/sensors/camera/fisheye_right@0',
            }
        else:
            return {
                'front': '/adas/sensors/camera/rgb_fisheye_front@0',
                'left': '/adas/sensors/camera/rgb_fisheye_left@0',
                'rear': '/adas/sensors/camera/rgb_fisheye_rear@0',
                'right': '/adas/sensors/camera/rgb_fisheye_right@0',
            }

    def reset(self):
        if self.db is not None:
            self.db.close()

        self.db = SQLiteDatabase(self.db_path)

        # Check if all tables exist
        for table_name in self.cam_topics.values():
            if not self.db.table_exists(table_name):
                raise ValueError(f'Table {table_name} does not exist in database {self.db_path}.')

        self.cursors = {
            cam_name: self.db.query(f'SELECT rti_cdr_sample, SampleInfo_reception_timestamp FROM "{topic}"')
            for cam_name, topic in self.cam_topics.items()
        }

    def next(self) -> Optional[FrameItem]:
        for name, cursor in self.cursors.items():
            while len(self.frame_buffers[name]) < self.buffer_size:
                row = self.cursors[name].fetchone()

                if row is None:
                    break

                self.frame_buffers[name].append(row)

        if any(not buffer for buffer in self.frame_buffers.values()):
            return None

        frames = {self.ref_cam: self.frame_buffers[self.ref_cam].pop(0)}

        for name, frame_buffer in self.frame_buffers.items():
            if name == self.ref_cam:
                continue

            frames[name] = self.find_closest(frames[self.ref_cam], frame_buffer)

        frame_item = FrameItem(
            front=self.decode(frames['front'][0]),
            left=self.decode(frames['left'][0]),
            rear=self.decode(frames['rear'][0]),
            right=self.decode(frames['right'][0]),
        )

        return frame_item

    def find_closest(self, target, sorted_list):
        pos = bisect.bisect_left(sorted_list, target[1], key=lambda x: x[1])

        if pos == 0:
            result = sorted_list[0]
        elif pos == len(sorted_list):
            result = sorted_list[-1]
        else:
            before = sorted_list[pos - 1]
            after = sorted_list[pos]

            if after[1] - target[1] < target[1] - before[1]:
                result = after
                pos = pos - 1
            else:
                result = before

            for _ in range(pos):
                del sorted_list[0]

        return result

    def decode(self, buf):
        buf = bytearray(buf)
        if self.compressed:
            arr = np.asarray(buf[60:], dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        else:
            arr = np.asarray(buf[16:], dtype=np.uint8)
            img = arr.reshape(800, 1280, 3)[..., ::-1]

        assert img.shape == (800, 1280, 3)
        return img
