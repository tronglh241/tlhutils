import json
from enum import Enum
from typing import Any, Dict, Iterator, Optional

import cv2
import numpy as np
import numpy.typing as npt

from .sqlite3_utils import SQLiteCursor, SQLiteDatabase
from .synchronizer import Item, Source, Synchronizer
from .wheel_odom_estimator import WheelOdomEstimator


class Gear(Enum):
    VF_GEAR_PARK = 'VF_GEAR_PARK'
    VF_GEAR_REVERSE = 'VF_GEAR_REVERSE'
    VF_GEAR_NEUTRAL = 'VF_GEAR_NEUTRAL'
    VF_GEAR_DRIVE = 'VF_GEAR_DRIVE'


class FrameItem:
    def __init__(
        self,
        front: npt.NDArray[Any],
        left: npt.NDArray[Any],
        rear: npt.NDArray[Any],
        right: npt.NDArray[Any],
        x: Optional[float] = None,
        y: Optional[float] = None,
        theta: Optional[float] = None,
        synchronized: bool = True,
        motion: bool = True,
    ) -> None:
        self.front: npt.NDArray[Any] = front
        self.left: npt.NDArray[Any] = left
        self.rear: npt.NDArray[Any] = rear
        self.right: npt.NDArray[Any] = right
        self.x: Optional[float] = x
        self.y: Optional[float] = y
        self.theta: Optional[float] = theta
        self.synchronized: bool = synchronized
        self.motion: bool = motion

    @property
    def cams(self) -> Dict[str, npt.NDArray[Any]]:
        return {
            'front': self.front,
            'left': self.left,
            'rear': self.rear,
            'right': self.right,
        }


class SQLiteSource(Source):
    def __init__(self, cursor: SQLiteCursor) -> None:
        self.cursor: SQLiteCursor = cursor
        super(SQLiteSource, self).__init__()

    def reset(self) -> None:
        pass

    def next(self) -> Optional[Item]:
        value = self.cursor.fetchone()

        if value is None:
            item = None
        else:
            item = Item(value[1], value[0])

        return item


class VinAIDDSBag:
    def __init__(
        self,
        db_path: str,
        db_json_path: Optional[str] = None,
        compressed: bool = False,
        grayscale: bool = False,
        time_diff_thres: float = 0.020,  # second
        dist_thres: float = 0.05,  # meter
        wheel_odom_estimator_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.db_path: str = db_path
        self.db_json_path: Optional[str] = db_json_path
        self.compressed: bool = compressed
        self.grayscale: bool = grayscale
        self.time_diff_thres = time_diff_thres
        self.dist_thres = dist_thres
        self.ref_cam: str = [name for name in self.cam_topics][0]

        self.db: Optional[SQLiteDatabase] = None
        self.synchronizer: Optional[Synchronizer] = None

        self.pose_included: bool = self.db_json_path is not None

        if self.pose_included:
            self.db_json: Optional[SQLiteDatabase] = None
            self.ref_sensor: str = [name for name in self.dbw_topics][0]
            self.wheel_odom_estimator: WheelOdomEstimator = WheelOdomEstimator(wheel_odom_estimator_config)

            self.last_x = 0.0
            self.last_y = 0.0

    def __iter__(self) -> Iterator[FrameItem]:
        self.reset()
        return self

    def __next__(self) -> FrameItem:
        next_item = self.next()

        if next_item is None:
            if self.db is not None:
                self.db.close()

            if self.pose_included and self.db_json is not None:
                self.db_json.close()

            raise StopIteration

        return next_item

    @property
    def cam_topics(self) -> Dict[str, str]:
        if self.grayscale:
            return {
                'front': '/adas/sensors/camera/gray_fisheye_front@0',
                'left': '/adas/sensors/camera/gray_fisheye_left@0',
                'rear': '/adas/sensors/camera/gray_fisheye_rear@0',
                'right': '/adas/sensors/camera/gray_fisheye_right@0',
            }

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

    @property
    def dbw_topics(self) -> Dict[str, str]:
        return {
            'idb': '/adas/dbw/vf_idb_info@0',
            'eps': '/adas/dbw/vf_eps_info@0',
            'vcu': '/adas/dbw/vf_vcu_info@0',
            'yss': '/adas/dbw/vf_yss_info@0',
        }

    def reset(self) -> None:
        if self.db is not None:
            self.db.close()

        self.db = SQLiteDatabase(self.db_path)
        cursors: Dict[str, SQLiteCursor] = {}

        # Check if all tables exist
        for table_name in self.cam_topics.values():
            if not self.db.table_exists(table_name):
                raise ValueError(f'Table {table_name} does not exist in database {self.db_path}.')

        cam_cursors = {
            name: self.db.query(f'SELECT rti_cdr_sample, SampleInfo_reception_timestamp FROM "{topic}"')
            for name, topic in self.cam_topics.items()
        }
        cursors.update(cam_cursors)

        if self.pose_included:
            if self.db_json is not None:
                self.db_json.close()

            assert self.db_json_path is not None
            self.db_json = SQLiteDatabase(self.db_json_path)

            # Check if all tables exist
            for table_name in self.dbw_topics.values():
                if not self.db_json.table_exists(table_name):
                    raise ValueError(f'Table {table_name} does not exist in database {self.db_path}.')

            dbw_cursors = {
                name: self.db_json.query(f'SELECT rti_json_sample, SampleInfo_reception_timestamp FROM "{topic}"')
                for name, topic in self.dbw_topics.items()
            }
            cursors.update(dbw_cursors)

        self.synchronizer = Synchronizer({name: SQLiteSource(cursor) for name, cursor in cursors.items()})
        self.synchronizer_it = iter(self.synchronizer)

    def next(self) -> Optional[FrameItem]:
        while True:
            try:
                sync_items = next(self.synchronizer_it)
            except StopIteration:
                return None

            if self.pose_included:
                if sync_items[self.ref_sensor].new:
                    dbw_values = {name: json.loads(sync_items[name].value) for name in self.dbw_topics}
                    self.wheel_odom_estimator.update_info_and_estimate(
                        speed_timestamp=dbw_values['idb']['timestamp'],
                        gear=2 if Gear(dbw_values['vcu']['actual_gear']) == Gear.VF_GEAR_REVERSE else 0,
                        wheel_speed=[
                            dbw_values['idb']['wheel_speed_fl'],
                            dbw_values['idb']['wheel_speed_fr'],
                            dbw_values['idb']['wheel_speed_rl'],
                            dbw_values['idb']['wheel_speed_rr'],
                        ],
                        yaw_rate=dbw_values['yss']['yaw_rate'],
                        steering_angle=dbw_values['eps']['steering_angle'],
                    )

            if sync_items[self.ref_cam].new:
                min_timestamp = min(sync_items[name].timestamp for name in self.cam_topics)
                max_time_diff = max(sync_items[name].timestamp - min_timestamp for name in self.cam_topics)

                frame_item = FrameItem(
                    front=self.decode(sync_items['front'].value),
                    left=self.decode(sync_items['left'].value),
                    rear=self.decode(sync_items['rear'].value),
                    right=self.decode(sync_items['right'].value),
                    synchronized=max_time_diff < self.time_diff_thres * 1e9,
                )

                if self.pose_included:
                    x, y, theta = self.wheel_odom_estimator.get_pose_at_time(sync_items[self.ref_cam].timestamp)
                    frame_item.x = x
                    frame_item.y = y
                    frame_item.theta = theta

                    frame_item.motion = (x - self.last_x) ** 2 + (y - self.last_y) ** 2 >= self.dist_thres ** 2
                    self.last_x = x
                    self.last_y = y

                break

        return frame_item

    def decode(self, buf: bytes) -> npt.NDArray[Any]:
        buf = bytearray(buf)
        if self.grayscale:
            arr = np.asarray(buf[16:], dtype=np.uint8)
            img: npt.NDArray[Any] = arr.reshape(800, 1280)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif self.compressed:
            arr = np.asarray(buf[60:], dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        else:
            arr = np.asarray(buf[16:], dtype=np.uint8)
            img = arr.reshape(800, 1280, 3)[..., ::-1]

        assert img.shape == (800, 1280, 3)
        return img
