import logging
from collections import deque, namedtuple
from typing import List

import numpy as np
import nokov.nokovsdk as nokovsdk

MarkerDataFrame = namedtuple("MarkerDataFrame", ["frame_id", "time_stamp", "marker_set_dict"])
RigidBodyDataFrame = namedtuple("RigidBodyDataFrame", ["frame_id", "time_stamp", "rigidbody_set_dict"])
SkeletonDataFrame = namedtuple("SkeletonDataFrame", ["frame_id", "time_stamp", "skeleton_set_dict"])


class CoordData(object):
    """
    Fundamental Data Structure
    Author: Chen Hao, chen960216@gmail.com
    Date: 20220415
    """

    def __init__(self, data: np.ndarray):
        self._data = data

    @property
    def x(self):
        if len(self._data) >= 1:
            return self._data[0]

    @property
    def y(self):
        if len(self._data) >= 2:
            return self._data[1]

    @property
    def z(self):
        if len(self._data) >= 3:
            return self._data[2]

    @property
    def coord(self):
        if len(self._data) >= 3:
            return self._data[:3]

    def __repr__(self):
        return f"Marker：X:{self.x:.2f} Y:{self.y:.2f} Z:{self.z:.2f}"


class MarkerData(CoordData):
    """
    Data Structure for Marker

    Author: Chen Hao, chen960216@gmail.com
    Date: 20220415
    """

    def __init__(self, x, y, z):
        """
        x, y ,z : 3D coordinate
        """

        super().__init__(data=np.array([x, y, z]) / 1000.0)


class RigidBodyData(CoordData):
    """
    Data Structure for Rigid Body
    Author: Chen Hao, chen960216@gmail.com, updated by weiwei
    Date: 20220415, 20220418weiwei
    """

    def __init__(self, x, y, z, qx, qy, qz, qw, mean_error):
        """
        x, y ,z : 3D coordinate
        qx, qy, qz, qw: Quaternion
        """
        super().__init__(data=np.array([x / 1000, y / 1000, z / 1000, qw, qx, qy, qz]))
        self._markers = []
        self._mean_error = mean_error

    @property
    def qx(self):
        return self._data[3]

    @property
    def qy(self):
        return self._data[4]

    @property
    def qz(self):
        return self._data[5]

    @property
    def qw(self):
        return self._data[6]

    @property
    def quat(self):
        return np.array([self.qw, self.qx, self.qy, self.qz])

    @property
    def markers(self):
        return self.markers

    @property
    def mean_error(self):
        return self._mean_error

    def add_marker(self, marker: MarkerData):
        self._markers.append(marker)

    def __len__(self):
        return len(self._markers)

    def __repr__(self):
        return f"Segment：Tx:{self.x:.2f} Ty:{self.y:.2f} Tz:{self.z:.2f} qw:{self.qw:.3f} qx:{self.qx:.3f} qy:{self.qy:.3f} qz:{self.qz:.3f}"


class SkeletonData(object):
    """
    Data Structure for Skeleton

    Author: Chen Hao, chen960216@gmail.com
    Date: 20220415
    """

    def __init__(self):
        self._rigidbodies = []

    @property
    def rigidbodys(self):
        return self._rigidbodies

    def add_rigidbody(self, body: RigidBodyData):
        self._rigidbodies.append(body)

    def __len__(self):
        return len(self._rigidbodies)

    def __repr__(self):
        out = f""
        for i in range(len(self)):
            if i > 0:
                out += "\n"
            rigidbody = self._rigidbodies[i]
            out += f"Segment：Tx:{rigidbody.x:.2f} Ty:{rigidbody.y:.2f} Tz:{rigidbody.z:.2f} qw:{rigidbody.qw:.3f} qx:{rigidbody.qx:.3f} qy:{rigidbody.qy:.3f} qz:{rigidbody.qz:.3f}"
        return out


class DataBuffer(object):
    """
    Data Buffer
    TODO: Use deque to replace the list

    Author: Chen Hao, chen960216@gmail.com
    Date: 20220415
    """

    def __init__(self, length):
        self._length = length
        self._data = []

    def append(self, data):
        if len(self._data) < self._length:
            self._data.append(data)
        else:
            self._data.pop(0)
            self._data.append(data)

    def get_last(self):
        if len(self._data) > 0:
            return self._data[-1]
        else:
            return None

    def clear(self):
        self._data = []


def to_marker_data(marker) -> MarkerData:
    """
    Author: Chen Hao, chen960216@gmail.com
    Date: 20220415
    Convert Raw Marker Data from Nokov Client to MarkerData
    """
    return MarkerData(x=marker[0], y=marker[1], z=marker[2])


def to_body_data(body: nokovsdk.RigidBodyData) -> RigidBodyData:
    """
    Author: Chen Hao, chen960216@gmail.com
    Date: 20220415
    Convert Raw RigidBody Data from Nokov Client to RigidBodyData
    """
    rigid_body_data = RigidBodyData(x=body.x,
                                    y=body.y,
                                    z=body.z,
                                    qx=body.qx,
                                    qy=body.qy,
                                    qz=body.qz,
                                    qw=body.qw,
                                    mean_error=body.MeanError)
    [rigid_body_data.add_marker(marker=to_marker_data(body.Markers[_])) for _ in range(body.nMarkers)]
    return rigid_body_data


def to_skeleton_data(skeleton: nokovsdk.SkeletonData) -> SkeletonData:
    """
    Author: Chen Hao, chen960216@gmail.com
    Date: 20220415
    Convert Raw Skeleton Data from Nokov Client to RigidBodyData
    """
    skeleton_data = SkeletonData()
    [skeleton_data.add_rigidbody(body=to_body_data(skeleton.RigidBodyData[_])) for _ in range(skeleton.nRigidBodies)]
    return skeleton_data


class NokovClient(object):
    """
    Author: Chen Hao, chen960216@gmail.com, updated by weiwei
    Date: 20220415, 20220418weiwei
    TODO: check if it is possible to match the MarkerID in RigidBody with MarkSet
    """

    def __init__(self, server_ip='10.1.1.198', data_buf_len=100, logger=logging.getLogger(__name__)):
        # buffer to restore data
        self._marker_set_buffer = DataBuffer(data_buf_len)
        self._rigidbody_set_buffer = DataBuffer(data_buf_len)
        self._skeleton_set_buffer = DataBuffer(data_buf_len)
        self._other_marker_set_buffer = DataBuffer(data_buf_len)
        # frame infomation
        self._cur_frame_no = 0
        self._pre_frame_no = 0
        # looger info
        self._logger = logger
        # init nokov client
        self._server_ip = server_ip
        self._client = nokovsdk.PySDKClient()
        self._client.PySetVerbosityLevel(0)
        self._client.PySetDataCallback(self._read_data_func, None)
        print("Begin to init the SDK Client")
        ret = self._client.Initialize(bytes(self._server_ip, encoding="utf8"))
        if ret == 0:
            print("Connect to the Seeker Succeed")
        else:
            print("Connect Failed: [%d]" % ret)
            exit(0)

    def _get_data_frame(self, buffer: DataBuffer):
        data = buffer.get_last()
        if data is None:
            return None
        else:
            return data
        # ntry = 0
        # while ntry < 3:
        #     data = buffer.get_last()
        #     if data is None:
        #         return None
        #     ntry += 1
        #     if data.frame_id == self._cur_frame_no:
        #         return data
        # return None

    def get_rigidbody_frame(self) -> RigidBodyDataFrame:
        return self._get_data_frame(buffer=self._rigidbody_set_buffer)

    def get_marker_set_frame(self) -> MarkerDataFrame:
        return self._get_data_frame(buffer=self._marker_set_buffer)

    def get_skeleton_set_frame(self) -> SkeletonDataFrame:
        return self._get_data_frame(buffer=self._skeleton_set_buffer)

    def _read_data_func(self, pFrameOfMocapData, pUserData):
        # check frame data
        if pFrameOfMocapData == None:
            self._logger.debug("Not get the data frame.\n")
        else:
            frameData = pFrameOfMocapData.contents
            self._cur_frame_no = frameData.iFrame
            if self._cur_frame_no == self._pre_frame_no:
                self._logger.debug("Current frame is not equal to previous frame")
                return
            self._pre_frame_no = self._cur_frame_no
            # frame and time stamp infomation
            frame_id, time_stamp = frameData.iFrame, frameData.iTimeStamp
            # read marker set data
            marker_set_dict = {}
            for iMarkerSet in range(frameData.nMarkerSets):
                marker_set = frameData.MocapData[iMarkerSet]
                # uncomment to show information in the marker_set
                # marker_set.show()
                marker_set_dict[marker_set.szName] = [to_marker_data(marker_set.Markers[_]) for _ in
                                                      range(marker_set.nMarkers)]
            self._marker_set_buffer.append(
                MarkerDataFrame(frame_id=frame_id, time_stamp=time_stamp, marker_set_dict=marker_set_dict))
            # read rigidbody data
            rigidbody_set_dict = {}
            for iBody in range(frameData.nRigidBodies):
                body = frameData.RigidBodies[iBody]
                # create a rigid body data
                # print([body.MarkerIDs[_] for _ in range(body.nMarkers)])
                rigidbody_set_dict[body.ID] = to_body_data(body)
            self._rigidbody_set_buffer.append(
                RigidBodyDataFrame(frame_id=frame_id, time_stamp=time_stamp, rigidbody_set_dict=rigidbody_set_dict))
            # read skeleton data
            skeleton_set_dict = {}
            for iSkeleton in range(frameData.nSkeletons):
                # Segments
                skeleton = frameData.Skeletons[iSkeleton]
                skeleton_set_dict[skeleton.skeletonID] = to_skeleton_data(skeleton)
            self._skeleton_set_buffer.append(
                SkeletonDataFrame(frame_id=frame_id, time_stamp=time_stamp, skeleton_set_dict=skeleton_set_dict))
            # read other marker information
            other_marker_set_dict = {}
            for i in range(frameData.nOtherMarkers):
                other_marker_set_dict[i] = to_marker_data(frameData.OtherMarkers[i])
            self._other_marker_set_buffer.append(
                MarkerDataFrame(frame_id=frame_id, time_stamp=time_stamp, marker_set_dict=other_marker_set_dict))


if __name__ == "__main__":
    # 160,
    server = NokovClient()
    while True:
        data = server.get_rigidbody_frame()
        if data is not None:
            print(data)
