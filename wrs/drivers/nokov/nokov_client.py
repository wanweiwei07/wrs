import logging
import ctypes
from collections import namedtuple

import numpy as np
import nokov.nokovsdk as nokovsdk

from wrs import basis as rm, modeling as gm
import wrs.modeling.model_collection as mc

MarkerDataFrame = namedtuple("MarkerDataFrame", ["frame_id", "time_stamp", "marker_set_dict"])
RigidBodyDataFrame = namedtuple("RigidBodyDataFrame", ["frame_id", "time_stamp", "rigidbody_set_dict"])
SkeletonDataFrame = namedtuple("SkeletonDataFrame", ["frame_id", "time_stamp", "skeleton_set_dict"])


class RigidBodyData(object):
    """
    Data Structure for Rigid Body
    Author: Chen Hao, chen960216@gmail.com, updated by weiwei
    Date: 20220415, 20220418weiwei
    """

    def __init__(self, x, y, z, qx, qy, qz, qw, mean_error, parent_id=0):
        """
        x, y ,z : 3D coordinate in meter
        qx, qy, qz, qw: Quaternion
        parent_id : the Parent index, 0 for global
        """
        self._data = np.array([x, y, z, qw, qx, qy, qz])
        self._markers = np.array([])
        self._mean_error = mean_error
        self._parent_id = parent_id

    @property
    def x(self):
        return self._data[0]

    @property
    def y(self):
        return self._data[1]

    @property
    def z(self):
        return self._data[2]

    @property
    def qw(self):
        return self._data[3]

    @property
    def qx(self):
        return self._data[4]

    @property
    def qy(self):
        return self._data[5]

    @property
    def qz(self):
        return self._data[6]

    @property
    def quat(self):
        return np.array([self.qw, self.qx, self.qy, self.qz])

    @property
    def markers(self):
        return self._markers

    @property
    def mean_error(self):
        return self._mean_error

    def get_homomat(self):
        homomat = rm.quaternion_matrix(self.quat)
        homomat[:3, 3] = self.get_pos()
        return homomat

    def get_pos(self):
        return np.array([self.x, self.y, self.z])

    def get_rotmat(self):
        return self.get_homomat()[:3, :3]

    def set_markers(self, markers: np.ndarray):
        self._markers = markers

    def gen_mesh_model(self, radius=.005, rgba=np.array([1, 0, 0, 1])) -> mc.ModelCollection:
        """
        Plot markers for rigid body
        """
        markers_mc = mc.ModelCollection()
        for i in range(len(self)):
            gm.gen_sphere(self.markers[i], radius=radius, rgba=rgba).attach_to(markers_mc)
        return markers_mc

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


def markers_to_np(markers, n_markers) -> np.ndarray:
    """
    Author: Chen Hao, chen960216@gmail.com
    Date: 20220415
    Convert Raw Marker Data from Nokov Client to numpy array. Each row represents one marker's position
    https://stackoverflow.com/questions/4355524/getting-data-from-ctypes-array-into-numpy
    """
    marker_set_markers_pntr = ctypes.cast(markers, ctypes.POINTER(ctypes.c_float * 3))
    marker_set_markers_np = np.ctypeslib.as_array(marker_set_markers_pntr,
                                                  shape=(n_markers,)).copy() / 1000
    return marker_set_markers_np


def to_body_data(body: nokovsdk.RigidBodyData) -> RigidBodyData:
    """
    Author: Chen Hao, chen960216@gmail.com
    Date: 20220415
    Convert Raw RigidBody Data from Nokov Client to RigidBodyData
    """
    rigid_body_data = RigidBodyData(x=body.x / 1000,
                                    y=body.y / 1000,
                                    z=body.z / 1000,
                                    qx=body.qx,
                                    qy=body.qy,
                                    qz=body.qz,
                                    qw=body.qw,
                                    mean_error=body.MeanError)
    rigid_body_data.set_markers(markers_to_np(body.Markers, body.nMarkers))
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
    * When the rigid body is added (by Add/Remove) to the scene, it will always be shown in the retrieved data,
      even it is not in the scene.
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

        # get data description
        ph_data_description = ctypes.POINTER(nokovsdk.DataDescriptions)()
        self._client.PyGetDataDescriptions(ph_data_description)
        data_description = ph_data_description.contents
        self.rigidbody_relations, self.skeleton_relations = self._parse_data(data_description)

    def _get_data_frame(self, buffer: DataBuffer):
        data = buffer.get_last()
        if data is None:
            return None
        else:
            return data

    def get_rigidbody_set_frame(self) -> RigidBodyDataFrame:
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
                marker_set_dict[marker_set.szName] = markers_to_np(marker_set.Markers, marker_set.nMarkers)
            # read other marker information
            marker_set_dict["others"] = markers_to_np(frameData.OtherMarkers, frameData.nOtherMarkers)
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

    def _parse_data(self, data_description):
        rigidbody_relations = {}
        skeleton_relations = {}
        for i in range(data_description.nDataDescriptions):
            print(data_description.arrDataDescriptions[i].dump_dict()['type'])
            data = data_description.arrDataDescriptions[i].dump_dict()['Data']
            if data.RigidBodyDescription:
                rigidbody_description = data.RigidBodyDescription.contents.dump_dict()
                rigidbody_relations[rigidbody_description['szName']] = {'ID': rigidbody_description['ID'],
                                                                        'parentID': rigidbody_description['parentID'],
                                                                        'offsetx': rigidbody_description['offsetx'],
                                                                        'offsety': rigidbody_description['offsety'],
                                                                        'offsetz': rigidbody_description['offsetz']}
            if data.SkeletonDescription:
                skeleton_description = data.SkeletonDescription.contents.dump_dict()
                skeleton_relations[skeleton_description['szName']] = {'RigidBodies': skeleton_description['RigidBodies'],
                                                                      'nRigidBodies': skeleton_description['nRigidBodies'],
                                                                      'skeletonID': skeleton_description['skeletonID']}
        return rigidbody_relations, skeleton_relations


if __name__ == "__main__":
    # 160,
    server = NokovClient()
    while True:
        # data = server.get_rigidbody_set_frame()
        data = server.get_marker_set_frame()
        # print(data.rigidbody_set_dict[0])
        # data = server.get_marker_set_frame()
        if data is not None:
            print(data)
