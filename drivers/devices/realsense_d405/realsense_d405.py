"""
Interface for Realsense D405. Datasheet: https://dev.intelrealsense.com/docs/intel-realsense-d400-series-product-family-datasheet
Author: Chen Hao (chen960216@gmail.com), 20220719, osaka
Requirement libs: 'pyrealsense2', 'numpy'
Importance: This program needs to connect to USB3 to work
"""
import time
import multiprocessing as mp

import numpy as np
import pyrealsense2 as rs

__VERSION__ = '0.0.1'
PROCESS_SLEEP_TIME = .1
# Read chapter 4 of datasheet for details
DEPTH_RESOLUTION = (1280, 720)
COLOR_RESOLUTION = (1280, 720)
DEPTH_FPS = 30
COLOR_FPS = 30


def stream_data(pipe: rs.pipeline, pc: rs.pointcloud):
    """
    Stream data for RealSense D504
    """
    # Acquire a frame
    frames = pipe.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    # get depth and color image
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    # Calculate point clouds and color textures for the point clouds
    points = pc.calculate(depth_frame)
    pc.map_to(color_frame)
    v, t = points.get_vertices(), points.get_texture_coordinates()
    verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz
    texcoords = np.asanyarray(t).view(np.float32).reshape(-1, 2)  # uv
    # Calculate normalized colors (rgb nx3) for the point cloud
    cw, ch = color_image.shape[:2][::-1]
    v, u = (texcoords * (cw, ch) + 0.5).astype(np.uint32).T
    np.clip(u, 0, ch - 1, out=u)
    np.clip(v, 0, cw - 1, out=v)
    pc_color = color_image[u, v] / 255
    pc_color[:, [0, 2]] = pc_color[:, [2, 0]]
    return (verts, pc_color, depth_image, color_image)


class _DataPipeline(mp.Process):
    """
    The process to stream data through Realsense API
    """

    def __init__(self, req_q: mp.Queue,
                 res_q: mp.Queue, ):
        mp.Process.__init__(self)
        # Require queue and receive queue to exchange data
        self._req_q = req_q
        self._res_q = res_q

    def run(self):
        # RealSense pipeline, encapsulating the actual device and sensors
        pipeline = rs.pipeline()
        config = rs.config()
        # Setup config
        config.enable_stream(rs.stream.depth, DEPTH_RESOLUTION[0], DEPTH_RESOLUTION[1], rs.format.z16, DEPTH_FPS)
        config.enable_stream(rs.stream.color, COLOR_RESOLUTION[0], COLOR_RESOLUTION[1], rs.format.bgr8, COLOR_FPS)

        # Start streaming with chosen configuration
        pipeline.start(config)

        # Declare pointcloud object, for calculating pointclouds and texture mappings
        pc = rs.pointcloud()

        # Streaming
        while True:
            req_packet = self._req_q.get()
            if req_packet == "stop":
                break
            self._res_q.put(stream_data(pipe=pipeline, pc=pc))
            time.sleep(PROCESS_SLEEP_TIME)
        pipeline.stop()


class RealSenseD405(object):
    def __init__(self, toggle_new_process: bool = False):
        """
        :param toggle_new_process: Open a new process to stream data
        """
        self._toogle_new_process = toggle_new_process
        if toggle_new_process:
            self._req_q = mp.Queue()  # queue to require data
            self._res_q = mp.Queue()  # queue to receive data
            self._pipeline = _DataPipeline(req_q=self._req_q,
                                           res_q=self._res_q, )
            self._pipeline.start()
        else:
            self._pipeline = rs.pipeline()
            self._config = rs.config()
            # Setup config
            self._config.enable_stream(rs.stream.depth, DEPTH_RESOLUTION[0], DEPTH_RESOLUTION[1], rs.format.z16,
                                       DEPTH_FPS)
            self._config.enable_stream(rs.stream.color, COLOR_RESOLUTION[0], COLOR_RESOLUTION[1], rs.format.bgr8,
                                       COLOR_FPS)
            # Start streaming with chosen configuration
            self._pipeline.start(self._config)
            # Declare pointcloud object, for calculating pointclouds and texture mappings
            self._pc = rs.pointcloud()

    def req_data(self):
        """
        Require 1) point cloud, 2) point cloud color, 3) depth image and 4) color image
        :return: List[np.array, np.array, np.array, np.array]
        """
        if self._toogle_new_process:
            self._req_q.put('')
            return self._res_q.get()
        else:
            return stream_data(pipe=self._pipeline, pc=self._pc)

    def get_pcd(self, return_color=False):
        """
        Get point cloud data. If return_color is True, additionally return pcd color
        :return: nx3 np.array
        """
        pcd, pcd_color, depth_img, color_img = self.req_data()
        if return_color:
            return pcd, pcd_color
        return pcd

    def get_color_img(self):
        """
        Get color image
        :return:
        """
        pcd, pcd_color, depth_img, color_img = self.req_data()
        return color_img

    def get_depth_img(self):
        """
        Get depth image
        :return:
        """
        pcd, pcd_color, depth_img, color_img = self.req_data()
        return depth_img

    def get_pcd_color_depth(self):
        """
        Return pcd, pcd_color, depth image and color image
        :return: List[np.array, np.array, np.array, np.array]
        """
        return self.req_data()

    def stop(self):
        '''
        Stops subprocess for ethernet communication. Allows program to exit gracefully.
        '''
        if self._toogle_new_process:
            self._req_q.put("stop")
            self._pipeline.terminate()
        else:
            self._pipeline.stop()

    def __del__(self):
        self.stop()


if __name__ == "__main__":
    import cv2

    rs_pipe = RealSenseD405(toggle_new_process=True)

    while True:
        pcd, pcd_color, depth_img, color_img = rs_pipe.get_pcd_color_depth()
        print(color_img.shape)
        print(pcd.shape)
        cv2.imshow("color image", color_img)
        k = cv2.waitKey(1)
        if k == 27:
            break

    rs_pipe.stop()
