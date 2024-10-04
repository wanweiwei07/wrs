import numpy as np
import pyrealsense2 as rs
import pickle
import struct
import cv2

# STREAM TYPE
STREAM_RGB = 1
STREAM_DEPTH = 2
STREAM_PTC = 3
STREAM_RGBPTC = 4


class RealSenseController:

    def __init__(self):
        self.pipeline, self.streamcfg = self.openPipeline()
        self.colorizer = rs.colorizer()
        self.pc = rs.pointcloud()
        self.releaselist = []

    def disableAllConfiguration(self):
        self.streamcfg.disable_all_streams()

    def useRGBCamera(self):
        self.streamcfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    def useDepthCamera(self):
        self.streamcfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    def stop(self):
        try:
            self.pipeline.stop()
        except:
            pass

    def start(self):
        pipeline_profile = self.pipeline.start(self.streamcfg)

    def openPipeline(self):
        cfg = rs.config()
        pipeline = rs.pipeline()
        # pipeline_profile = pipeline.start(cfg)
        # sensor = pipeline_profile.get_device().first_depth_sensor()
        return pipeline, cfg

    def getRGBImgStream(self):
        self.disableAllConfiguration()
        self.stop()
        self.useRGBCamera()
        self.start()
        stream = RealSenseStreamer(pipeline=self.pipeline, streamertype=STREAM_RGB)
        self.releaselist.append(stream)
        return stream

    def getDepthImgStream(self):
        self.disableAllConfiguration()
        self.stop()
        self.useDepthCamera()
        self.start()
        stream = RealSenseStreamer(pipeline=self.pipeline, streamertype=STREAM_DEPTH)
        self.releaselist.append(stream)
        return stream

    def getPointCloudStream(self):
        self.disableAllConfiguration()
        self.stop()
        self.useDepthCamera()
        self.start()
        stream = RealSenseStreamer(pipeline=self.pipeline, streamertype=STREAM_PTC, pc=self.pc)
        self.releaselist.append(stream)
        return stream

    def getRGBPointCloudStream(self):
        self.disableAllConfiguration()
        self.stop()
        self.useDepthCamera()
        self.useRGBCamera()
        self.start()
        stream = RealSenseStreamer(pipeline=self.pipeline, streamertype=STREAM_RGBPTC, pc=self.pc,
                                   colorizer=self.colorizer)
        self.releaselist.append(stream)
        return stream

    def releaseStreamer(self):
        if len(self.releaselist) == 0:
            self.stop()


class RealSenseStreamer:
    """
    The Streamer should be released by user
    """

    def __init__(self, pipeline, streamertype=STREAM_RGB, pc=None, colorizer=None):
        self.pipeline = pipeline
        self.streamtype = streamertype
        self.decimate_filter = rs.decimation_filter()
        decimate = 1
        self.decimate_filter.set_option(rs.option.filter_magnitude, 2 ** decimate)
        self.colorizer = colorizer
        self.pc = pc
        self.frame_data = ''

    def update_frame(self):
        depth, timestamp = self._getStreamAndTimestamp()
        if depth is not None:
            # convert the depth image to a string for broadcast
            data = pickle.dumps(depth)
            # capture the lenght of the data portion of the message
            length = struct.pack('<I', len(data))
            # include the current timestamp for the frame
            ts = struct.pack('<d', timestamp)
            # for the message for transmission
            self.frame_data = length + ts + data

    def test(self):
        depth, timestamp = self._getStreamAndTimestamp()
        return depth, timestamp

    def _getStreamAndTimestamp(self):
        frames = self.pipeline.wait_for_frames()
        # take owner ship of the frame for further processing
        # frames.keep()
        color_frame = None
        depth_frame = None
        ts = frames.get_timestamp()
        if self.streamtype == STREAM_RGB or self.streamtype == STREAM_RGBPTC:
            color_frame = frames.get_color_frame()
        if self.streamtype == STREAM_DEPTH or self.streamtype == STREAM_PTC or self.streamtype == STREAM_RGBPTC:
            depth_frame = frames.get_depth_frame()
        if self.streamtype == STREAM_RGB:
            return np.asanyarray(color_frame.get_data()), ts
        if self.streamtype == STREAM_DEPTH:
            return np.asanyarray(depth_frame.get_data()), ts
        if self.streamtype == STREAM_PTC:
            depth_frame = self.decimate_filter.process(depth_frame)
            points = self.pc.calculate(depth_frame)
            v, t = points.get_vertices(), points.get_texture_coordinates()
            verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz
            return verts, ts
        if self.streamtype == STREAM_RGBPTC:
            depth_frame = self.decimate_filter.process(depth_frame)
            points = self.pc.calculate(depth_frame)
            v, t = points.get_vertices(), points.get_texture_coordinates()
            verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz
            texcoords = np.asanyarray(t).view(np.float32).reshape(-1, 2)  # uv
            tex0 = np.rint(texcoords[:,0]*631).astype(np.int32)
            tex1 = np.rint(texcoords[:,1]*479).astype(np.int32)
            color_array = np.asanyarray(color_frame.get_data())
            rgb = color_array[tex0, tex1, :]#xyzrgb
            return np.hstack((verts, rgb)), ts
        return None, None


if __name__ == "__main__":
    import cv2

    # test = RealSenseController().getRGBImgStream()
    ctl = RealSenseController()
    # while True:
    # test = ctl.getDepthImgStream()sudo rm pyrealsense*
    # depthimg, timeframe1 = test.test()
    # depthcolormap = cv2.applyColorMap(cv2.convertScaleAbs(depthimg, alpha=0.03), cv2.COLORMAP_JET)
    #
    # test = ctl.getRGBImgStream()
    # colorimg, timeframe2 = test.test()
    # images = np.hstack((colorimg, depthcolormap))
    # print(timeframe1-timeframe2)
    # cv2.imshow("w",images)
    # cv2.waitKey(0)

    test = ctl.getPointCloudStream().test()
    print(test)
