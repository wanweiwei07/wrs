import async_base as ab
import cv2
import pickle
import struct
import numpy as np

class EtherSenseClient(ab.AsyncClient):
    def __init__(self, address, port):
        super().__init__(address=address, port=port)
        self.remainingBytes = 0
        self.buffer = bytearray()
        self._return_value = None

    def get_depth_image(self):
        """
        :return: depthimage
        author: weiwei
        date: 20210108
        """
        self.runtask(self._get_depth_image)
        return self._return_value

    def get_rgb_image(self):
        """
        :return: rgb image
        author: weiwei
        date: 20210108
        """
        self.runtask(self._get_rgb_image)
        return self._return_value

    def get_pointcloud(self):
        """
        :return: np array
        author: weiwei
        date: 20210108
        """
        self.runtask(self._get_pointcloud)
        return self._return_value

    def get_rgb_pointcloud(self):
        """
        :return: np array
        author: weiwei
        date: 20210108
        """
        self.runtask(self._get_rgb_pointcloud)
        return self._return_value

    async def _get_depth_image(self):
        await self.handle_write(str(ab.READ_DEPTH))
        await self.handle_read(self.handle_img_frame)

    async def _get_rgb_image(self):
        await self.handle_write(str(ab.READ_RGB))
        await self.handle_read(self.handle_img_frame)

    async def _get_pointcloud(self):
        await self.handle_write(str(ab.READ_PTC))
        await self.handle_read(self.handle_pointcloud_frame)

    async def _get_rgb_pointcloud(self):
        await self.handle_write(str(ab.READ_RGB_PTC))
        await self.handle_read(self.handle_rgb_pointcloud_frame)

    async def handle_write(self, message):
        if self.writer is not None:
            self.writer.write(message.encode())
            await self.writer.drain()

    async def handle_read(self, callback):
        # get the expected frame size
        self.frame_length = struct.unpack('<I', await self.reader.read(4))[0]
        self.timestamp = struct.unpack('<d', await self.reader.read(8))
        self.remainingBytes = self.frame_length
        while self.remainingBytes != 0:
            # request the frame data until the frame is completely in buffer
            data = await self.reader.read(self.remainingBytes)
            self.buffer += data
            self.remainingBytes -= len(data)
            # once the frame is fully recived, process/display it
        callback()

    def handle_img_frame(self):
        # convert the frame from string to numerical data
        imdata = pickle.loads(self.buffer)
        self._return_value = cv2.resize(imdata, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        self.buffer = bytearray()

    def handle_pointcloud_frame(self):
        # convert the frame from string to numerical data
        imdata = pickle.loads(self.buffer)
        self._return_value = np.frombuffer(imdata, dtype=np.float32).reshape(-1,3)
        self.buffer = bytearray()

    def handle_rgb_pointcloud_frame(self):
        # convert the frame from string to numerical data
        imdata = pickle.loads(self.buffer)
        self._return_value = np.frombuffer(imdata, dtype=np.float32).reshape(-1,6)
        self._return_value[:, 3:] = self._return_value[:, 3:]/255.0 # regulate to 0-1
        self.buffer = bytearray()

if __name__ == '__main__':

    client = EtherSenseClient(address="10.2.0.202", port=18360)
    cv2.namedWindow("window")
    while True:
        cv2.imshow("window", client.get_rgb_image())
        cv2.waitKey(1)
    # time.sleep(3)
    # cv2.imshow("window2", client.get_rgb_image())
    # cv2.waitKey(1)
    # time.sleep(3)

    import wrs.visualization.panda.rpc.rviz_client as rv_client
    from wrs import modeling as gm

    rvc = rv_client.RVizClient(host="localhost:182001")
    rvc.reset()
    last_rmt= None
    while True:
        pcd = gm.GeometricModel(initor=client.get_rgb_pointcloud())
        current_rmt = rvc.showmodel_to_remote(pcd)
        if last_rmt is not None:
            rvc.unshow_model(last_rmt)
        last_rmt = current_rmt
    base.run()
