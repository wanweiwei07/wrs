import numpy as np
import pyrealsense2 as rs
import pickle
import struct
import async_base as ab
import realsense_controller as rsc


class EtherSenseServer(ab.AsyncServer):
    def __init__(self, address="localhost", port=8888):
        super().__init__(address=address, port=port)
        self.rsc = rsc.RealSenseController()

    async def handle_read(self, reader, data=None):
        data = await reader.read(1)
        if len(data) < 1:
            return None
        command = int(data)
        if command not in ab.server_opt:
            print("Command cannot be recognized")
            return None
        return command

    async def handle_write(self, writer, data=None, readout=None):
        if readout is not None and not writer.is_closing():
            if self.streamer is None:
                if readout == ab.READ_RGB:
                    self.streamer = self.rsc.getRGBImgStream()
                elif readout == ab.READ_DEPTH:
                    self.streamer = self.rsc.getDepthImgStream()
                elif readout == ab.READ_PTC:
                    self.streamer = self.rsc.getPointCloudStream()
                elif readout == ab.READ_RGB_PTC:
                    self.streamer = self.rsc.getRGBPointCloudStream()
                else:
                    raise Exception("Unknown Error")

            if len(self.streamer.frame_data) == 0:
                self.streamer.update_frame()
                writer.write(self.streamer.frame_data)
                await writer.drain()
                self.streamer.frame_data = b''


server = EtherSenseServer(address="192.168.11.8", port=18360)
server.start_server()
