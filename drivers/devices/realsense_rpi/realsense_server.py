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
        print("New data is ", data)
        command = int(data)
        if command not in ab.server_opt:
            print("Command cannot be recognized")
            return None
        return command

    async def handle_write(self, writer, data=None, readout=None):
        print("START WRITING")
        if readout is not None and not writer.is_closing():
            if self.streamer is None:
                if readout == ab.READ_RGB_IMG:
                    self.streamer = self.rsc.getRGBImgStream()
                elif readout == ab.READ_DEPTH_IMG:
                    self.streamer = self.rsc.getDepthImgStream()
                elif readout == ab.READ_PTC:
                    self.streamer = self.rsc.getPointCloudStream()
                else:
                    raise Exception("Unknown Error")

            # if not hasattr(self.rsc, 'frame_data'):
            #     self.streamer.update_frame()
            # # the frame has been sent in it entirety so get the latest frame
            if len(self.streamer.frame_data) == 0:
                self.streamer.update_frame()
                # print(self.streamer.frame_data)
                writer.write(self.streamer.frame_data)
                await writer.drain()
                self.streamer.frame_data = b''
        print("ENDWRITING")

        # else:
        #     # send the remainder of the frame_data until there is no data remaining for transmition
        #     remaining_size = self.send(self.frame_data)
        #     self.frame_data = self.frame_data[remaining_size:]


server = EtherSenseServer(address="localhost", port=8888)
server.start_server()
