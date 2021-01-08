import asyncio
import sys
import os
import time

READ_DEPTH = 1
READ_RGB = 2
READ_PTC = 3
READ_RGB_PTC = 4
server_opt = {
    1: 'READ_DEPTH',
    2: 'READ_RGB',
    3: 'READ_PTC',
    4: 'READ_RGB_PTC'
}

class AsyncServer:
    def __init__(self, address="localhost", port=8888):
        self.address = address
        self.port = port
        self.loop = asyncio.get_event_loop()
        self.disconnection_timeout = 1
        self.streamer = None

    async def server(self):
        server = await asyncio.start_server(
            self.handle_connection, self.address, self.port)

        addr = server.sockets[0].getsockname()
        print(f'Serving on {addr}')

        async with server:
            await server.serve_forever()

    async def handle_connection(self, reader, writer):
        print("-------- Connection successfully --------")
        disconnect_countdown_starttime = None
        while not writer.is_closing():
            try:
                data = await self.handle_pre()
                readout = await self.handle_read(reader, data)
                if readout is None:
                    if disconnect_countdown_starttime is None:
                        disconnect_countdown_starttime = time.time()
                    disconnect_countdown_endtime = time.time()
                    if disconnect_countdown_endtime - disconnect_countdown_starttime > self.disconnection_timeout:
                        del self.streamer
                        self.streamer = None
                        raise Exception(f"Do not receive data within {self.disconnection_timeout}s")
                    else:
                        continue
                disconnect_countdown_starttime = None
                await self.handle_write(writer, data, readout)
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(e, fname, exc_tb.tb_lineno)
                del self.streamer
                self.streamer = None
                writer.close()
                if not writer.is_closing():
                    await writer.wait_closed()
                break
        print(f"Connection Break: -- {writer.is_closing()}")

    async def handle_pre(self):
        pass

    async def handle_read(self, reader, data=None):
        pass

    async def handle_write(self, writer, data=None, readout=None):
        pass

    def start_server(self):
        self.loop.run_until_complete(self.server())
        self.loop.close()


class AsyncClient:
    def __init__(self, address="localhost", port=8888):
        self.address = address
        self.port = port
        self.loop = asyncio.get_event_loop()
        self.reader = None
        self.writer = None
        self.loop = asyncio.get_event_loop()
        self.loop.run_until_complete(self.create_connection())

    async def create_connection(self):
        reader, writer = await asyncio.open_connection(
            self.address, self.port)
        self.reader = reader
        self.writer = writer

    def runtask(self, task, *q, **kwargs):
        self.loop.run_until_complete(task(*q, **kwargs))

    def close(self):
        print('Close the connection')
        if self.writer is not None:
            self.writer.close()
        self.loop.close()
