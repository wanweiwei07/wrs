import grpc
import time
import numpy as np
from concurrent import futures
import visualization.panda.rpc.rviz_pb2 as rv_msg
import visualization.panda.rpc.rviz_pb2_grpc as rv_rpc


class RVizClient(object):

    def __init__(self, host="localhost:18300"):
        channel = grpc.insecure_channel(host)
        self.stub = rv_rpc.RVizStub(channel)

    def run_code(self, code):
        """
        :param code: string
        :return:
        author: weiwei
        date: 20201229
        """
        code_bytes = code.encode('utf-8')
        return_val = self.stub.run_code(rv_msg.CodeRequest(code=code_bytes)).value
        if return_val == rv_msg.Status.ERROR:
            print("Something went wrong with the server!! Try again!")
            raise Exception()
        else:
            return
            print("The given code is succesfully executed.")
