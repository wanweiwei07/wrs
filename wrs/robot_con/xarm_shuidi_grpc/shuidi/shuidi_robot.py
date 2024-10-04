import json
import socket
import time


class ShuidiRobot(object):
    """
    Wrapper for controlling Yunjin Shuidi2 robot_s using Shuidi's string API
    API Webpage, http://waterdocs.pages.yunjichina.com.cn/user_manual/exports/water_api.html
    author: hao, revised by weiwei
    date: 20210329
    """

    def __init__(self, ip="192.168.10.10", port="31001"):
        """
        :param ip:
        :param port:
        """
        self._address = (ip, port)
        self._socket = socket.create_connection(self._address, timeout=2)
        print(f"Connection on {self._address}")

    def data_send_recv(self, data):
        self._socket.send(data.encode())
        recv_data_raw = self._socket.recv(4096)
        for data in recv_data_raw.decode().split("\n"):
            trimmed_recv_data = json.loads(data)
            if trimmed_recv_data["end_type"] == "response":
                return trimmed_recv_data

    def data_recv(self):
        recv_data_raw = self._socket.recv(1024)
        recv_data = json.load(recv_data_raw)
        return recv_data

    def move_to_marker(self, target_name, uuid=1, max_continuous_retries=30, distance_tolerance=None,
                       theta_tolerance=None, angle_offset=0, yaw_goal_reverse_allowed=-1, occupied_tolerance=None):
        """
        移动到目标代号
        :param target_name: 目标点位代号
        :param uuid:
        :param max_continuous_retries: 原地最大连续重试次数（机器人原地不动时，重试次数超过此值则任务失败)
        :param distance_tolerance: 距离容差,类型float,单位米。(当目标位置被占据等原因无法到达时，机器人移动到目标此距离之内也算任务成功。)
        :param theta_tolerance: 角度容差，类型float，单位弧度。(到达目标点位后，角度小于此值后任务成功)
        :param angle_offset: 到达位置后的角度偏移, 例如使用marker=m1发送任务时,会以m1的角度+angle_offset的角度作为最终方向执行任务.
        :param yaw_goal_reverse_allowed:双向停靠控制参数，取值1或0或-1。对于双向行走的机器人，此参数用于机器人停靠到点位时，是否允许尾部跟点位方向一致。
        :param occupied_tolerance: 让步停靠距离参数，单位米。 当目标点位被占用时，设置此参数机器人会直接在点位附近停靠以完成任务，而不再尝试移动到点位上。距离以占用物边缘至机器人中心计算。
        :return:
        """
        command = ShuidiRobot.generate_request_data(uuid, "move",
                                                    {"marker": target_name,
                                                     "max_continuous_retries": max_continuous_retries,
                                                     "distance_tolerance": distance_tolerance,
                                                     "theta_tolerance": theta_tolerance,
                                                     "angle_offset": angle_offset,
                                                     "yaw_goal_reverse_allowed": yaw_goal_reverse_allowed,
                                                     "occupied_tolerance": occupied_tolerance})
        return self.data_send_recv(command)

    def move_to_location(self, x, y, theta, uuid=1, max_continuous_retries=30, distance_tolerance=None,
                         theta_tolerance=None, angle_offset=0, yaw_goal_reverse_allowed=-1, occupied_tolerance=None):
        """
        :param x: x<地图中x轴坐标>
        :param y: y<地图中y轴坐标>
        :param theta: theta<地图中相对theta值>
        :param uuid:
        :param max_continuous_retries: 原地最大连续重试次数（机器人原地不动时，重试次数超过此值则任务失败)
        :param distance_tolerance: 距离容差,类型float,单位米。(当目标位置被占据等原因无法到达时，机器人移动到目标此距离之内也算任务成功。)
        :param theta_tolerance: 角度容差，类型float，单位弧度。(到达目标点位后，角度小于此值后任务成功)
        :param angle_offset: 到达位置后的角度偏移, 例如使用marker=m1发送任务时,会以m1的角度+angle_offset的角度作为最终方向执行任务.
        :param yaw_goal_reverse_allowed:双向停靠控制参数，取值1或0或-1。对于双向行走的机器人，此参数用于机器人停靠到点位时，是否允许尾部跟点位方向一致。
        :param occupied_tolerance: 让步停靠距离参数，单位米。 当目标点位被占用时，设置此参数机器人会直接在点位附近停靠以完成任务，而不再尝试移动到点位上。距离以占用物边缘至机器人中心计算。
        :return:
        """
        command = ShuidiRobot.generate_request_data(uuid, "move",
                                                    {"location": f"{x},{y},{theta}",
                                                     "max_continuous_retries": max_continuous_retries,
                                                     "distance_tolerance": distance_tolerance,
                                                     "theta_tolerance": theta_tolerance,
                                                     "angle_offset": angle_offset,
                                                     "yaw_goal_reverse_allowed": yaw_goal_reverse_allowed,
                                                     "occupied_tolerance": occupied_tolerance})
        return self.data_send_recv(command)

    def move_to_markers(self, markers: list, uuid=1, max_continuous_retries=5, distance_tolerance=0.5, count=1):
        """
        多目标点移动
        :param markers 想要巡游的点位列表 e.g: ["m1","m2","m3"]
        :param uuid:
        :param max_continuous_retries: 原地最大连续重试次数（机器人原地不动时，重试次数超过此值则任务失败)
        :param distance_tolerance: 距离容差,类型float,单位米。(当目标位置被占据等原因无法到达时，机器人移动到目标此距离之内也算任务成功。)
        :param count: 巡游的次数，所有点位走过一遍之后计为一次, 不选时为默认一次，-1表示无限循环。
        :return:
        """
        command = ShuidiRobot.generate_request_data(uuid, "move",
                                                    {"markers": ",".join(markers),
                                                     "max_continuous_retries": max_continuous_retries,
                                                     "distance_tolerance": distance_tolerance,
                                                     "n_sec_minor": count})
        return self.data_send_recv(command)

    def cancel_move(self, uuid=1):
        """
        使机器人主动放弃当前正在执行的移动任务，成功取消后可使机器人进入新的待命状态。
        在机器人执行接口1-机器人移动命令过程中，如果需要终止机器人当前的移动状态，可以调用此接口。机器人会在接收“移动取消”命令之后，原地停止，等待再次的move指令。
        取消当前正在进行的移动指令
        """
        command = ShuidiRobot.generate_request_data(uuid, "move/cancel")
        return self.data_send_recv(command)

    def robot_status(self, uuid=1):
        """
        获取机器人当前全局状态，包括移动任务的状态。
        除配合接口1周期监听move的反馈之外，也可从此接口中监听机器人整机的其他信息，
        包括“是否处于充电状态”、“是否处于急停状态”、“所剩电池点量百分比”、“当前相对于地图的位置”、“当前楼层”，具体见示例。
        建议调用频率为1-2HZ，可以实时监控任务状态，作为流程控制的逻辑判断。
        """
        command = ShuidiRobot.generate_request_data(uuid, "robot_status")
        return self.data_send_recv(command)

    def robot_info(self, uuid=1):
        """
        调用接口可以获取机器人的一些基本信息。
        """
        command = ShuidiRobot.generate_request_data(uuid, "robot_info")
        return self.data_send_recv(command)

    def marker_insert(self, name: str, uuid=1, type=0, num=1):
        """
        :param name:点位名字(string类型,不支持特殊字符), 如果name已经存在，则更新坐标。
        :param uuid:
        :param type:点位类型(int类型), 常用类型有：0(一般点位)，1(前台点)，7(闸机),3(电梯外),4(电梯内),11(充电桩)等等。
        每种类型的点位都具有特定的功能和各自的属性，除普通类型外其他类型请尽可能使用机器人监控页面进行添加，
        避免属性值异常造成程序运行的异常。
        在通常情况下不建议用户自定义类型，
        如果使用自定义类型请使用大于1000的值，
        以免跟机器人定义的类型产生冲突。
        随着机器人软件版本不同后续可能会增加新的类型以及类型对应的属性，请以监控页面建图工具里的标记点位为准。
        :param num: 点位编号(int类型) 某些类型的点位具有num(编号)属性，例如电梯点,闸机点，充电桩点等。
        :return:
        """

        command = ShuidiRobot.generate_request_data(uuid, "markers/insert",
                                                    {"name": name,
                                                     "end_type": type,
                                                     "num": num})
        return self.data_send_recv(command)

    def marker_insert_by_pose(self, x: float, y: float, theta: float, name: str, uuid=1, type=0, num=1,
                              floor: int = None):
        """
        :param x: 地图坐标x(float类型)
        :param y: 地图坐标y(float类型)
        :param theta: 点位的方向(float类型), 	取值范围[-π, π]
        :param name:点位名字(string类型,不支持特殊字符), 如果name已经存在，则更新坐标。
        :param uuid:
        :param type:点位类型(int类型), 常用类型有：0(一般点位)，1(前台点)，7(闸机),3(电梯外),4(电梯内),11(充电桩)等等。
        每种类型的点位都具有特定的功能和各自的属性，除普通类型外其他类型请尽可能使用机器人监控页面进行添加，
        避免属性值异常造成程序运行的异常。
        在通常情况下不建议用户自定义类型，
        如果使用自定义类型请使用大于1000的值，
        以免跟机器人定义的类型产生冲突。
        随着机器人软件版本不同后续可能会增加新的类型以及类型对应的属性，请以监控页面建图工具里的标记点位为准。
        :param num: 点位编号(int类型) 某些类型的点位具有num(编号)属性，例如电梯点,闸机点，充电桩点等。
        :param floor: 楼层(int类型非0), 默认为机器人当前楼层,如果楼层不存在则会返回错误.
        :return:
        """
        command = ShuidiRobot.generate_request_data(uuid, "markers/insert_by_pose",
                                                    {"x": x,
                                                     "y": y,
                                                     "theta": theta,
                                                     "floor": floor,
                                                     "name": name,
                                                     "end_type": type,
                                                     "num": num})
        return self.data_send_recv(command)

    def marker_query_list(self, floor=None, uuid=1):
        """
        获取机器人在当前地图中的所有点位(marker)信息,
        每个点位信息中包括”点位名称”、”楼层”、”点位坐标”、”点位方向”和”点位类型”。
        其中orientation以四元数的形式保存点位的方向，
        可以转化成前文阐述的坐标系中的theta
        :param floor: 想要查询的楼层, 如果不选则返回所有楼层的点位信息
        :param uuid:
        :return:
        """
        command = ShuidiRobot.generate_request_data(uuid, "markers/query_list", {"floor": floor})
        return self.data_send_recv(command)

    def marker_query_brief(self, uuid=1):
        """
        查询当前地图所有点位的摘要信息，此接口返回比marker_query_list获取marker点位列表更加简洁的点位信息。
        :return:
        """
        command = ShuidiRobot.enerate_request_data(uuid, "markers/query_brief")
        return self.data_send_recv(command)

    def marker_delete(self, name, uuid=1):
        """
        删除已经标记的marker点位，如果点位名称不存在则返回失败。
        :param name: 点位名字
        :param uuid:
        :return:
        """
        command = ShuidiRobot.generate_request_data(uuid, "markers/delete", {"name": name})
        return self.data_send_recv(command)

    def marker_count(self, uuid=1):
        """
        获取当前地图中的点位数量。
        :return:
        """
        command = ShuidiRobot.generate_request_data(uuid, "markers/n_sec_minor")
        return self.data_send_recv(command)

    def joy_control(self, uuid=1, linear_velocity=0, angular_velocity=0):
        """
        对机器人进行部分的直接控制，如自转或停止（此控制优先级高于move指令），
        返回succeeded表示机器人已成功接收并开始运行此指令
        :return:
        """
        command = ShuidiRobot.generate_request_data(uuid, "joy_control",
                                                    {"angular_velocity": angular_velocity,
                                                     "linear_velocity": linear_velocity})
        return self.data_send_recv(command)

    def estop(self, flag=False, uuid=1):
        """
        使机器人进入自由停止模式，机器人可被推动。
        :return:
        """
        command = ShuidiRobot.generate_request_data(uuid, "estop",
                                                    {"flag": "true" if flag else "false"})
        return self.data_send_recv(command)

    def position_adjust(self, marker: str, uuid=1):
        """
        使用此接口可将机器人位置校正到marker所标记的位置。
        使用时可先将机器人推至marker标记的位置，然后用调用此接口进行位置校正。
        :param marker: 用以校定位置的marker名 (已经标定的marker点)
        :param uuid:
        :return:
        """
        command = ShuidiRobot.generate_request_data(uuid, "position_adjust", {"marker": marker})
        return self.data_send_recv(command)

    def position_adjust_by_pose(self, x: float, y: float, theta: float, floor=None, uuid=1):
        """
        :param x: 地图坐标x(float类型)
        :param y: 地图坐标y(float类型)
        :param theta: 点位的方向(float类型),
        :param floor: 楼层 当前地图中存在的楼层	不填则默认为机器人当前楼层
        :param uuid:
        :return:
        """
        command = ShuidiRobot.generate_request_data(uuid, "position_adjust_by_pose",
                                                    {"x": x,
                                                     "y": y,
                                                     "theta": theta,
                                                     "floor": floor})
        return self.data_send_recv(command)

    def request_data(self, topic="robot_status", frequency=2, uuid=1):
        """
        请求server端以一定频率发送topic类型的数据。当请求成功后，server端会以一定频率发送数据给请求的client。
        :param topic: 请求的实时数据类型
        robot_status(请求的实时数据类型)/human_detection(人腿识别模块)/robot_velocity(请求的实时数据类型)
        :param frequency: 发送频率 默认2HZ
        :return:
        """
        command = ShuidiRobot.generate_request_data(uuid, "request_data",
                                                    {"topic": topic,
                                                     "frequency": frequency})
        return self.data_send_recv(command)

    def set_params(self, max_speed=None, max_speed_ratio=None, max_speed_linear=None, max_speed_angular=None, uuid=1):
        """
        :param max_speed: 机器人最大行进速度(百分比)  可选[0.3, 0.7]	小于0.3取0.3，大于0.7取0.7	v0.4.2-v0.5.1（已弃用）
        :param max_speed_ratio: 机器人最大行进速度百分比  可选[0.3, 1.4]	小于0.3取0.3，大于1.4取1.4	v0.5.2-v0.8.5.1(已弃用)
        :param max_speed_linear: max_speed_linear  机器人最大直线速度  可选[0.1, 1.0] (m/s)  小于0.1取0.1，大于1.0取1.0
        :param max_speed_angular: max_speed_angular  机器人最大角速度  可选[0.5, 3.5] (rad/s)  小于0.5取0.5，大于3.5取3.5
        :param uuid:
        :return:
        """
        command = ShuidiRobot.generate_request_data(uuid, "set_params",
                                                    {"max_speed": max_speed,
                                                     "max_speed_ratio": max_speed_ratio,
                                                     "max_speed_linear": max_speed_linear,
                                                     "max_speed_angular": max_speed_angular})
        return self.data_send_recv(command)

    def get_params(self, uuid=1):
        """
        获取已设置的参数列表以及当前值。各版本支持的参数见set_params 设置参数。
        """
        command = ShuidiRobot.generate_request_data(uuid, "get_params")
        return self.data_send_recv(command)

    def wifi_list(self, uuid=1):
        """
        获取机器人当前可用的WiFi列表，返回中包含SSID和信号强度。
        """
        command = ShuidiRobot.generate_request_data(uuid, "wifi/list")
        return self.data_send_recv(command)

    def wifi_connect(self, SSID, password=None, uuid=1):
        """
        :param SSID: WiFi的SSID  必选 当前的环境WiFi
        :param password: WiFi密码  可选 SSID对应 如果已经连接过，可以不填
        :param uuid:
        :return:
        """
        command = ShuidiRobot.generate_request_data(uuid, "wifi/connect",
                                                    {"SSID": SSID, "password": password})
        return self.data_send_recv(command)

    def wifi_get_active_connection(self, uuid=1):
        """
        获取机器人当前连接的WiFi的SSID.
        """
        command = ShuidiRobot.generate_request_data(uuid, "wifi/get_active_connection")
        return self.data_send_recv(command)

    def wifi_info(self, uuid=1):
        """
        获取机器人当前通过环境WiFi分配到的IP地址和无线网卡的物理地址。
        """
        command = ShuidiRobot.generate_request_data(uuid, "wifi/info")
        return self.data_send_recv(command)

    def wifi_detail_list(self, uuid=1):
        """
        获取机器人当前可用的WiFi列表详细信息。
        """
        command = ShuidiRobot.generate_request_data(uuid, "wifi/detail_list")
        return self.data_send_recv(command)

    def map_list(self, uuid=1):
        """
        获取机器人中所有的地图名称和楼层。
        """
        command = ShuidiRobot.generate_request_data(uuid, "map/list")
        return self.data_send_recv(command)

    def map_list_info(self, uuid=1):
        """
        获取机器人中所有的地图的详细信息。
        """
        command = ShuidiRobot.enerate_request_data(uuid, "map/list_info")
        return self.data_send_recv(command)

    def map_set_current_map(self, map_name, floor, uuid=1):
        """
        设置机器人当前地图。
        注:设置成功后会重启water服务，所以有可能收不到response。
        :param map_name: map_name  地图名 必选
        :param floor: floor	楼层	必选
        :param uuid:
        :return:
        """
        command = ShuidiRobot.generate_request_data(uuid, "map/set_current_map",
                                                    {"map_name": map_name, "floor": floor})
        return self.data_send_recv(command)

    def map_get_current_map(self, uuid=1):
        """
        获取机器人当前地图。
        """
        command = ShuidiRobot.generate_request_data(uuid, "map/get_current_map")
        return self.data_send_recv(command)

    def shutdown(self, reboot=False, delay=0, uuid=1):
        """
        调用接口关闭或者重新启动机器人,在电源关闭之前会有通知发出(见接口10 机器人主动通知)，在通知发出后10s电源关闭。如果是重启的话, 重新上电与电源关闭间会有5s间隔。
        注:可能会收不到response。
        :param reboot: reboot	关机后是否重启	可选	true/false	缺省为false	v0.5.10
        :param delay: delay	关机后多久重启	可选	[0, 14400] (单位分钟)	reboot为true时才生效	v0.8.1
        :param uuid:
        :return:
        """
        command = ShuidiRobot.generate_request_data(uuid, "shutdown",
                                                    {"reboot": "true" if reboot else "false",
                                                     "delay": delay})
        return self.data_send_recv(command)

    def set_luminance(self, value, uuid=1):
        # TODO: value [0-100]
        command = ShuidiRobot.generate_request_data(uuid, "LED/set_luminance", {"value": value})
        return self.data_send_recv(command)

    def set_luminance(self, r, g, b, uuid=1):
        # TODO: r g b [0-100]
        command = ShuidiRobot.generate_request_data(uuid, "LED/set_color", {"r": r, "g": g, "b": b})
        return self.data_send_recv(command)

    def diagnosis_get_result(self, uuid=1):
        """
        获取自诊断结果。
        :param uuid:
        :return:
        """
        command = ShuidiRobot.generate_request_data(uuid, "diagnosis/get_result")
        return self.data_send_recv(command)

    def get_power_status(self, uuid=1):
        """
        获取电池的电压、充电电压、电流、电量等信息。
        :param uuid:
        :return:
        """
        command = ShuidiRobot.generate_request_data(uuid, "get_power_status")
        return self.data_send_recv(command)

    def get_planned_path(self, uuid=1):
        """
        获取机器人当前规划的全局路径。
        注: 返回的路径上点的最大数量有限制，
        如果超过上限则在路径上平均取一定数量的点返回；如果当前没有任务，则返回空的路径。
        :param uuid:
        :return:
        """
        command = ShuidiRobot.generate_request_data(uuid, "get_planned_path")
        return self.data_send_recv(command)

    def make_plan(self, start_x, start_y, start_floor, goal_x, goal_y, goal_floor, uuid=1):
        """
        从起始位置到目标位置规划出一条最短路径, 返回路径的长度.
        """
        command = ShuidiRobot.generate_request_data(uuid, "make_plan",
                                                    {"start_x": start_x,
                                                     "start_y": start_y,
                                                     "start_floor": start_floor,
                                                     "goal_x": goal_x,
                                                     "goal_y": goal_y,
                                                     "goal_floor": goal_floor})
        return self.data_send_recv(command)

    @staticmethod
    def generate_request_data(uuid, command="", parameters={}):
        """
        generate the request string following Shuidi's API
        :param uuid:
        :param command:
        :param parameters:
        :return:
        """
        para = "?"
        for key, val in parameters.items():
            if val is None:
                continue
            tmp_para = f"{key}={val}"
            para += tmp_para
            para += "&"
        # print(f"/api/{command}{para}uuid={uuid}")
        return f"/api/{command}{para}uuid={uuid}"


if __name__ == '__main__':
    w = ShuidiRobot()


    def turn_lft(speed=0.1):
        w.joy_control(angular_velocity=speed, linear_velocity=0)


    def turn_rgt(speed=0.1):
        print(w.joy_control(angular_velocity=-speed, linear_velocity=0))


    def move_front(speed=0.1):
        w.joy_control(angular_velocity=0, linear_velocity=speed)


    def move_back(speed=0.1):
        w.joy_control(angular_velocity=0, linear_velocity=-speed)


    turn_rgt(speed=.1)
    time.sleep(.5)
    turn_rgt(speed=.1)
    time.sleep(.5)
    turn_rgt(speed=.1)
    time.sleep(.5)
    turn_rgt(speed=.1)
    time.sleep(.5)
    turn_rgt(speed=.1)
