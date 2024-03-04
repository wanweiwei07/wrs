import json
import math
import os
import pickle
import shutil
import tempfile
import threading
import turtle

# import cairosvg
# import canvasvg
import cv2
# import geojson
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon

import config


def draw_star(*args):
    turtle.color("purple")
    turtle.pensize(5)
    turtle.goto(0, 0)
    turtle.speed(1)
    for i in range(6):
        turtle.forward(100)
        turtle.right(144)


def pos_timer():
    global sub_timer
    sub_timer = threading.Timer(0.01, pos_timer)
    sub_timer.start()
    plist_ss.append(turtle.position())


def get_draw_result(draw_func, *args, **kwargs):
    global plist_ss
    global timer
    plist_ss = []
    timer = threading.Timer(1, pos_timer)
    timer.start()
    draw_func(*args, **kwargs)
    sub_timer.cancel()
    p_narray = np.array(plist_ss)
    pl_center = ((min(p_narray[:, 0]) + max(p_narray[:, 0])) / 2,
                 (min(p_narray[:, 1]) + max(p_narray[:, 1])) / 2)
    p_narray = p_narray - np.repeat([pl_center], len(p_narray), axis=0)

    return list(p_narray), turtle.getscreen().getcanvas()


def dump_drawpath(plist, f_name, root=config.ROOT + "/drawpath/pointlist/"):
    pickle.dump(plist, open(root + f_name, "wb"))


def load_drawpath(f_name, root=config.ROOT + "/drawpath/pointlist/"):
    return pickle.load(open(root + f_name, "rb"))


def draw_by_plist(plist):
    pen = turtle.Turtle()
    turtle.hideturtle()
    pen.color("red")
    pen.width(5)
    pen.speed(5)
    pen.up()
    pen.setpos(plist[0])
    pen.down()
    for point in plist:
        pen.goto(point)


def save_canvas_png(canvas, f_name, root=config.ROOT + "/drawpath/img/"):
    tmpdir = tempfile.mkdtemp()  # create a temporary directory
    tmpfile = os.path.join(tmpdir, 'tmp.svg')  # name of file to save SVG to
    canvasvg.saveall(tmpfile, canvas)
    with open(tmpfile) as svg_input, open(root + f_name, 'wb') as png_output:
        cairosvg.svg2png(bytestring=svg_input.read(), write_to=png_output)
    shutil.rmtree(tmpdir)  # clean up temp file(s)


def plot_ms(plist_ms):
    for s in plist_ms:
        print(len(s))
    merged_data = [p for s in plist_ms for p in s]
    x = [p[0] for p in merged_data]
    y = [p[1] for p in merged_data]

    plt.scatter(x, y)
    plt.show()


def plot_ss(plist_ss):
    x = [p[0] for p in plist_ss]
    y = [p[1] for p in plist_ss]

    plt.scatter(x, y)
    plt.show()


'''
code for getting stroke points from turtle drawing
'''


class TurtleStrokeRecorder:
    def __init__(self, record_interval=0.01):
        self.main_timer = None
        self.record_interval = record_interval

        self.record_stroke_pts = []

    def start(self, delay=0):
        self.record_stroke_pts = []
        delay_timer = threading.Timer(delay, self.__record)
        delay_timer.start()

    def cancel_and_get_records(self):
        self.main_timer.cancel()
        record_stroke_pts = list(dict.fromkeys(self.record_stroke_pts))
        self.record_stroke_pts = []
        return record_stroke_pts

    def set_record_interval(self, interval):
        self.record_interval = interval

    def __record(self):
        self.record_stroke_pts.append(turtle.position())
        self.main_timer = threading.Timer(self.record_interval, self.__record)
        self.main_timer.start()


def draw_star_wt_recorder():
    stroke_records = []
    stroke_recorder = TurtleStrokeRecorder()

    turtle.color("purple")
    turtle.pensize(5)
    turtle.goto(0, 0)
    turtle.speed(1)

    for i in range(6):
        stroke_recorder.start(0)
        turtle.forward(100)
        # tmp = stroke_recorder.cancel_and_get_records()
        stroke_records.append(stroke_recorder.cancel_and_get_records())
        turtle.right(144)

    return stroke_records, turtle.getscreen().getcanvas()


def draw_pig_wt_recorder():
    stroke_records = []
    stroke_recorder = TurtleStrokeRecorder()

    turtle.pensize(4)
    turtle.hideturtle()
    turtle.color("red")
    turtle.setup(840, 500)
    turtle.speed(3)

    # 鼻子
    turtle.pu()
    turtle.goto(-100, 100)
    turtle.pd()
    turtle.seth(-30)

    a = 0.4
    stroke_recorder.set_record_interval(0.2)
    stroke_recorder.start()
    for i in range(120):
        if 0 <= i < 30 or 60 <= i < 90:
            a = a + 0.08
            turtle.lt(3)  # 向左转3度
            turtle.fd(a)  # 向前走a的步长
        else:
            a = a - 0.08
            turtle.lt(3)
            turtle.fd(a)
    stroke_records.append(stroke_recorder.cancel_and_get_records())
    stroke_recorder.set_record_interval(0.01)

    turtle.pu()
    turtle.seth(90)
    turtle.fd(25)
    turtle.seth(0)
    turtle.fd(10)
    turtle.pd()
    turtle.seth(10)

    stroke_recorder.start()
    turtle.circle(5)
    stroke_records.append(stroke_recorder.cancel_and_get_records())

    turtle.pu()
    turtle.seth(0)
    turtle.fd(20)
    turtle.pd()
    turtle.seth(10)

    stroke_recorder.start()
    turtle.circle(5)
    stroke_records.append(stroke_recorder.cancel_and_get_records())

    # 头
    turtle.pu()
    turtle.seth(90)
    turtle.fd(41)
    turtle.seth(0)
    turtle.fd(0)
    turtle.pd()
    turtle.seth(180)

    stroke_recorder.start()
    turtle.circle(300, -30)
    turtle.circle(100, -60)
    turtle.circle(80, -100)
    turtle.circle(150, -20)
    turtle.circle(60, -95)
    turtle.seth(161)
    turtle.circle(-300, 15)
    stroke_records.append(stroke_recorder.cancel_and_get_records())

    turtle.pu()
    turtle.goto(-100, 100)
    turtle.pd()
    turtle.seth(-30)

    for i in range(60):
        if 0 <= i < 30 or 60 <= i < 90:
            a = a + 0.08
            turtle.lt(3)  # 向左转3度
            turtle.fd(a)  # 向前走a的步长
        else:
            a = a - 0.08
            turtle.lt(3)
            turtle.fd(a)

    # 耳朵
    turtle.pu()
    turtle.seth(90)
    turtle.fd(-7)
    turtle.seth(0)
    turtle.fd(70)
    turtle.pd()
    turtle.seth(100)

    stroke_recorder.start()
    turtle.circle(-50, 50)
    turtle.circle(-10, 120)
    turtle.circle(-50, 54)
    stroke_records.append(stroke_recorder.cancel_and_get_records())

    turtle.pu()
    turtle.seth(90)
    turtle.fd(-12)
    turtle.seth(0)
    turtle.fd(30)
    turtle.pd()
    turtle.seth(100)

    stroke_recorder.start()
    turtle.circle(-50, 50)
    turtle.circle(-10, 120)
    turtle.circle(-50, 56)
    stroke_records.append(stroke_recorder.cancel_and_get_records())

    # 眼睛
    turtle.pu()
    turtle.seth(90)
    turtle.fd(-20)
    turtle.seth(0)
    turtle.fd(-95)
    turtle.pd()

    stroke_recorder.start()
    turtle.circle(15)
    stroke_records.append(stroke_recorder.cancel_and_get_records())

    turtle.pu()
    turtle.seth(90)
    turtle.fd(12)
    turtle.seth(0)
    turtle.fd(-3)
    turtle.pd()

    stroke_recorder.start()
    turtle.circle(3)
    stroke_records.append(stroke_recorder.cancel_and_get_records())

    turtle.pu()
    turtle.seth(90)
    turtle.fd(-25)
    turtle.seth(0)
    turtle.fd(40)
    turtle.pd()

    stroke_recorder.start()
    turtle.circle(15)
    stroke_records.append(stroke_recorder.cancel_and_get_records())

    turtle.pu()
    turtle.seth(90)
    turtle.fd(12)
    turtle.seth(0)
    turtle.fd(-3)
    turtle.pd()

    stroke_recorder.start()
    turtle.circle(3)
    stroke_records.append(stroke_recorder.cancel_and_get_records())

    # 腮
    turtle.pu()
    turtle.seth(90)
    turtle.fd(-95)
    turtle.seth(0)
    turtle.fd(65)
    turtle.pd()

    stroke_recorder.start()
    turtle.circle(30)
    stroke_records.append(stroke_recorder.cancel_and_get_records())

    # 嘴
    turtle.pu()
    turtle.seth(90)
    turtle.fd(15)
    turtle.seth(0)
    turtle.fd(-100)
    turtle.pd()
    turtle.seth(-80)

    stroke_recorder.start()
    turtle.circle(30, 40)
    turtle.circle(40, 80)
    stroke_records.append(stroke_recorder.cancel_and_get_records())
    #
    # # 身体
    # turtle.pu()
    # turtle.seth(90)
    # turtle.fd(-20)
    # turtle.seth(0)
    # turtle.fd(-78)
    # turtle.pd()
    # turtle.seth(-130)
    #
    # stroke_recorder.start()
    # turtle.circle(100, 10)
    # turtle.circle(300, 30)
    # turtle.seth(0)
    # turtle.fd(230)
    # turtle.seth(90)
    # turtle.circle(300, 30)
    # turtle.circle(100, 3)
    # turtle.seth(-135)
    # turtle.circle(-80, 63)
    # turtle.circle(-150, 24)
    # stroke_records.append(stroke_recorder.cancel_and_get_records())
    #
    # # 手
    # turtle.pu()
    # turtle.seth(90)
    # turtle.fd(-40)
    # turtle.seth(0)
    # turtle.fd(-27)
    # turtle.pd()
    # turtle.seth(-160)
    #
    # stroke_recorder.start()
    # turtle.circle(300, 15)
    # stroke_records.append(stroke_recorder.cancel_and_get_records())
    #
    # turtle.pu()
    # turtle.seth(90)
    # turtle.fd(15)
    # turtle.seth(0)
    # turtle.fd(0)
    # turtle.pd()
    # turtle.seth(-10)
    #
    # stroke_recorder.start()
    # turtle.circle(-20, 90)
    # stroke_records.append(stroke_recorder.cancel_and_get_records())
    #
    # turtle.pu()
    # turtle.seth(90)
    # turtle.fd(30)
    # turtle.seth(0)
    # turtle.fd(237)
    # turtle.pd()
    # turtle.seth(-20)
    #
    # stroke_recorder.start()
    # turtle.circle(-300, 15)
    # stroke_records.append(stroke_recorder.cancel_and_get_records())
    #
    # turtle.pu()
    # turtle.seth(90)
    # turtle.fd(20)
    # turtle.seth(0)
    # turtle.fd(0)
    # turtle.pd()
    # turtle.seth(-170)
    #
    # stroke_recorder.start()
    # turtle.circle(20, 90)
    # stroke_records.append(stroke_recorder.cancel_and_get_records())
    #
    # # 脚
    # turtle.pensize(4)
    # turtle.pu()
    # turtle.seth(90)
    # turtle.fd(-75)
    # turtle.seth(0)
    # turtle.fd(-180)
    # turtle.pd()
    # turtle.seth(-90)
    #
    # stroke_recorder.start()
    # turtle.fd(40)
    # turtle.seth(-180)
    # turtle.pensize(15)
    # turtle.fd(20)
    # stroke_records.append(stroke_recorder.cancel_and_get_records())
    #
    # turtle.pensize(4)
    # turtle.pu()
    # turtle.seth(90)
    # turtle.fd(40)
    # turtle.seth(0)
    # turtle.fd(90)
    # turtle.pd()
    # turtle.seth(-90)
    #
    # stroke_recorder.start()
    # turtle.fd(40)
    # turtle.seth(-180)
    # turtle.pensize(15)
    # turtle.fd(20)
    # stroke_records.append(stroke_recorder.cancel_and_get_records())
    #
    # # 尾巴
    # turtle.pensize(4)
    # turtle.pu()
    # turtle.seth(90)
    # turtle.fd(70)
    # turtle.seth(0)
    # turtle.fd(95)
    # turtle.pd()
    # turtle.seth(0)
    #
    # stroke_recorder.start()
    # turtle.circle(70, 20)
    # turtle.circle(10, 330)
    # turtle.circle(70, 30)
    # stroke_records.append(stroke_recorder.cancel_and_get_records())

    return stroke_records, turtle.getscreen().getcanvas()


def img2polygon(png_f_name, json_f_name, root=config.ROOT + "/drawpath/"):
    polypic = cv2.imread(os.path.join(root, "img", png_f_name))
    gray = cv2.cvtColor(polypic, cv2.COLOR_BGR2GRAY)

    ret, binary = cv2.threshold(gray, 75, 255, cv2.THRESH_BINARY)
    # kernel = np.ones((5, 5), np.uint8)
    # dilation = cv2.dilate(binary, kernel, iterations=1)
    # erosion = cv2.erode(dilation,kernel,iterations=1)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    result = {}
    for i, c in enumerate(contours):
        c = [p[0] for p in c]

        poly = Polygon(c).simplify(0.5)
        result[i] = poly

    with open(os.path.join(root, "json", json_f_name), "w") as f:
        f.write(geojson.dumps(result))


def fit_drawpath_in_center_ms(drawpath_ms):
    p_set = []
    for s in drawpath_ms:
        for p in s:
            p_set.append(p)
    p_set = np.array(p_set)
    center = np.array(((max(p_set[:, 0]) - min(p_set[:, 0])) / 2, (max(p_set[:, 1]) - min(p_set[:, 1])) / 2))

    plist_ms_resize = []
    for s in drawpath_ms:
        s_resize = []
        for p in s:
            p_resize = np.array(p) - center
            s_resize.append(p_resize)
        plist_ms_resize.append(s_resize)
    return plist_ms_resize


def json2plist_ms(json_f_name, root=config.ROOT + "/drawpath/json/"):
    with open(os.path.join(root, json_f_name)) as f:
        poly_dic = json.load(f)
    drawpath_ms = []
    for k, v in poly_dic.items():
        drawpath_ms.append(v["coordinates"][0])

    drawpath_ms_resize = fit_drawpath_in_center_ms(drawpath_ms)

    return np.array(drawpath_ms_resize)


def gen_circle(r=.015, interval=15):
    p_list = []
    for i in range(-180, 180, interval):
        p_list.append((r * math.sin(math.radians(i)), r * math.cos(math.radians(i))))
    return p_list


def gen_square(side_len=40, step=1):
    p_list = []
    corner = np.asarray([(-side_len / 2, -side_len / 2), (-side_len / 2, side_len / 2),
                         (side_len / 2, side_len / 2), (side_len / 2, -side_len / 2)]).astype(int)
    for i, v in enumerate(corner):
        p1 = corner[i - 1]
        p2 = corner[i]
        if p1[0] == p2[0]:
            if p1[1] > p2[1]:
                step = -abs(step)
            else:
                step = abs(step)
            for j in range(p1[1], p2[1], step):
                p_list.append((p1[0], j))
        else:
            if p1[0] > p2[0]:
                step = -abs(step)
            else:
                step = abs(step)
            for j in range(p1[0], p2[0], step):
                p_list.append((j, p1[1]))

    return p_list


def gen_grid(side_len=100, grid_len=10, step=1):
    drawpath_ms = []
    for x in range(0, side_len + 1, grid_len):
        stroke = [(x - side_len / 2, y - side_len / 2) for y in range(0, side_len + 1, step)]
        if len(drawpath_ms) % 2 == 0:
            drawpath_ms.append(stroke[::-1])
        else:
            drawpath_ms.append(stroke)
    for y in range(0, side_len + 1, grid_len):
        stroke = [(x - side_len / 2, y - side_len / 2) for x in range(0, side_len + 1, step)]
        if len(drawpath_ms) % 2 != 0:
            drawpath_ms.append(stroke[::-1])
        else:
            drawpath_ms.append(stroke)
    return drawpath_ms


def gen_line(len=100, interval=1, axis=0):
    if axis == 0:
        return [(0, y - len / 2) for y in range(0, len + interval, interval)]
    else:
        return [(x - len / 2, 0) for x in range(0, len + interval, interval)]


# def get_dist(drawpath_ms, size=(80, 80)):
#     grid_points_dist = []
#     strip = size[0] // 10
#     for i in range(0, strip + 2):
#         for j in range(0, strip + 2):
#             x = int(j * 10)
#             y = int(i * 10)
#
#             hl_id = i
#             vl_id = j + strip
#
#             hlp_id = int((strip + strip * (-1 ** i)) / 2 + (-1 ** i) * j)
#             vlp_id = int((strip + strip * (-1 ** j)) / 2 + (-1 ** j) * i)
#             print(hl_id, vl_id, hlp_id, vlp_id)
#             dist = np.linalg.norm(np.array(drawpath_ms[hl_id][hlp_id]) - np.array(drawpath_ms[vl_id][vlp_id]))
#             grid_points_dist.append(dist)
#
#     print(grid_points_dist)


def get_dist(drawpath_ms, size=(80, 80)):
    grid_points_dist = []
    x_strip = size[0] // 10

    for i in range(0, x_strip + 1):
        for j in range(x_strip + 1, len(drawpath_ms)):
            if i % 2 == 0:
                h_id = size[0] - (j - x_strip - 1) * 10
            else:
                h_id = (j - x_strip - 1) * 10
            if j % 2 == 0:
                v_id = i * 10
            else:
                v_id = size[1] - i * 10

            print(i, j, h_id, v_id)

            p1 = np.asarray(drawpath_ms[i][h_id])
            p2 = np.asarray(drawpath_ms[j][v_id])
            dist = np.linalg.norm(p1 - p2)
            print(dist)
            grid_points_dist.append(dist)

    print(len(grid_points_dist), grid_points_dist)


if __name__ == '__main__':
    pl_f_name = "grid.pkl"
    img_f_name = "grid.jpg"
    # json_f_name = "pig.json"

    '''
    dump point list
    '''
    # drawpath = gen_square()
    # draw_by_plist(drawpath)
    # plist_ss, canvas = get_draw_result(draw_star)
    # print(len(plist_ss))
    # dump_drawpath(plist_ss, pl_f_name)
    # save_canvas_png(canvas, img_f_name)

    # plist_ms, canvas = draw_pig_wt_recorder()
    # dump_drawpath(plist_ms, pl_f_name)
    # save_canvas_png(canvas, img_f_name)
    # plot_ms(plist_ms)

    drawpath_ms = gen_grid(side_len=80, grid_len=10, step=20)
    plot_ms(drawpath_ms)

    # dump_drawpath(drawpath_ms, pl_f_name)
    # for plist in drawpath_ms:
    #     draw_by_plist(plist)
    #     print(plist)
    # get_dist(drawpath_ms, size=(80, 80))

    '''
    load and draw point list
    '''
    # plist_ms = load_drawpath(pl_f_name)
    #
    # for plist in plist_ms:
    #     draw_by_plist(plist)
    # plot_multistroke(plist_ms)

    '''
    image to point list
    '''
    # img2polygon(img_f_name, json_f_name)
    # plist_ms = json2plist_ms(json_f_name)
    # for plist in plist_ms:
    #     draw_by_plist(plist)
    # plot_multistroke(plist_ms)
