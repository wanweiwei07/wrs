import numpy as np
from wrs.drivers.devices.realsense.realsense_d400s import RealSenseD405


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better value mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


def crop_img(img):
    r = cv2.boundingRect(img)
    print(r)
    return r


class RealSenseD405Crop(object):
    def __init__(self):
        self._rs1 = RealSenseD405()
        crop_data = fs.load_json('resources/crop_data.json')
        self.crop_lt1 = crop_data['lt1']
        self.crop_img_sz_1 = crop_data['img_sz1']
        self.crop_lt2 = crop_data['lt2']
        self.crop_img_sz_2 = crop_data['img_sz2']
        crop_plant = fs.load_json('resources/crop_plant.json')
        self.crop_lt_p = crop_plant['lt']
        self.crop_img_sz_p = crop_plant['img_sz']

    def get_color_img(self, ):
        rs: RealSenseD405 = self._rs1
        return rs.get_color_img()

    def get_pcd_texture_depth(self, ):
        rs: RealSenseD405 = self._rs1
        return rs.get_pcd_texture_depth()

    def crop_img(self, img, lt_p, img_size):
        # in opencv format
        lt_w, lt_h = lt_p
        w, h = img_size
        rb_w, rb_h = lt_w + w, lt_h + h
        return img[lt_h:rb_h, lt_w: rb_w]

    def get_learning_feature(self):
        img_o = self.get_color_img()
        img_c = self.crop_img(img_o, lt_p=self.crop_lt1, img_size=self.crop_img_sz_1)
        img_cb = self.crop_img(img_o, lt_p=self.crop_lt2, img_size=self.crop_img_sz_2)
        return img_c, img_cb, img_o

    def get_plant_feature(self):
        img_o = self.get_color_img()
        img_p = self.crop_img(img_o, lt_p=self.crop_lt_p, img_size=self.crop_img_sz_p)
        return img_o, img_p


if __name__ == "__main__":
    # print(find_devices())

    import cv2
    import file_sys as fs

    # def click_event(event, x, y, flags, params):
    #     # checking for left mouse clicks
    #     if event == cv2.EVENT_LBUTTONDOWN:
    #         # displaying the coordinates
    #         # on the Shell
    #         print(x, ' ', y)
    #
    #         # displaying the coordinates
    #         # on the image window
    #         font = cv2.FONT_HERSHEY_SIMPLEX
    #         cv2.putText(img, str(x) + ',' +
    #                     str(y), (x, y), font,
    #                     1, (255, 0, 0), 2)
    #         cv2.imshow('image', img)
    #
    #     # checking for right mouse clicks
    #     if event == cv2.EVENT_RBUTTONDOWN:
    #         # displaying the coordinates
    #         # on the Shell
    #         print(x, ' ', y)
    #
    #         # displaying the coordinates
    #         # on the image window
    #         font = cv2.FONT_HERSHEY_SIMPLEX
    #         b = img[y, x, 0]
    #         g = img[y, x, 1]
    #         r = img[y, x, 2]
    #         cv2.putText(img, str(b) + ',' +
    #                     str(g) + ',' + str(r),
    #                     (x, y), font, 1,
    #                     (255, 255, 0), 2)
    #         cv2.imshow('image', img)
    #
    #
    # # setting mouse handler for the image
    # # and calling the click_event() function
    #
    # cv2.imshow('image', img)
    #
    # cv2.setMouseCallback('image', click_event)
    # # displaying the image
    # # wait for a key to be pressed to exit
    # cv2.waitKey(0)

    rsd = RealSenseD405Crop()

    # close the window
    cv2.destroyAllWindows()

    # fs.dump_json({"lt": (685, 35), "img_sz": (80, 80)}, "crop_data.json")
    crop_data = fs.load_json('resources/crop_data.json')

    # def click_event(event, x, y, flags, params):
    #     cv2.imshow('image',
    #                letterbox(rsd.crop_img(img, crop_data['lt'], crop_data['img_sz']), new_shape=[300, 300], auto=True)[
    #                    0])

    img = rsd.get_color_img()
    cv2.imshow('image', img)
    # cv2.setMouseCallback('image', click_event)
    cv2.setWindowProperty('image', cv2.WND_PROP_TOPMOST, 1)
    while 1:
        crop_data = fs.load_json('resources/crop_data.json')
        img1 = rsd.get_color_img()
        img2 = rsd.get_color_img()

        # img1_r = rsd.crop_img(img1, crop_data['lt1'], crop_data['img_sz1'])
        img1_r = rsd.crop_img(img1, crop_data['lt2'], crop_data['img_sz2'])

        cv2.imshow('image', letterbox(img1_r, new_shape=[400, 400], auto=True)[0])

        cv2.waitKey(100)
    # while 1:
    #     img1 = rsd.get_color_img(device_id=1)
    #     img2 = rsd.get_color_img(device_id=2)
    #     # crop_img(img1)
    #     cv2.imshow("img", np.concatenate((img1, img2)))
    #     cv2.waitKey(0)
