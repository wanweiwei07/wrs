import copy

import cv2
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from vit_pytorch.efficient import ViT
from linformer import Linformer
from PIL import Image
import numpy as np

from wrs import basis as rm
import math
import time
import random


def spiral(num):
    x = []
    y = []
    r = 0.000
    theta = 0
    for i in range(num):
        theta = theta + math.pi / (2 + 0.25 * i)
        r = r + 0.0002 / (1 + 0.1 * i)
        x.append(r * math.cos(theta))
        y.append(r * math.sin(theta))
    return x, y


x_list, y_list = spiral(200)
spiral_list = np.zeros((200, 2))
spiral_list.T[0] = x_list
spiral_list.T[1] = y_list


def get_rbt_gl_from_pipette(pipette_gl_pos, pipette_gl_angle=0):
    dist = 0.007
    pipette_gl_rot = rm.rotmat_from_axangle(np.array([0, 0, 1]), np.radians(pipette_gl_angle))
    pipette_gl_mat = rm.homomat_from_posrot(pipette_gl_pos, pipette_gl_rot)
    pipette_tcp_pos = np.array([-0.008, -0.15485, 0.01075]) + np.array([0.0015, -dist, -0.0058])
    pipette_tcp_rot = np.dot(rm.rotmat_from_axangle(np.array([0, 0, 1]), -math.pi / 2),
                             rm.rotmat_from_axangle(np.array([0, 1, 0]), -math.pi / 2))
    pipette_tcp_mat = rm.homomat_from_posrot(pipette_tcp_pos, pipette_tcp_rot)
    rbt_tcp_pos = np.array([0, 4.7, 10]) / 1000
    rbt_tcp_mat = rm.homomat_from_posrot(rbt_tcp_pos, np.eye(3))
    rbt_gl_mat = np.dot(np.dot(pipette_gl_mat, np.linalg.inv(pipette_tcp_mat)), rbt_tcp_mat)
    rot_euler = rm.rotmat_to_euler(rbt_gl_mat[:3, :3])
    return np.append(np.append(rbt_gl_mat[:3, 3], rot_euler), 5)


class ResnetModel(object):
    def __init__(self, model_path, device="cpu"):
        self.model_resnet50 = self.model_load(model_path, device)
        self.pic_transformer = transforms.Compose([transforms.ToTensor()])

    def model_load(self, path, device):
        new_model = torchvision.models.resnet50(pretrained=False).to(device)
        new_model.load_state_dict(torch.load(path, map_location=torch.device(device)))
        new_model.eval()
        return new_model

    def get_score(self, pic, show_img=False):
        pic = Image.fromarray(pic)
        pic_tensor = self.pic_transformer(pic)
        pic_tensor = pic_tensor.unsqueeze(0)
        model = self.model_resnet50
        [val_output] = model(pic_tensor).detach().numpy()
        if show_img:
            def show_img_callback():
                img = np.array(transforms.ToPILImage()(pic_tensor.squeeze(0)))
                cv2.imshow("img", img)
                cv2.waitKey(0)
        else:
            show_img_callback = lambda: None
        # print(val_output)
        return val_output.argmax(), val_output, show_img_callback


class TransformerModel(object):
    def __init__(self, model_sata, img_size=(45, 80), patch_size=5, num_classes=145, device="cpu"):
        self.model_vit = self.model_load(model_sata, img_size, patch_size, num_classes, device)
        self.pic_transformer = transforms.Compose([transforms.ToTensor()])

    def model_load(self, path, img_size, patch_size, num_classes, device):
        efficient_transformer = Linformer(dim=128,
                                          seq_len=int(img_size[0] / patch_size * img_size[1] / patch_size) + 1,
                                          # n*m patches + 1 cls-token
                                          depth=12,
                                          heads=8,
                                          k=64)
        new_model = ViT(dim=128,
                        image_size=img_size,
                        patch_size=patch_size,
                        num_classes=num_classes,
                        transformer=efficient_transformer,
                        channels=3).to(device)
        new_model.load_state_dict(torch.load(path, map_location=torch.device(device)))
        new_model.eval()
        return new_model

    def get_score(self, pic, show_img=False):
        pic = Image.fromarray(pic)
        pic_tensor = self.pic_transformer(pic)
        pic_tensor = pic_tensor.unsqueeze(0)
        model = self.model_vit
        [val_output] = model(pic_tensor).detach().numpy()
        if show_img:
            def show_img_callback():
                img = np.array(transforms.ToPILImage()(pic_tensor.squeeze(0)))
                cv2.imshow("img", img)
                cv2.waitKey(0)
        else:
            show_img_callback = lambda: None
        # print(val_output)
        return val_output.argmax(), val_output, show_img_callback


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, padding=1),
                                     nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(True),
                                     nn.MaxPool2d(2, 2),

                                     nn.Conv2d(64, 128, kernel_size=3, padding=1),
                                     nn.Conv2d(128, 128, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU(True),
                                     nn.MaxPool2d(2, 2),

                                     nn.Conv2d(128, 256, kernel_size=3, padding=1),
                                     nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(256),
                                     nn.ReLU(True))

        self.decoder = nn.Sequential(nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1),
                                     nn.ReLU(True),
                                     nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1),
                                     nn.ReLU(True),
                                     nn.ConvTranspose2d(64, 3, kernel_size=3, padding=1))

        self.decoder2 = nn.Sequential(nn.ConvTranspose2d(3, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
                                      nn.ReLU(True),
                                      nn.ConvTranspose2d(3, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
                                      nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.decoder2(x)
        return x


class SciModel(object):
    def __init__(self, model_path_ae, model_path_knn, img_size=(45, 80), device="cpu"):
        self.img_size = img_size
        self.model_loader(model_path_ae, model_path_knn, device)
        self.pic_transformer = transforms.Compose([transforms.Resize((48, img_size[1])), transforms.ToTensor()])

    def model_loader(self, path_ae, path_sci, device):
        self.model_ae = Autoencoder().to(device)
        self.model_ae.load_state_dict(torch.load(path_ae, map_location=torch.device(device)))
        self.model_ae.eval()
        self.model_sci = torch.load(path_sci, map_location=torch.device(device))

    def get_score(self, pic):
        pic = Image.fromarray(pic)
        pic_tensor = self.pic_transformer(pic)
        pic_tensor = pic_tensor.unsqueeze(0)
        pic_feature = self.model_ae.encoder(pic_tensor)
        pic_feature = self.model_ae.decoder(pic_feature)
        pic_feature = pic_feature.view(-1, self.img_size[1] * 9).detach().numpy()
        score = self.model_sci.predict(pic_feature)
        return score


class MaskRCNNModel(object):
    def __init__(self, model_path, device="cpu"):
        self.leaf_model = torch.load(model_path, map_location=torch.device(device))
        self.leaf_model.eval()

    def leaf_biggest(self, img):
        img_input = copy.deepcopy(img)
        mask_list, box_list, img_mask = self.mask_detect(img_input, 90)
        if mask_list is None:
            print("No feasible result")
            return None, None, None
        area_list = []
        center_list = []
        for mask in mask_list:
            area = np.sum(mask)
            # print(area)
            cx, cy, theta = self.get_mask_center(mask)
            area_list.append(area)
            center_list.append(np.array([cx, cy]))
        area_list = np.array(area_list)
        center_list = np.array(center_list)
        center = np.int0(np.average(a=center_list, axis=0, weights=np.sqrt(area_list)))

        max_id = np.argmax(area_list)
        cx, cy, theta = self.get_mask_center(mask_list[max_id])

        cv2.circle(img_input, (cx, cy), 3, (155, 200, 55), 2)
        img_input = self.apply_mask(img_input, np.around(mask_list[max_id]).astype(np.uint8), color=np.array([200,128,0]))
        cv2.rectangle(img_input, (box_list[max_id][0], box_list[max_id][1]), (box_list[max_id][2], box_list[max_id][3]),
                      color=(0, 255, 0),
                      thickness=1)
        cv2.imshow("max_mask", img_input)
        return img_input, [(cx, cy)], img_mask

    def leaf_biggest_two(self, img):
        img_input = copy.deepcopy(img)
        mask_list, box_list, img_mask = self.mask_detect(img_input, 90)
        if mask_list is None:
            print("No feasible result")
            return None, None, None
        area_list = []
        center_list = []
        for mask in mask_list:
            area = np.sum(mask)
            # print(area)
            cx, cy, theta = self.get_mask_center(mask)
            area_list.append(area)
            center_list.append(np.array([cx, cy]))
        area_list = np.array(area_list)
        center_list = np.array(center_list)
        area_argsort = np.argsort(area_list)[::-1]

        mask_center_list = []
        for leaf_count in range(2):
            mask_id = area_argsort[leaf_count]
            cx, cy, theta = self.get_mask_center(mask_list[mask_id])

            cv2.circle(img_input, (cx, cy), 3, (155, 200, 55), 2)
            img_input = self.apply_mask(img_input, np.around(mask_list[mask_id]).astype(np.uint8),
                                        color=np.array([200, 128, 0]))
            cv2.rectangle(img_input, (box_list[mask_id][0], box_list[mask_id][1]),
                          (box_list[mask_id][2], box_list[mask_id][3]),
                          color=(0, 255, 0),
                          thickness=1)
            mask_center_list.append(np.array([cx, cy]))
        cv2.imshow("max_mask", img_input)
        return img_input, mask_center_list, img_mask

    def leaf_biggest_four(self, img):
        img_input = copy.deepcopy(img)
        mask_list, box_list, img_mask = self.mask_detect(img_input, 90)
        if mask_list is None:
            print("No feasible result")
            return None, None, None
        area_list = []
        center_list = []
        for mask in mask_list:
            area = np.sum(mask)
            # print(area)
            cx, cy, theta = self.get_mask_center(mask)
            area_list.append(area)
            center_list.append(np.array([cx, cy]))
        area_list = np.array(area_list)
        center_list = np.array(center_list)
        area_argsort = np.argsort(area_list)[::-1]

        mask_center_list=[]
        for leaf_count in range(4):

            mask_id = area_argsort[leaf_count]
            cx, cy, theta = self.get_mask_center(mask_list[mask_id])

            cv2.circle(img_input, (cx, cy), 3, (155, 200, 55), 2)
            img_input = self.apply_mask(img_input, np.around(mask_list[mask_id]).astype(np.uint8), color=np.array([200,128,0]))
            cv2.rectangle(img_input, (box_list[mask_id][0], box_list[mask_id][1]), (box_list[mask_id][2], box_list[mask_id][3]),
                          color=(0, 255, 0),
                          thickness=1)
            mask_center_list.append(np.array([cx, cy]))
        cv2.imshow("max_mask", img_input)
        return img_input, mask_center_list, img_mask

    def four_leaf_plus_center(self, img, center_pixel):
        img_input = copy.deepcopy(img)
        img_input, mask_center_list, img_mask = self.leaf_biggest_four(img_input)
        cx, cy = center_pixel
        cv2.circle(img_input, (cx, cy), 3, (155, 200, 55), 2)

        return img_input, mask_center_list, img_mask

    def plant_center(self, img):
        img_input = copy.deepcopy(img)
        mask_list, box_list, img_mask = self.mask_detect(img_input, 40)
        if mask_list is None:
            print("No feasible result")
            return None, None, None
        area_list = []
        center_list = []
        for mask in mask_list:
            area = np.sum(mask)
            cx, cy, theta = self.get_mask_center(mask)
            area_list.append(area)
            center_list.append(np.array([cx, cy]))
        area_list = np.array(area_list)
        center_list = np.array(center_list)
        center = np.int0(np.average(a=center_list, axis=0, weights=np.sqrt(area_list)))
        cv2.circle(img_input, center, 3, (0, 255, 0), thickness=2)
        return img_input, [center], img_mask

    def mask_detect(self, img_input, tgt_score):
        plant_img = img_input.copy()
        img = Image.fromarray(plant_img)
        transform = transforms.Compose([transforms.ToTensor()])
        img = transform(img)
        predict = self.leaf_model([img])

        score_list = predict[0]['scores'].detach().numpy()
        predict_num = len(score_list)
        # print(predict_num, score_list)

        mask_list = predict[0]['masks'].squeeze().detach().cpu().numpy()
        box_list = predict[0]['boxes'].detach().numpy()
        box_list = np.around(box_list).astype(np.int32)
        id_list = self.mask_filter(mask_list, score_list, tgt_score)
        if len(id_list) < 1:
            return None, None, None
        mask_list = mask_list[id_list]
        box_list = box_list[id_list]
        img_mask = self.full_masks(img_input, mask_list, box_list)
        return mask_list, box_list, img_mask

    def get_mask_center(self, mask):
        mask = np.around(mask).astype(np.uint8)
        mask_counters, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_counter = max(mask_counters, key=cv2.contourArea)
        M = cv2.moments(max_counter)
        cX = round(M["m10"] / M["m00"])
        cY = round(M["m01"] / M["m00"])
        a = M["m20"] / M["m00"] - cX * cX
        b = M["m11"] / M["m00"] - cX * cY
        c = M["m02"] / M["m00"] - cY * cY
        theta = cv2.fastAtan2(2 * b, (a - c) / 2)
        return cX, cY, theta

    def mask_filter(self, mask_list, score_list, tgt_score):
        if len(mask_list) < 1:
            return None
        mask_area = np.sum(mask_list, axis=(1, 2))
        mask_area_avar = np.average(mask_area)
        id_list = []
        for i in range(len(mask_list)):
            if score_list[i] < (tgt_score / 100):
                break
            mask = np.around(mask_list[i]).astype(np.uint8)
            try:
                cx, cy, theta = self.get_mask_center(mask)
                c_distance = (cx - 150) * (cx - 150) + (cy - 170) * (cy - 170)
                # print(cX, cY, np.sqrt(c_distance))
                if np.sqrt(c_distance) > 120:
                    continue
            except:
                print(i)
                continue
            if (mask_area[i] > mask_area_avar * 3) & (tgt_score < 50):
                continue
            id_list.append(i)
        return id_list

    def apply_mask(self, image, mask, color=None, alpha=0.5):
        if color is None:
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            color = np.array([r, g, b])
        for c in range(3):
            image[:, :, c] = np.where(mask == 1,
                                      image[:, :, c] *
                                      (1 - alpha) + alpha * color[c] * 255,
                                      image[:, :, c])
        return image

    def full_masks(self, img_input, mask_list, box_list):
        plant_img = copy.deepcopy(img_input)
        for id in range(len(mask_list)):
            cx, cy, theta = self.get_mask_center(mask_list[id])
            cv2.circle(plant_img, (cx, cy), 2, (155, 200, 55), -1)
            plant_img = self.apply_mask(plant_img, mask_list[id])
            cv2.rectangle(plant_img, (box_list[id][0], box_list[id][1]), (box_list[id][2], box_list[id][3]),
                          color=(0, 255, 0),
                          thickness=1)
        return plant_img


if __name__ == "__main__":
    tic = time.time()
    # model_row_s = TransformerModel("tip_cam_spiral_c", (45, 80), 5, 165)
    leaf_model = MaskRCNNModel("model/mask_rcnn_model")
    # model_row_tip = GetDirection("tip_cam_spiral", (45, 80), 5, 145)
    img = cv2.imread("./capture/2023_5_29/plant_20_origin.png")
    plant_img =img[135:295, 240:400]
    img_size = plant_img.shape[:2]
    plant_img = cv2.resize(plant_img, (img_size[0] * 2, img_size[1] * 2))
    cv2.imshow("plant", plant_img)
    # cv2.waitKey(0)
    plant_mask, leaf_center_pixel, leaf_img_mask = leaf_model.leaf_biggest_four(plant_img)
    print(plant_mask, leaf_center_pixel, leaf_img_mask)
    cv2.imshow("mask",leaf_img_mask)
    cv2.waitKey(0)
    print("model:", time.time() - tic)
    tic = time.time()
