"""
A high-level interface of the Zivid SDK for the WRS system
Author: Hao Chen <chen960216@gmail.com>
Update Notes: 20230213 Implement the fundamental functions
Prerequisite:
    Install Zivid  SDK:
        "pip install zivid"
Reference: https://support.zivid.com/en/latest/index.html
"""
from pathlib import Path
import datetime
import numpy as np
import zivid


class Zivid(object):
    @staticmethod
    def default_3d_settings() -> zivid.Settings:
        print("Configuring processing settings for capture:")
        settings = zivid.Settings()
        settings.experimental.engine = "phase"
        filters = settings.processing.filters
        filters.smoothing.gaussian.enabled = True
        filters.smoothing.gaussian.sigma = 1.5
        filters.noise.removal.enabled = True
        filters.noise.removal.threshold = 7.0
        filters.outlier.removal.enabled = True
        filters.outlier.removal.threshold = 5.0
        filters.reflection.removal.enabled = True
        filters.reflection.removal.experimental.mode = "global"
        filters.experimental.contrast_distortion.correction.enabled = True
        filters.experimental.contrast_distortion.correction.strength = 0.4
        filters.experimental.contrast_distortion.removal.enabled = False
        filters.experimental.contrast_distortion.removal.threshold = 0.5
        color = settings.processing.color
        color.balance.red = 1.0
        color.balance.blue = 1.0
        color.balance.green = 1.0
        color.gamma = 1.0
        settings.processing.color.experimental.mode = "automatic"
        print(settings.processing)
        # print("Configuring acquisition settings different for all HDR acquisitions")
        exposure_values = [
            (2.71, 1.03, 1677,),
            (9.24, 1, 1677,),
            (2, 1.68, 8385,)
        ]
        for (aperture, gain, exposure_time) in exposure_values:
            settings.acquisitions.append(
                zivid.Settings.Acquisition(
                    aperture=aperture,
                    exposure_time=datetime.timedelta(microseconds=exposure_time),
                    brightness=1.8,
                    gain=gain,
                )
            )
        return settings

    @staticmethod
    def default_2d_setting() -> zivid.Settings2D():
        settings_2d = zivid.Settings2D()
        settings_2d.acquisitions.append(zivid.Settings2D.Acquisition())
        settings_2d.acquisitions[0].exposure_time = datetime.timedelta(microseconds=30000)
        settings_2d.acquisitions[0].aperture = 11.31
        settings_2d.acquisitions[0].brightness = 1.80
        settings_2d.acquisitions[0].gain = 2.0
        settings_2d.processing.color.balance.red = 1.0
        settings_2d.processing.color.balance.green = 1.0
        settings_2d.processing.color.balance.blue = 1.0
        settings_2d.processing.color.gamma = 1.0
        return settings_2d

    @staticmethod
    def save_parameters(settings: zivid.Settings, settings_file: str or Path):
        print(f"Saving settings to file: {settings_file}")
        settings.save(settings_file)

    @staticmethod
    def load_parameters(settings_file: str or Path) -> zivid.Settings:
        print(f"Loading settings from file: {settings_file}")
        settings_from_file = zivid.Settings.load(settings_file)
        return settings_from_file

    def __init__(self, use_suggest_setting=False, serial_number: str = None, file_cam_path: str or Path = None):
        self.app = zivid.Application()
        if file_cam_path is None:
            self.cam = self.app.connect_camera(serial_number=serial_number)
        else:
            self.cam = self.app.create_file_camera(file_cam_path)
        self._settings_2d = self.default_2d_setting()
        self._settings_3d = self.default_3d_settings() if not use_suggest_setting else self.cam_parameters_suggest()

    def _trigger_3d_frame(self, get_snr=False) -> (np.ndarray, np.ndarray):
        with self.cam.capture(self._settings_3d) as frame_3d:
            point_cloud = frame_3d.point_cloud()
            xyz = point_cloud.copy_data("xyz")
            rgba = point_cloud.copy_data("rgba")
            if get_snr:
                snr = point_cloud.copy_data("snr")
        return xyz, rgba

    def _trigger_2d_frame(self) -> np.ndarray:
        with self.cam.capture(self._settings_2d) as frame_2d:
            img = frame_2d.image_rgba().copy_data()
        return img

    def get_pcd_rgba(self) -> (np.ndarray, np.ndarray, np.ndarray):
        """
        Get 1. point cloud 2. color of the point cloud 3. rgba image
        :return: nx3 raw point cloud data (with nan point),indices with no nan data, rgba image
        """
        xyz, rgba = self._trigger_3d_frame()
        # change unit from mm to meter
        raw_pcd = xyz.reshape(-1, 3) / 1000
        pcd_no_nan_indices = (~np.isnan(raw_pcd)).any(axis=1)
        return raw_pcd, pcd_no_nan_indices, rgba

    def get_rgba(self) -> np.ndarray:
        """
        Get RGBA image
        :return: rgba image
        """
        img = self._trigger_2d_frame()
        return img

    def get_temperature(self) -> dict:
        return {
            'general': self.cam.state.temperature.general,
            'lens': self.cam.state.temperature.lens,
            'dmd': self.cam.state.temperature.dmd,
            'led': self.cam.state.temperature.led,
            'pcb': self.cam.state.temperature.pcb,
        }

    def cam_parameters_suggest(self) -> zivid.Settings:
        suggest_settings_parameters = zivid.capture_assistant.SuggestSettingsParameters(
            max_capture_time=datetime.timedelta(milliseconds=1200),
            ambient_light_frequency=zivid.capture_assistant.SuggestSettingsParameters.AmbientLightFrequency.none,
        )
        print(f"Running Capture Assistant with parameters: {suggest_settings_parameters}")
        settings = zivid.capture_assistant.suggest_settings(self.cam, suggest_settings_parameters)
        return settings

    def release(self):
        self.app.release()

    def print_camera_list(self):
        cameras = self.app.cameras()
        for camera in cameras:
            print(f"Camera Info:  {camera}")

    def __str__(self):
        return f"zivid-python: {zivid.__version__} | " \
               f"Zivid SDK: {zivid.SDKVersion.full} | " \
               f"Camera Model: {self.cam.info.model_name} | " \
               f"Serial Number: {self.cam.info.serial_number}"

    def __del__(self):
        self.release()


if __name__ == "__main__":
    import cv2

    cam = Zivid()
    pcd, pcd_no_nan_indices, img = cam.get_pcd_rgba()
    print(pcd)
    # img = cam.get_rgba()
    # cv2.imshow("img", img)
    # cv2.waitKey(0)
    # print(img)
