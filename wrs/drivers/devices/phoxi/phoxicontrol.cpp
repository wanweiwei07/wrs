//pybind11 interface for phoxicontrol
//
//author: weiwei
//date: 20191128
//
// Cooperate with external RGB camera to generate color pcd
// author: hao chen
// data: 20220318

#include <iostream>
#include <memory>
#include <vector>
#include "Phoxi.h"

// added by chen 20220318
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <PhoXiOpenCVSupport.h>
#include <fstream>
#if defined(_WIN32)
#include <windows.h>
#elif defined (__linux__)
#include <unistd.h>
#endif
//#include "PhoLocalization.h"

class PhoxiControl {

private:
	bool isconnected = false;
	pho::api::PPhoXi cam;
	int frameid = -1;
	bool isframeobtained = false;
	pho::api::PFrame frame;

	// added by chen 20220318
	bool iscalibloaded = false;
	bool isexternalcamconnected = false;
	pho::api::AdditionalCameraCalibration calibration;
	cv::VideoCapture cap;
	cv::Mat externalcamimage;

private:
	bool isDeviceAvailable(pho::api::PhoXiFactory& factory, const std::string& serialNumber) {
		if (!factory.isPhoXiControlRunning()) {
			std::cout << "[!] PhoXi Control Software is not running." << std::endl;
			return false;
		}
		std::vector <pho::api::PhoXiDeviceInformation> deviceList = factory.GetDeviceList();
		if (deviceList.empty()) {
			std::cout << "[!] 0 devices found." << std::endl;
			return false;
		}
		bool isFound = false;
		for (std::size_t i = 0; i < deviceList.size(); ++i) {
			if (deviceList[i].HWIdentification == serialNumber) {
				isFound = true;
			}
		}
		return isFound;
	}

	pho::api::PPhoXi connectToDevice(pho::api::PhoXiFactory& factory, const std::string& serialNumber) {
		if (!factory.isPhoXiControlRunning()) {
			std::cout << "PhoXi Control Software is not running!" << std::endl;
			return 0;
		}
		pho::api::PPhoXi PhoXiDevice = factory.CreateAndConnect(serialNumber);
		return PhoXiDevice;
	}

	void configDevice(const pho::api::PPhoXi& PhoXiDevice, const std::string& resolution) {
		// Set trigger to "freerun" mode
		if (PhoXiDevice->TriggerMode != pho::api::PhoXiTriggerMode::Software) {
			if (PhoXiDevice->isAcquiring()) {
				if (!PhoXiDevice->StopAcquisition()) {
					throw std::runtime_error("Error in StopAcquistion");
				}
			}
			PhoXiDevice->TriggerMode = pho::api::PhoXiTriggerMode::Software;
			std::cout << "[*] Software mode was set." << std::endl;
			if (!PhoXiDevice->TriggerMode.isLastOperationSuccessful()) {
				throw std::runtime_error(PhoXiDevice->TriggerMode.GetLastErrorMessage().c_str());
			}
		}
		// Just send Texture and DepthMap
		pho::api::FrameOutputSettings currentOutputSettings = PhoXiDevice->OutputSettings;
		pho::api::FrameOutputSettings newOutputSettings = currentOutputSettings;
		newOutputSettings.SendPointCloud = true;
		newOutputSettings.SendNormalMap = true;
		newOutputSettings.SendDepthMap = true;
		newOutputSettings.SendConfidenceMap = true;
		newOutputSettings.SendTexture = true;
		PhoXiDevice->OutputSettings = newOutputSettings;
		// Configure the device resolution
		pho::api::PhoXiCapturingMode mode = PhoXiDevice->CapturingMode;
		if (resolution == "low") {
			mode.Resolution.Width = 1032;
			mode.Resolution.Height = 772;
		}
		else {
			mode.Resolution.Width = 2064;
			mode.Resolution.Height = 1544;
		}
		PhoXiDevice->CapturingMode = mode;
	}

	bool getFrame(const pho::api::PPhoXi& PhoXiDevice, pho::api::PFrame& FrameReturn) {
		// start device acquisition if necessary
		if (!PhoXiDevice->isAcquiring()) PhoXiDevice->StartAcquisition();
		// clear the current acquisition buffer
		PhoXiDevice->ClearBuffer();
		if (!PhoXiDevice->isAcquiring()) {
			std::cout << "[!] Your device could not start acquisition!" << std::endl;
			return false;
		}
		std::cout << "[*] Waiting for a frame." << std::endl;
		int frameId = PhoXiDevice->TriggerFrame();
		FrameReturn = PhoXiDevice->GetSpecificFrame(frameId);
		//FrameReturn = PhoXiDevice->GetFrame(pho::api::PhoXiTimeout::Infinity);
		if (!FrameReturn) {
			std::cout << "[!] Failed to retrieve the frame!" << std::endl;
			return false;
		}
		std::cout << "[*] A new frame is captured." << std::endl;
		//PhoXiDevice->StopAcquisition();
		return true;
	}

	bool checkFrame(const pho::api::PFrame& FrameIn) {
		if (FrameIn->Empty()) {
			std::cout << "Frame is empty.";
			return false;
		}
		if ((FrameIn->DepthMap.Empty()) || (FrameIn->Texture.Empty()) ||
			(FrameIn->PointCloud.Empty()) || (FrameIn->NormalMap.Empty()))
			return false;
		return true;
	}

	// added by hao chen 20220318
	bool loadCalibration(const std::string& calibpath) {
		// check if calibration file can be loaded
		std::ifstream stream(calibpath);
		if (!stream.good())
			return false;
		// cehck if calibration file is correct
		calibration.LoadFromFile(calibpath);
		auto isCorrect = true
				&& calibration.CalibrationSettings.DistortionCoefficients.size() > 4
				&& calibration.CameraResolution.Width != 0
				&& calibration.CameraResolution.Height != 0;
		if (!isCorrect)
			return false;
		return true;
	}

	bool configExternalCam() {
		cap.open(cv::CAP_DSHOW);
		if (!cap.isOpened())
		{
			std::cout << "[!] ERROR: Can't initialize camera capture" << std::endl;
			return false;
		}
		cap.set(cv::CAP_PROP_FRAME_WIDTH, 3840);
		cap.set(cv::CAP_PROP_FRAME_HEIGHT, 2160);
		cap.set(cv::CAP_PROP_AUTOFOCUS, 0);
		cap.set(cv::CAP_PROP_FOCUS, 0);
		std::cout << "[*] Successfully initialize camera capture" << std::endl;
		return true;
	}

	bool getColorImage() {
		if (isexternalcamconnected) {
			cap.read(externalcamimage);
			cap.read(externalcamimage);
		}
		else
			return false;
		return true;
	}

	bool setColorPointCloudTexture() {
		if (isframeobtained && isexternalcamconnected) {
			auto TextureSize = frame->GetResolution();
			// Set the deafult value RGB(0,0,0) of the texture
			cv::Mat cvTextureRGB =
				cv::Mat(TextureSize.Height,
					TextureSize.Width,
					CV_8UC3,
					cv::Scalar(0., 0., 0.));
			// Zero-point
			pho::api::Point3_32f ZeroPoint(0.0f, 0.0f, 0.0f);
			// Parameters of computation-----------------------------------------
			cv::Mat MCWCMatrix = cv::Mat(4, 4, cv::DataType<float>::type);
			cv::Mat trans = cv::Mat::eye(4, 4, cv::DataType<float>::type);
			// Set 'trans' matrix == rotation and translation together in 4x4 matrix
			const pho::api::RotationMatrix64f& transformRotation =
				calibration.CoordinateTransformation.Rotation;
			for (int y = 0; y < transformRotation.Size.Height; ++y) {
				for (int x = 0; x < transformRotation.Size.Width; ++x) {
					trans.at<float>(y, x) = (float)transformRotation[y][x];
				}
			}
			trans.at<float>(0, 3) = (float)calibration.CoordinateTransformation.Translation.x;
			trans.at<float>(1, 3) = (float)calibration.CoordinateTransformation.Translation.y;
			trans.at<float>(2, 3) = (float)calibration.CoordinateTransformation.Translation.z;
			// Set MCWCMatrix to the inverse of 'trans'
			MCWCMatrix = trans.inv();

			// Set projection parameters from CameraMatrix of the external camera
			float fx, fy, cx, cy;
			fx = (float)calibration.CalibrationSettings.CameraMatrix[0][0];
			fy = (float)calibration.CalibrationSettings.CameraMatrix[1][1];
			cx = (float)calibration.CalibrationSettings.CameraMatrix[0][2];
			cy = (float)calibration.CalibrationSettings.CameraMatrix[1][2];

			// Set distortion coefficients of the external camera
			float k1, k2, p1, p2, k3;
			k1 = (float)calibration.CalibrationSettings.DistortionCoefficients[0];
			k2 = (float)calibration.CalibrationSettings.DistortionCoefficients[1];
			p1 = (float)calibration.CalibrationSettings.DistortionCoefficients[2];
			p2 = (float)calibration.CalibrationSettings.DistortionCoefficients[3];
			k3 = (float)calibration.CalibrationSettings.DistortionCoefficients[4];

			// Set the resolution of external camera
			int width, height;
			width = calibration.CameraResolution.Width;
			height = calibration.CameraResolution.Height;
			// End of setting the parameters--------------------------------------

			// Loop through the PointCloud
			for (int y = 0; y < frame->PointCloud.Size.Height; ++y) {
				for (int x = 0; x < frame->PointCloud.Size.Width; ++x) {
					// Do the computation for a valid point only
					if (frame->PointCloud[y][x] != ZeroPoint) {

						// Point in homogeneous coordinates
						cv::Mat vertexMC = cv::Mat(4, 1, cv::DataType<float>::type);
						vertexMC.at<float>(0, 0) =
							frame->PointCloud[y][x].x;
						vertexMC.at<float>(1, 0) =
							frame->PointCloud[y][x].y;
						vertexMC.at<float>(2, 0) =
							frame->PointCloud[y][x].z;
						vertexMC.at<float>(3, 0) = 1;

						// Perform the transformation into the coordinates of external camera
						cv::Mat vertexWC = MCWCMatrix * vertexMC;

						// Projection from 3D to 2D
						cv::Mat camPt = cv::Mat(2, 1, cv::DataType<float>::type);
						camPt.at<float>(0, 0) = vertexWC.at<float>(0, 0) / vertexWC.at<float>(2, 0);
						camPt.at<float>(1, 0) = vertexWC.at<float>(1, 0) / vertexWC.at<float>(2, 0);

						// The distortion of the external camera need to be taken into account for details see e.g.
						// https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
						pho::api::float32_t xx, xy2, yy, r2, r4, r6;

						xx = camPt.at<float>(0, 0) * camPt.at<float>(0, 0);
						xy2 = 2 * camPt.at<float>(0, 0) * camPt.at<float>(1, 0);
						yy = camPt.at<float>(1, 0) * camPt.at<float>(1, 0);

						r2 = xx + yy;
						r4 = r2 * r2;
						r6 = r4 * r2;

						// Constant related to the radial distortion
						pho::api::float32_t c = (1 + k1 * r2 + k2 * r4 + k3 * r6);

						// Both radial and tangential distortion are applied
						cv::Mat dist = cv::Mat(2, 1, cv::DataType<float>::type);
						dist.at<float>(0, 0) = c * camPt.at<float>(0, 0) +
							p1 * xy2 + p2 * (r2 + 2 * xx);
						dist.at<float>(1, 0) = c * camPt.at<float>(1, 0) +
							p1 * (r2 + 2 * yy) + p2 * xy2;

						// Final film coordinates
						cv::Mat position = cv::Mat(4, 1, cv::DataType<float>::type);
						position.at<float>(0, 0) = (dist.at<float>(0, 0) * fx + cx);
						position.at<float>(1, 0) = (dist.at<float>(1, 0) * fy + cy);
						position.at<float>(2, 0) = (vertexWC.at<float>(2, 0) - 5000) / 5000;
						position.at<float>(3, 0) = 1.;

						//(i,j) -> screen space coordinates
						int i = (int)std::round(position.at<float>(0, 0));
						int j = (int)std::round(position.at<float>(1, 0));

						if (i >= 0 && i < width && j >= 0 && j < height) {
							// The loaded extCameraImage has channels ordered like BGR
							auto yr = cvTextureRGB.ptr<uint8_t>(y);
							auto jr = externalcamimage.ptr<uint8_t>(j);

							// Set R - 0th channel
							yr[3 * x + 0] = jr[3 * i + 2];
							// Set G - 1st channel
							yr[3 * x + 1] = jr[3 * i + 1];
							// Set B - 2nd channel
							yr[3 * x + 2] = jr[3 * i + 0];
						}
					}
				}
			}

			pho::api::Mat2D<pho::api::ColorRGB_32f> textureRGB(TextureSize);
			ConvertOpenCVMatToMat2D(cvTextureRGB, textureRGB);
			frame->TextureRGB = textureRGB;
			return true;
		}
		else {
			std::cout << "Grap a frame first!" << std::endl;
			return false;
		}
	}



	/// <summary>
	/// connec to the sensor, and save the cam to a global variable
	/// </summary>
	/// <param name="serialno"> serial number </param>
	/// <param name="portno"> port number </param>
	/// <param name="resolution"> resolution "low" or "high" </param>
	/// <returns> phoxi cam object </returns>
	///
	/// <author> weiwei </author>
	/// <date> 20191128 </date>
	/// <param name="calibpath"> file path for the calibration file</param>
	/// <author> hao chen </author>
	/// <date> 20220318 </date>
	bool connect(std::string serialno, unsigned int portno, std::string resolution, const std::string& calibpath) {
		if (isconnected) {
			std::cout << "[!] The device has been connected. There is no need to connect again." << std::endl;
			return true;
		}
		if (resolution != "high" && resolution != "low") {
			std::cout << "[!] Resolution must be one of [\"low\", \"high\"]." << std::endl;
			return false;
		}
		// check if any connected device matches the requested serial number
		pho::api::PhoXiFactory factory;
		bool isFound = isDeviceAvailable(factory, serialno);
		if (!isFound) {
			std::cout << "[!] Requested device (serial number: " << serialno << ") not found!" << std::endl;
			return false;
		}
		// connect to the device
		cam = connectToDevice(factory, serialno);
		if (!cam->isConnected()) {
			std::cout << "[!] Could not connect to device." << std::endl;
			return false;
		}
		else {
			std::cout << "[*] Successfully connected to device." << std::endl;
			isconnected = true;
		}
		// configure the device
		configDevice(cam, resolution);
		// calibration file path
		if (!calibpath.empty()) {
			if (loadCalibration(calibpath)) {
				std::cout << "[*] Successfully load calibration matrix" << std::endl;
				if (configExternalCam()) {
					iscalibloaded = true;
					isexternalcamconnected = true;
				}
			}
			else {
				std::cout << "[!] Fail to load calibration maxtrix. File path error. Color point cloud functionality stops." << std::endl;
			}

		}
		else {
			std::cout << "[!] Calibration file path does not be specified. Color point cloud functionality stops." << std::endl;
		}

		return true;
	}

public:
	/// <summary>
	/// constructor
	/// </summary>
	/// <param name="serialno"></param>
	/// <param name="portno"></param>
	/// <param name="resolution"></param>
	/// <param name="calibpath"> file path for the calibration file</param>
	/// <returns></returns>
	PhoxiControl(std::string serialno, unsigned int portno, std::string resolution, std::string calibpatth) {
		connect(serialno, portno, resolution, calibpatth);
	}

	/// <summary>
	/// destructor
	/// </summary>
	/// <param name="serialno"></param>
	/// <param name="portno"></param>
	/// <param name="resolution"></param>
	/// <returns></returns>
	~PhoxiControl() {
		if (isconnected) {

		}
	}

	/// <summary>
	/// capture a frame, and save it to the global frame variable
	/// </summary>
	/// <returns></returns>
	///
	/// <author> weiwei </author>
	/// <date> 20191128 </date>
	bool captureframe() {
		bool success = getFrame(cam, frame);
		if (checkFrame(frame) && success) {
			isframeobtained = true;
			frameid += 1;
			std::cout << "A new frame is obtained. FrameID: " << frameid << std::endl;
			//added by chen 20220318
			if (getColorImage()) {
				if (setColorPointCloudTexture()) {
					std::cout << "Color texture generated successfully. FrameID: " << frameid << std::endl;
				}
			}
			return true;
		}
		else {
			return false;
		}
	}

	int getframeid() {
		if (isframeobtained) {
			return frameid;
		}
		else {
			std::cout << "Grap a frame first!" << std::endl;
			return -1;
		}
	}

	int getframewidth() {
		if (isframeobtained) {
			return frame->DepthMap.Size.Width;
		}
		else {
			std::cout << "Grap a frame first!" << std::endl;
			return 0;
		}
	}

	int getframeheight() {
		if (isframeobtained) {
			return frame->DepthMap.Size.Height;
		}
		else {
			std::cout << "Grap a frame first!" << std::endl;
			return 0;
		}
	}

	unsigned long getdepthmapdatasize() {
		if (isframeobtained) {
			return frame->DepthMap.GetDataSize() / sizeof(float);
		}
		else {
			std::cout << "Grap a frame first!" << std::endl;
			return 0;
		}
	}

	std::vector<float> gettexture() {
		if (isframeobtained) {
			unsigned long datasize = getdepthmapdatasize();
			float* textureptr = (float*)frame->Texture.GetDataPtr();
			return std::vector<float>(textureptr, textureptr + datasize);
		}
		else {
			std::cout << "Grap a frame first!" << std::endl;
		}
	}

	std::vector<float> getdepthmap() {
		if (isframeobtained) {
			unsigned long datasize = getdepthmapdatasize();
			float* depthmapptr = (float*)frame->DepthMap.GetDataPtr();
			return std::vector<float>(depthmapptr, depthmapptr + datasize);
		}
		else {
			std::cout << "Grap a frame first!" << std::endl;
		}
	}

	std::vector<float> getpointcloud() {
		if (isframeobtained) {
			int height = frame->PointCloud.Size.Height;
			int width = frame->PointCloud.Size.Width;
			std::vector<float> result(height * width * 3);
			for (int i = 0; i < height; i++) {
				for (int j = 0; j < width; j++) {
					result[i * width * 3 + j * 3 + 0] = frame->PointCloud[i][j].x;
					result[i * width * 3 + j * 3 + 1] = frame->PointCloud[i][j].y;
					result[i * width * 3 + j * 3 + 2] = frame->PointCloud[i][j].z;
				}
			}
			return result;
		}
		else {
			std::cout << "Grap a frame first!" << std::endl;
		}
	}

	std::vector<float> getnormals() {
		if (isframeobtained) {
			int height = frame->NormalMap.Size.Height;
			int width = frame->NormalMap.Size.Width;
			std::vector<float> result(height * width * 3);
			for (int i = 0; i < height; i++) {
				for (int j = 0; j < width; j++) {
					result[i * width * 3 + j * 3 + 0] = frame->NormalMap[i][j].x;
					result[i * width * 3 + j * 3 + 1] = frame->NormalMap[i][j].y;
					result[i * width * 3 + j * 3 + 2] = frame->NormalMap[i][j].z;
				}
			}
			return result;
		}
		else {
			std::cout << "Grap a frame first!" << std::endl;
		}
	}

	bool saveply(const std::string filepath, int xmin, int xmax, int ymin, int ymax, int zmin, int zmax) {
		if (isframeobtained) {
			int height = frame->PointCloud.Size.Height;
			int width = frame->PointCloud.Size.Width;
			for (int i = 0; i < height; i++) {
				for (int j = 0; j < width; j++) {
					float x = frame->PointCloud[i][j].x;
					float y = frame->PointCloud[i][j].y;
					float z = frame->PointCloud[i][j].z;
					if ((x < xmin || x > xmax) || (y < ymin || y > ymax) || (z < zmin || z > zmax)) {
						frame->PointCloud[i][j].x = 0;
						frame->PointCloud[i][j].y = 0;
						frame->PointCloud[i][j].z = 0;
					}
				}
			}
			return frame->SaveAsPly(filepath, true, false, true, false, false, false, false, false);
		}
		else {
			std::cout << "Grap a frame first!" << std::endl;
			return false;
		}
	}

	//added by chen 20220318
	std::vector<uchar> getexternalcamimg() {
		if (isexternalcamconnected) {
			cv::Mat flat = externalcamimage.reshape(1, externalcamimage.total() * externalcamimage.channels());
			return externalcamimage.isContinuous() ? flat : flat.clone();
		}
		else {
			std::cout << "Fail to get rgb texture" << std::endl;
		}
	}

	std::vector<float> getrgbtexture() {
		if (isframeobtained && isexternalcamconnected) {
			unsigned long datasize = frame->TextureRGB.GetDataSize();
			float* textureptr = (float*)frame->TextureRGB.GetDataPtr();
			return std::vector<float>(textureptr, textureptr + datasize/4);
		}
		else {
			std::cout << "Fail to get rgb texture" << std::endl;
		}
	}

	/// <summary> DEPRECATED, for it needs the commercial dongle and only compiles with VS2013
	/// locate a mesh model from a pointcloud
	/// </summary>
	/// <param name="cfg_filename">file generated by the phoxi3d locator configurator</param>
	/// <param name="sceneply_filename">ply file saved using the phoxicontrol saveply python interface</param>
	/// <param name="timeout">stop criteria -- max matching time</param>
	/// <param name="nresults">stop criteria -- max number of candidates</param>
	/// <returns>[x, y, z, r00, r01, r02, r10, r11, r12, r20, r21, r22, [], ...]</returns>
	///
	/// <author> weiwei </author>
	/// <date> 20191202 </date>
//	std::vector<float> findmodel(std::string cfg_filename, int timeout, int nresults) {
//		std::unique_ptr<pho::sdk::PhoLocalization> Localization;
//		try {
//			std::cout << "bad localization1" << std::endl;
//			Localization.reset(new pho::sdk::PhoLocalization());
//			std::cout << "bad localization2" << std::endl;
//		}
//		catch (const pho::sdk::AuthenticationException &ex) {
//			std::cout << ex.what() << std::endl;
//		}
//		pho::sdk::SceneSource Scene;
//		try {
//			std::cout << "bad scence source1" << std::endl;
//			Scene = pho::sdk::SceneSource::PhoXi(cam);
//			std::cout << "bad scence source2" << std::endl;
//		}
//		catch (const pho::sdk::PhoLocalizationException &ex) {
//			std::cout << "bad scence source3" << std::endl;
//			std::cout << ex.what() << std::endl;
//		}
//		Localization->LoadLocalizationConfiguration(cfg_filename);
//		Localization->SetSceneSource(Scene);
//		//// Setting stop criteria manually
//		Localization->SetStopCriterion(pho::sdk::StopCriterion::Timeout(timeout));
//		Localization->SetStopCriterion(pho::sdk::StopCriterion::NumberOfResults(nresults));
//		pho::sdk::AsynchroneResultQueue AsyncQueue = Localization->StartAsync(frame);
//		pho::sdk::TransformationMatrix4x4 Transform;
//		std::vector<float> results;
//		while (AsyncQueue.GetNext(Transform)) {
//			results.push_back(Transform[0][3]);
//			results.push_back(Transform[1][3]);
//			results.push_back(Transform[2][3]);
//			for(int i=0; i<3; i++) {
//				for(int j=0; j<3; j++) {
//					results.push_back(Transform[i][j]);
//				}
//			}
//		}
//		return results;
//	}
};

/// <summary>
/// python interface
/// </summary>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // vector
#include <pybind11/operators.h>//operator

namespace py = pybind11;

PYBIND11_MODULE(phoxicontrol, m) {
	py::class_<PhoxiControl>(m, "PhoxiControl")
		.def(py::init<std::string, unsigned int, std::string, std::string>())
		.def("captureframe", &PhoxiControl::captureframe)
		.def("getframeid", &PhoxiControl::getframeid)
		.def("getframewidth", &PhoxiControl::getframewidth)
		.def("getframeheight", &PhoxiControl::getframeheight)
		.def("getdepthmapdatasize", &PhoxiControl::getdepthmapdatasize)
		.def("gettexture", &PhoxiControl::gettexture)
		.def("getdepthmap", &PhoxiControl::getdepthmap)
		.def("getpointcloud", &PhoxiControl::getpointcloud)
		.def("getnormals", &PhoxiControl::getnormals)
		.def("saveply", &PhoxiControl::saveply)
		.def("getrgbtexture", &PhoxiControl::getrgbtexture)
		.def("getexternalcamimg", &PhoxiControl::getexternalcamimg);
	//		.def("findmodel", &PhoxiControl::findmodel);
}