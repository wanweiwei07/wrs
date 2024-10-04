import wrs.vision.ar_marker.make_pdfboard as mkpdf
import cv2.aruco as aruco

mkpdf.make_aruco_board(1,1, marker_dict=aruco.DICT_4X4_250, start_id=1, marker_size=200)