"""
NEED USB3 cameras to collect images simultaneously
"""

import cv2

if __name__ == '__main__':

    # cams_test = 10
    # for i in range(cams_test):
    #     cap = cv2.VideoCapture(i)
    #     test, frame = cap.read()
    #     print("i : "+str(i)+" /// result: "+str(test))

    cap = []
    cap.append(cv2.VideoCapture(0))
    cap.append(cv2.VideoCapture(2))
    cap.append(cv2.VideoCapture(4))

    flag = 0
    while(True):
        frame1 = cap[0].read()[1]
        frame2 = cap[1].read()[1]
        frame3 = cap[2].read()[1]

        if frame1 is not None:
            cv2.imshow('center', frame1)
            if flag == 0:
                cv2.moveWindow('center', 450, 200)

        if frame2 is not None:
            cv2.imshow('lft', frame2)
            if flag == 0:
                cv2.moveWindow('lft', 100, 800)

        if frame3 is not None:
            cv2.imshow('rgt', frame3)
            if flag == 0:
                cv2.moveWindow('rgt', 800, 800)

        flag = 1
        k = cv2.waitKey(1)
        if k == 27:
            break
    cv2.destroyAllWindows()

