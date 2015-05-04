#! /usr/bin/python
import numpy as np
import cv2
import glob

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# images = glob.glob('our_calib_data_resize/*.jpg');
cap = cv2.VideoCapture(0)
img_count = 16;
while img_count != 31:

    # img = cv2.imread(fname)
    # print(fname)
    ret, img = cap.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (7,6),None)
    cv2.imshow('img',gray)
    cv2.waitKey(1)
    # If found, add object points, image points (after refining them)
    print(ret)
    if ret == True:
        objpoints.append(objp)

        cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners)
        print "enter yes or no"
        k = cv2.waitKey(0) & 0xFF
        if k == 27:
            continue
        elif k == 32:
            cv2.imwrite("just_for_fun/image_"+str(img_count)+".jpg", img);
            img_count = img_count + 1;


        # Draw and display the corners
        cv2.drawChessboardCorners(img, (7,6), corners,ret)
        cv2.imshow('img',img)
        cv2.waitKey(500)
    # break;    
    
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
print("ret : {0}".format(ret))
print("mtx : {0}".format(mtx))
print("dist : {0}".format(dist))
print("rvecs : {0}".format(rvecs))
print("tvecs : {0}".format(tvecs))

cv2.destroyAllWindows()