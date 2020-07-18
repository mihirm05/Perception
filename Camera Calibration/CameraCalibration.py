import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

np.set_printoptions(suppress=True)

# attempt to calculate the camera matrix for a smartphone camera

def calibration():
    # Defining the dimensions of checkerboard
    checkerboardPattern = (6, 9)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.0001)
    # print(criteria)

    # Creating vector to store vectors of 3D points for each checkerboard image
    objpoints = []
    # Creating vector to store vectors of 2D points for each checkerboard image
    imgpoints = []

    # Defining the world coordinates for 3D points
    objp = np.zeros((1, checkerboardPattern[0] * checkerboardPattern[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:checkerboardPattern[0], 0:checkerboardPattern[1]].T.reshape(-1, 2)

    # Extracting path of individual image stored in a given directory
    path = os.getcwd() + '/images/'
    images = glob.glob(path + 'c[0-9][0-9].jpg')

    for fname in images:
        plt.figure(figsize=(15, 15))
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        # If desired number of corners are found in the image then ret = true
        ret, corners = cv2.findChessboardCorners(gray, checkerboardPattern,
                                                 cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

        objpoints.append(objp)
        # refining pixel coordinates for given 2d points.
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, checkerboardPattern, corners, ret)
        plt.xlabel('Image Width')
        plt.ylabel('Image Height')
        plt.imshow(img)
        plt.title('Calibration')

    cv2.destroyAllWindows()

    # camera calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    img = cv2.imread(path + 'calibration3.jpg')

    plt.title('Distorted Image')
    plt.show()
    h, w = img.shape[:2]

    # get new camera matrix
    newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    # un-distort the input image
    undistortedImage = cv2.undistort(img, mtx, dist, None, newCameraMatrix)
    plt.figure(figsize=(15, 15))
    plt.imshow(undistortedImage)
    plt.title('Undistorted Image')

    print("Camera matrix (focal length in pixels) : \n")
    print(newCameraMatrix)
    print("distortion coefficients are: \n")
    print(dist)

    # conversion of focal length units (pixels to mm)
    sensorDimensionWidth = 4.71
    sensorDimensionHeight = 3.49
    imageWidth = img.shape[0]
    imageHeight = img.shape[1]

    # formula for conversion adopted from OpenCV forum
    newCameraMatrix[0][0] = newCameraMatrix[0][0] * sensorDimensionWidth / imageWidth
    newCameraMatrix[1][1] = newCameraMatrix[1][1] * sensorDimensionHeight / imageHeight

    print("Camera matrix (focal length in mm) : \n")
    print(newCameraMatrix)

    # projection error calculation (object points to image points)
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    print("total error: {}".format(mean_error / len(objpoints)))

    # generating rotation matrix using rodrigues function
    rotation_mat = np.zeros(shape=(3, 3))
    R = cv2.Rodrigues(rvecs[0], rotation_mat)[0]
    P = np.column_stack((np.matmul(mtx, R), tvecs[0]))
    # print('R is: \n', R)
    print('P matrix is: \n', P)


def main():
    calibration()


if __name__ == '__main__':
    main()
