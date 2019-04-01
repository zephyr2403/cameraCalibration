import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

# preparing object points, image contains 8x6 intersections black and white squares
objp = np.zeros((6*8,3), np.float32)
objp[:,:2] = np.mgrid[0:8, 0:6].T.reshape(-1,2)

'''
Camera Calibration is mapping 3D world point to 2D image 
'''
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('images/GO*.jpg')

# Step through the list and search for chessboard corners
for imageName in images:
    image = cv2.imread(imageName)
    grayImage= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(grayImage, (8,6), None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(image, (8,6), corners, ret)

        cv2.imshow('Detected Chessboard Corners', image)
        cv2.waitKey(500)

cv2.destroyAllWindows()


# loading distorted image
testImg = cv2.imread('images/test_image.jpg')

# Do camera calibration given object points and image points
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (testImg.shape[1], testImg.shape[0]),None,None)

#undistorting the image via calibration matrix 
dst = cv2.undistort(testImg, mtx, dist, None, mtx)
#Saving the undistorted image
cv2.imwrite('images/test_undist.jpg',dst)

# Visualize undistortion
_, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.imshow(testImg)
ax1.set_title('Original Image', fontsize=30)
ax2.imshow(dst)
ax2.set_title('Undistorted Image', fontsize=30)
plt.show()