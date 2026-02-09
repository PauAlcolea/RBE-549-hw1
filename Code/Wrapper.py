#!/usr/bin/env python
import sys
import numpy as np
from argparse import ArgumentParser
import os
import cv2

"""
This function will load the 13 data images
@param images_dir is a string containing the path to the directory for the calibration images
"""
def initialize_calib(images_dir=str):
    calib_images = []

    # go through all of the files in the calibration directory and return them as a list if they are jpgs
    for filename in os.listdir(images_dir):
        if filename.endswith(".jpg"):
            image = cv2.imread((images_dir + filename))
            if image is not None:
                image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                calib_images.append(image_gray)
    
    return calib_images


def main():
    # get path to current directory
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(curr_dir)

    # Arguments
    parser = ArgumentParser()
    parser.add_argument(
        "-i", "--dir", type=str, default=f"{parent_dir}/Calib_Imgs/", help="Directory with all the images for calibration are"
    )
    args = parser.parse_args()
    images_dir = args.dir


    # calib_corners are the four corners of the unwarped actual checkerboard pattern, points go counter clockwise from buttom left 
    # sq_sz = 21.5
    # calib_x = 8
    # calib_y = 5



    # Initialize all of the images that will be used for calibration, they are in dir = "/../Calib_Imgs"   
    imgs = initialize_calib(images_dir)
    for im in imgs:
        ret, corners = cv2.findChessboardCorners(im, (9,6), None)
        if ret == True:
            with_corners = cv2.drawChessboardCorners(im, (9,6), corners, ret)
            cv2.imshow("cornered", with_corners)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    # Solve for approximate K (camera calibration matrix)
    # Section 3.1 of paper
    # Use cv2.findChessboardCorners to find the corners of the checker baord with appropiate paramet4rs

    # Approximate R (rotation matrix) or t (translation of the camera)
    # section 3.1
    # neglect conversion from normal matrix to rotation matrix

    # Approximate distortion k = [k1, k2]
    # k = [0, 0] is a good approximation for right now
    # k = np.transpose(np.array([0, 0]))

    # Non-Linear Geometric Error Minimization
    # ∑i=1N∑j=1M||xi,j−x̂ i,j(K,Ri,ti,Xj,k)||
    # use scipy.optimize to minimize the function
    # section 3.3




if __name__ == "__main__":
    main()