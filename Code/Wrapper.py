#!/usr/bin/env python
import sys
import numpy as np
from argparse import ArgumentParser
import os
import cv2
import matplotlib.pyplot as plt

"""
This function will load the 13 data images
@param images_dir is a string containing the path to the directory for the calibration images
"""
def initialize_calib(images_dir=str):
    calib_images = []

    # go through all of the files in the calibration directory and return them as a list if they are jpgs
    for filename in os.listdir(images_dir):
        if filename.endswith(".jpg"):
            image = cv2.imread(os.path.join(images_dir, filename))
            if image is not None:
                image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                calib_images.append(image_gray)
    
    return calib_images

"""
get_corners takes a list of images and gets the corners
"""
def get_corners(images: list[np.ndarray], h, w):
    corners_list2D = []
    corners_list = []

    for im in images:
        # Shape (6,9) is inner grid and is in Height (y) and width (x)
        ret, corners_original = cv2.findChessboardCorners(im, (w, h), None)
        
        #if it returned the proper corners
        if ret == True:
            corners = corners_original.reshape((h, w, 2))
            corners2D = corners.reshape(-1,2)

            corners_list2D.append(corners2D)
            corners_list.append(corners)

            
            # # Visualize Corners
            with_corners = cv2.drawChessboardCorners(im, (w, h), corners_original, ret)
            cv2.imshow("cornered", with_corners)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Proper corners with specified grid shape have not been identified")
            continue

    return corners_list2D, corners_list

"""
Helper function to get the vij used for the b calculation
"""
def get_vij(hi:np.ndarray, hj:np.ndarray) -> np.ndarray:
    vij = np.array([hi[0]*hj[0],
                        hi[0]*hj[1] + hi[1]*hj[0],
                        hi[1]*hj[1],
                        hi[2]*hj[0] + hi[0]*hj[2],
                        hi[2]*hj[1] + hi[1]*hj[2],
                        hi[2]*hj[2]]).T
    
    return vij


# LLM Function to Visualize the corner alignments
def visualize_corners_alignment(img, corners2D, reference2D, H=None, show=True):
    """
    Visualize detected corners vs. reference points.
    
    Parameters:
        img : np.ndarray
            The original grayscale image
        corners2D : np.ndarray, shape (N,2)
            Detected chessboard corners from OpenCV
        reference2D : np.ndarray, shape (N,2)
            Reference world coordinates of corners
        H : np.ndarray, optional
            Homography to project reference points into image
        show : bool
            Whether to display the figure immediately
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    # Draw detected corners in red
    for pt in corners2D:
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(img_rgb, (x, y), 5, (255,0,0), -1)
    
    # Project reference points if H is provided
    if H is not None:
        ref_h = np.hstack([reference2D, np.ones((reference2D.shape[0],1))])
        proj_ref = (H @ ref_h.T).T
        proj_ref /= proj_ref[:,2].reshape(-1,1)
    else:
        proj_ref = reference2D
    
    # Draw reference points in green
    for pt in proj_ref:
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(img_rgb, (x, y), 5, (0,255,0), -1)
    
    if show:
        plt.figure(figsize=(10,8))
        plt.imshow(img_rgb[..., ::-1])  # convert BGR -> RGB for matplotlib
        plt.title("Red: detected corners, Green: projected reference corners")
        plt.axis("off")
        plt.show()
    
    return img_rgb


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


    # reference2D
    sq_sz = 21.5
    h = 6
    w = 9
    grid_shape = (h, w)

    reference = np.zeros((h, w, 2), dtype=np.float32)
    for i in range(h):
        for j in range(w):
            reference[i][j][0] = sq_sz * j # x 
            reference[i][j][1] = sq_sz * i # y 

    # get all of the points in the grid and then extract the four corners, those will be for homography calculation
    # reshape so that it is an array of arrays
    reference2D = reference.reshape(-1, 2)
    four_corners_world = np.array([reference[0][0], reference[0][w-1], reference[h-1][0], reference[h-1][w-1]]).reshape(-1, 2)

    # Initialize all of the images that will be used for calibration, they are in dir = "/../Calib_Imgs" and get the points in the grid
    imgs = initialize_calib(images_dir)

    # iterate through all the images, get all of the chessboard corners and then extract the four actual corners to compare with the reference
    homographies = []
    corners2D, corners = get_corners(imgs, h, w)


    for crn in corners:
        four_corners_image = np.array([crn[0][0], crn[0][w-1], crn[h-1][0], crn[h-1][w-1]]).reshape(-1, 2)

        # homogrpahy by comparing both corners in the world and the image
        H, mask = cv2.findHomography(four_corners_world, four_corners_image)
        homographies.append(H)


    # print(homographies[0])
    # Three homographies are enough to get the intrinsic values)
    V_rows = []
    for i in range(3):
        if i >= len(homographies):
            print("there are not enough images to find a fully constrained solution")
            break
        h_img = homographies[i]
        h1 = h_img[:, 0]
        h2 = h_img[:, 1]

        v12 = get_vij(hi=h1, hj=h2)
        v11 = get_vij(hi=h1, hj=h1)
        v22 = get_vij(hi=h2, hj=h2)

        V_rows.append(v12.T)
        V_rows.append((v11 - v22).T)


    V = np.vstack(V_rows)

    # Solve the equation Vb = 0 with svd
    # b is a 6D vector that represents B (it can represent it as a vector of only 6 beccause B is symmetric)
    # b = [B11, B12, B22, B13, B23, B33].T
    U, S, VT = np.linalg.svd(V)
    b = VT[-1]
    print(b)
    



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