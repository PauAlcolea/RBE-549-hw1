#!/usr/bin/env python
import sys
import numpy as np
from argparse import ArgumentParser
import os
import cv2
import matplotlib.pyplot as plt

# Some global variables
sq_sz = 21.5
h = 6
w = 9

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
def get_corners(images: list[np.ndarray]):
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
            # with_corners = cv2.drawChessboardCorners(im, (w, h), corners_original, ret)
            # cv2.imshow("cornered", with_corners)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
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


def get_intrinsic(corners_image, corners_world):
    # Solve for approximate K (camera calibration matrix)
    # Section 3.1 of paper
    # Use cv2.findChessboardCorners to find the corners of the checker baord with appropiate parameters
    four_corners_world = np.array([corners_world[0][0], corners_world[0][w-1], corners_world[h-1][0], corners_world[h-1][w-1]]).reshape(-1, 2)
    homographies = []

    for crn in corners_image:
        four_corners_image = np.array([crn[0][0], crn[0][w-1], crn[h-1][0], crn[h-1][w-1]]).reshape(-1, 2)

        # homogrpahy by comparing both corners in the world and the image
        H, mask = cv2.findHomography(four_corners_world, four_corners_image)
        homographies.append(H)

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

    # reconstructed B from b
    B = np.array([
        [b[0], b[1], b[3]],
        [b[1], b[2], b[4]],
        [b[3], b[4], b[5]]
    ])
    
    #construct K from B
    # From Appendix B of Zhang's paper
    B11 = B[0][0]
    B12 = B[0][1]
    B13 = B[0][2]
    B22 = B[1][1]
    B23 = B[1][2]
    B33 = B[2][2]
    
    cy = (B12 * B13 - B11 * B23) / (B11 * B22 - (B12**2))   # v0 in Zhang's paper
    l = B33 - (B13**2 + cy*(B12 * B13 - B11 * B23)) / B11   # lambda
    fx = np.sqrt(l / B11)                                   # alpha in Zhang's
    fy = np.sqrt( l*B11 / (B11 * B22 - (B12**2)))           # Beta in zhangs
    y = (-B12 * (fx**2) * fy) / l                           # Gamma
    cx = (y * cy / fy) - (B13 * (fx**2) / l)                # u0 in Zhang's paper
    
    K = np.array([
        [fx, y, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    
    return K, homographies, l

"""
Approximate R (rotation matrix) or t (translation of the camera)
section 3.1
neglect conversion from normal matrix to rotation matrix
"""
def get_extrinsics(homographies, imgs, K, l):
    # imshow rectified image
    pixels_per_mm = 5  # choose something reasonable
    rect_w = int((w - 1) * sq_sz * pixels_per_mm)
    rect_h = int((h - 1) * sq_sz * pixels_per_mm)
    # Translation to properly see the new image
    T = np.array([[1, 0, 200],[0, 1, 150],[0, 0, 1]])
    
    extrinsics = []
    for index, homography in enumerate(homographies):
        h1_2 = homography[:, 0]
        h2_2 = homography[:, 1]
        h3_2 = homography[:, 2]

        r1 = l * np.linalg.inv(K) @ h1_2
        r2 = l * np.linalg.inv(K) @ h2_2
        r3 = np.cross(r1, r2)
        t = l * np.linalg.inv(K) @ h3_2

        # print("r1", r1)
        # print("r2", r2)
        # print("r3", r3)
        # print("t", t)
        R = np.transpose(np.vstack((r1, r2, r3, t)))
        # print(R)
        extrinsics.append(R)

        # Rectification:
        H_rect = T @ np.linalg.inv(homography)
        warped = cv2.warpPerspective(imgs[index], H_rect, (rect_w, rect_h))
        # cv2.imshow(f"rectification for image {index}", warped)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows() 
    return extrinsics

"""
This function is supposed to convert the world points into each of the images coordinates
@param corners_image
@param corners_world
@return corners in the image
"""
def project_points(corners_world, K, extrinsics, k):
    # make the points in 3d to add the z
    world_points_3d = np.column_stack([corners_world, np.zeros(len(corners_world))])
    world_in_image = []
    for Rt in extrinsics:
        R = Rt[:,0:3]
        t = Rt[:, 3]

        projected_points = np.empty((len(corners_world), 2))

        for ind, point3d in enumerate(world_points_3d):
            # get point in camera view
            cam_point = R @ point3d + t
            
            # normalize the 
            x_norm = cam_point[0] / cam_point[2]
            y_norm = cam_point[1] / cam_point[2]

            # radial distortion
            k1, k2 = k[0], k[1]
            r2 = (x_norm ** 2) + (y_norm ** 2)
            rad = 1 + k1 * r2 + k2 * (r2**2)
            x_dist = x_norm * rad
            y_dist = y_norm * rad

            # Intrinsic Matrix for the pixel coordinates
            u = K[0][0] * x_dist + K[0][1] * y_dist + K[0][2]   #u = fx * x_dist + gamma * y_dist + cx
            v = K[1][1] * y_dist + K[1][2]                      #v = fy * y_dist + cy

            
            projected_points[ind] = [u, v]
        world_in_image.append(projected_points)
    return world_in_image

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

    reference = np.zeros((h, w, 2), dtype=np.float32)
    world2D = reference.reshape(-1,2)

    for i in range(h):
        for j in range(w):
            reference[i][j][0] = sq_sz * j # x 
            reference[i][j][1] = sq_sz * i # y 

    # Initialize all of the images that will be used for calibration, they are in dir = "/../Calib_Imgs" and get the points in the grid
    imgs = initialize_calib(images_dir)

    # iterate through all the images, get all of the chessboard corners and then extract the four actual corners to compare with the reference
    corners2D, corners = get_corners(imgs)

    K, homographies, l = get_intrinsic(corners, reference)

    extrinsics = get_extrinsics(homographies, imgs, K, l)

    # Approximate distortion k = [k1, k2]
    # k = [0, 0] is a good approximation for right now
    k1 = 0
    k2 = 0
    k = np.transpose(np.array([k1, k2]))

    # Non-Linear Geometric Error Minimization
    # ∑i=1N∑j=1M||xi,j−x̂ i,j(K,Ri,ti,Xj,k)||
    # use scipy.optimize to minimize the function
    # section 3.3
    # rotate the real "world" points into the camera's orientation
    
    # pack all of the parameters into one vector
    # Global ones: fx, fy, cx, cy, k1, k2
    # Local ones: r11, r12, r13, t11, t12, t13, r21, r22, r23, t21, t22, t23
    # Compute reprojection errors

    # input_vector = [K, R, k]
    # make a function that takes world points and puts them onto every image
    world_in_image = project_points(world2D, K, extrinsics, k)
    loss_list = [(a - b) for a, b in zip(corners2D, world_in_image)]

    for index, im in enumerate(imgs):
        # Visualize first image
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        # Show the image
        ax.imshow(im, cmap='gray')

        # Plot projected corners (what your model predicts)
        projected_x = world_in_image[index][:, 0]
        projected_y = world_in_image[index][:, 1]
        ax.scatter(projected_x, projected_y, c='red', s=30, marker='x', label='Projected World to Image')

        ax.legend()
        ax.set_title('World Points projected onto images')
        plt.show()



if __name__ == "__main__":
    main()
