# RBE-549-hw1
Computer Vision 540 Homework1 AutoCalib

To run the code, simply run the python script: Wrapper.py found under the Code directory
It takes an argument -i where you can write the directory of where the calibration images can be found, this is not necessary if they are under the Calib_imgs directory becaue that is the default for the argument

The script gets initial estiamtes for the intrisic matrix, extrinsics and distortion of the camera, and then uses them as initial parameters to run an optimization function. this outputs a set of parameters that is more accurate. The loss is calcualted with the difference between what the corners in the dirtorted image were found to be and the projection of the world points onto that image with the help of the extrincis, intrinsic matrix and the dirtortion.