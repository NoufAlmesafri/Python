import numpy as np
import cv2
import sys, time, math
import pyrealsense2 as rs
import keyboard     # using module keyboard
import math
from scipy.spatial.transform import Rotation as R
import cv2.aruco as aruco

def ArUco():

    #----[Setting the camera]----
    # Configure color streams
    pipeline = rs.pipeline()
    config = rs.config()
    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    
    
    #----[Intrensic Parameters]----
    #color_profile = rs.video_stream_profile(pipeline_profile.get_stream(rs.stream.color))
    #color_intrinsics = color_profile.get_intrinsics()
    #w, h = color_intrinsics.width, color_intrinsics.height


    #---[Find RGB]---
    found_rgb = False   #Intial Value
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    #----[ArUco Detection Code]----
    # Start streaming
    pipeline.start(config)


    #---[Parametrs]---
    marker_size= 9              #[cm]
    Camera_matrix=np.array([[611.301, 0, 320.367],[0, 611.506, 246.129],[0,0,1]])
    Distortion_Coefficients=np.array([0.,    0.,   0.,   0.,    0.])


    while True:
        #---Aruco Dictionary
        arucodic=aruco.getPredefinedDictionary(aruco.DICT_6X6_1000)
        parameters=aruco.DetectorParameters_create()
        parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        parameters.cornerRefinementWinSize = 5
        parameters.cornerRefinementMinAccuracy = 0.001
        parameters.cornerRefinementMaxIterations = 5
        #--Read Camera Frame
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame() # not used
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue
        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        #--Convert to gray scale
        gray_image=cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        #---Find aruco markers
        corners, ids, rejected=  aruco.detectMarkers(image=gray_image, dictionary=arucodic, parameters=parameters)
        if ids != None :
            # the following line is not right 
            Pose_Vector= aruco.estimatePoseSingleMarkers(corners,marker_size, Camera_matrix, Distortion_Coefficients)
            #---unpack output
            Rotation_Vector = Pose_Vector[0][0][0]
            Translation_Vector = Pose_Vector[1][0][0]
            # Convert Vector r and (180,0,0) to Rotation Matrix
            Rotation_Vector_Matrix = R.from_rotvec(Rotation_Vector)
            Rot_matrix = R.from_rotvec([3.14,0,0])
            Rotation_Vector_Matrix = Rotation_Vector_Matrix.as_matrix()
            Rot_matrix = Rot_matrix.as_matrix()
            # Add the Rotation Matrices
            rotation_matrix_final = np.multiply(Rotation_Vector_Matrix, Rot_matrix)
            # Convert it back to vector
            r = R.from_matrix(rotation_matrix_final)
            Rotation_Vector = r.as_euler('zyx', degrees=True)
            Translation_Vector = np.multiply(Translation_Vector, 10) # cm 2 mm
            aruco_position = np.concatenate((Translation_Vector,Rotation_Vector), axis=0)
            # print (rvec)
            # print (tvec)
            cv2.imshow("test", cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
            cv2.waitKey(1)
            break
        else:
            cv2.imshow("test", cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
            cv2.waitKey(1)
    # Stop streaming
    pipeline.stop()

    return (aruco_position)