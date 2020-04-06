import cv2
import numpy as np
import os
from pytorch_openpose.src.body import Body 

# import body estimation model
body_estimation = Body('pytorch_openpose/model/body_pose_model.pth')

def extract_frames(video_path, frame_path, n_frames = 16):
    '''
    Extract n frames from video in video_path and store in frame_path
    '''

    # Create Folder in frame_path 
    os.makedirs(frame_path, exist_ok = True)

    # Create Video Capture
    cap = cv2.VideoCapture(video_path)

    # Get Total Frame Count
    frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Get N frames from video
    wanted_frames = np.linspace(1, frame_total - 1, n_frames, dtype = np.int16)

    # Read each frame
    frame_count = 0
    for frame_idx in range(frame_total):

        success, frame = cap.read()

        if success is False: # Error catching
            continue

        # If in the list
        if (frame_idx in wanted_frames):
            
            # Save current frame in a jpg file
            fname = frame_path + '/frame' + str(frame_count) + '.jpg'
            cv2.imwrite(fname, frame)

            frame_count += 1

    cap.release()
    return

def extract_pose_features_multi(frame_path):
    '''
    Extract numpy feature array from 1 frame folder and save it in a file. 
    Feature Array has a shape of (n_frames, 168)
    Array is also saved as an .npz file for the RNN
    '''

    feature_array = []

    frames = os.listdir(frame_path)
    #print(len(frames))
    
    # Extract features for each frame
    for frame in frames:
        f_path = os.path.join(frame_path, frame)
        frame_features = extract_pose_features(f_path)

        if len(feature_array) == 0:
            feature_array = frame_features
        else:
            #print(frame)
            #print("frame_features: " + str(frame_features.shape))
            #print("feature_array: " + str(feature_array.shape))
            feature_array = np.vstack((feature_array, frame_features))
    
    # Save full feature array 
    np.savez(frame_path + '/features', feature_array=feature_array)

    return feature_array

def extract_pose_features(frame_path):
    '''
    Extract pose features via open pose for 1 video frame
    Feature Array is a numpy array shaped (1, 168)
    '''

    feature_array = []

    # Import image
    frame_img = cv2.imread(frame_path)

    # Body Estimation 
    candidate, subset = body_estimation(frame_img)

    # Just get the first 3 people in subset - might need to improve this
    for person_idx in range(3):

        person = subset[person_idx]
    
        for point_idx in range(len(person)):
            #print("point_idx: " + str(person[point_idx]))
            
            if point_idx < 18:
                candidate_idx = int(person[point_idx])
                #print("candidate point #" + str(candidate_idx))
                
                point_coord = np.ones((3,))
                
                if candidate_idx < 0:
                    #print("Occluded")
                    point_coord = point_coord * -1
                else:
                    point_coord = candidate[candidate_idx,:3]
                    #print(point_coord)
                    
                #print(point_coord)
                feature_array = np.hstack((feature_array, point_coord))
            else:
                # Add score and total parts 
                #print(person[point_idx])
                feature_array = np.hstack((feature_array, person[point_idx]))

    return feature_array