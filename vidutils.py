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
    if n_frames < 0:
        wanted_frames = range(frame_total)
    else:
        wanted_frames = np.linspace(1, frame_total - 1, n_frames, dtype = np.int16)

    # Get number of frames per second
    fps = cap.get(cv2.CAP_PROP_FPS)

    # a timeline is essentially a mapping from a frame index to time in seconds
    timeline = []

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

            timeline.append(frame_idx/fps)

            frame_count += 1

    cap.release()

    # saving the constructed timeline to file
    np.savez(frame_path + '/timeline.npz', timeline = np.array(timeline))

    return

def extract_pose_features_multi(frame_path, num_frames):
    '''
    Extract numpy feature array from 1 frame folder and save it in a file. 
    Feature Array has a shape of (n_frames, 168)
    Array is also saved as an .npz file for the RNN
    '''

    feature_array = []

    frames = os.listdir(frame_path)
    #print(len(frames))
    
    # Reorder the frames because the file names have confused them
    ordered_frames = [None] * num_frames # frame count
    f_count = 0
    for frame_cand in frames:
        if '.jpg' not in frame_cand:
            continue
        else: 
            idx = int(frame_cand.replace("frame", "")[0:-4])
            ordered_frames[idx] = frame_cand
            f_count += 1
    
    # If: < n_frames, duplicate the last frame
    if f_count < num_frames:
        print("*"*10)
        print("Warning: Frames less than {}. Duplicating.".format(num_frames))
        print("*"*10)
        for idx in range((num_frames-f_count)):
            ordered_frames[f_count + idx] = ordered_frames[f_count - 1]
    
    # Extract features for each frame
    for frame in ordered_frames:
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
    try: 
        candidate, subset = body_estimation(frame_img)
        person_count = subset.shape[0]
    except: 
        print("Warning: OpenPose failed. Adding row of -1's")
        person_count = 0

    # Just get the first 3 people in subset
    for person_idx in range(min(person_count,3)):

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
                
    # Handle case where OpenPose does not detect people - just use "-1" for the entire thing
    
    if person_count < 3 and len(feature_array) != 168:
        
        # Get the number of people missing
        num_missing = 3 - person_count
        
        print('*'*10)
        print("Warning: OpenPose recognizes {}/3 people in frame. Adding missing data.".format(person_count))
        print(frame_path)
        print('*'*10)
        
        # Insert array of -1 for the missing data and ensure feature array is the correct size
        missing_data = np.ones((num_missing*56,)) * -1
        feature_array = np.hstack((feature_array, missing_data))

    return feature_array

def reshape_syn_features(subset_all, candidate_all):
    '''
    Reshape features from those already extracted from openpose. 
    Subset and candidate should already be in a list of results per frame.
    Feature Array is a numpy array shaped (1, 168)
    '''

    total_features = []
    for frame in range(len(candidate_all)):
        subset = subset_all[frame]
        candidate = candidate_all[frame]

        # Body Estimation 
        feature_array = []
        try: 
            person_count = subset.shape[0]
        except: 
            print("Warning: OpenPose failed. Adding row of -1's")
            person_count = 0

        # Just get the first 3 people in subset
        for person_idx in range(min(person_count,3)):

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
                    
        # Handle case where OpenPose does not detect people - just use "-1" for the entire thing
        
        if person_count < 3 and len(feature_array) != 168:
            
            # Get the number of people missing
            num_missing = 3 - person_count
            
            print('*'*10)
            print("Warning: OpenPose recognizes {}/3 people in frame. Adding missing data.".format(person_count))
            print(frame_path)
            print('*'*10)
            
            # Insert array of -1 for the missing data and ensure feature array is the correct size
            missing_data = np.ones((num_missing*56,)) * -1
            feature_array = np.hstack((feature_array, missing_data))
    
        # append to total features list
        if len(total_features) == 0:
            total_features = feature_array
        else:
            total_features = np.vstack((total_features, feature_array))

    return total_features