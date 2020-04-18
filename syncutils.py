import numpy as np
import math
import os

import cv2
from scipy import interpolate
from scipy.ndimage.filters import gaussian_filter1d

import json

#----------------------------------------------------------------------------------------------------

class limb:

    # `p0` and `p1` are of the form `np.array([x, y])`
    def __init__(this, p0, p1):
        this.p0 = p0
        this.p1 = p1

    # angle between the limb and the vertical line (in radians)
    def angle(this):
        dp = this.p0 - this.p1
        return math.atan2(abs(dp[0]), dp[1])


class body:

    """
    # class members:

    root:       point    # is used only for sorting different bodies in space (point(x, y) is np.array([x, y]))
    limbs:      dict     # mapping {limb_name -> limb_instance}

    limb_nodes: dict     # [static member] mapping {limb_name -> pair of indices of corresponding keypoints}

    """

    limb_nodes = \
    {
        'neck':           (1,  0),

        'right_shoulder': (1,  2),
        'right_forearm':  (2,  3),
        'right_arm':      (3,  4),
        'right_thigh':    (8,  9),
        'right_leg':      (9,  10),

        'left_shoulder':  (1,  5),
        'left_forearm':   (5,  6),
        'left_arm':       (6,  7),
        'left_thigh':     (11, 12),
        'left_leg':       (12, 13)
    }

    # `current_subset` is an element of `subset` for current body;
    # `candidate` is a full list of keypoints returned by `body_estimation`;
    #
    # note: if `current_subset` and/or `candidate` is None the created `body` is 'empty':
    #       `limbs` will contain every limb name but all of them will map to `None`
    #
    def __init__(this, candidate = None, current_subset = None):

        #helper functions
        
        # index `idx` represents a keypoint in `current_subset`;
        # returns an array of the form `[x, y]`
        #
        # note: if the keypoint refered to by `idx` is absent
        #       returns `None`
        #
        def get_point_by_index(idx):

            if current_subset is not None and candidate is not None:

                # remapping 'index of a keypoint' -> 'index of the corresponding keypoint coordinates'
                idx = int(current_subset[idx])

                if idx == -1:
                    return None
                else:
                    return candidate[idx][0:2]
            else:
                return None

        # returns a `limb` constructed using points refered by keypoint indices `idx1` and `idx2`:
        # 
        #  -> limb(p1, p2)
        #
        #  where:
        #         `p1` and `p2` are of the form `[x, y]` and denote a keypoint position
        #
        #  note: if any of the keypoints refered to by `idx1` or `idx2` is absent
        #        returns `None`
        #
        def create_limb(idx1, idx2):
           
            p1 = get_point_by_index(idx1)
            p2 = get_point_by_index(idx2)

            if p1 is None or p2 is None:
                return None
            else:
                return limb(p1, p2)

        #------------------------------------------------------------------------

        # the body root is assumed to be the keypoint 1 (bottom of the neck)
        this.root = get_point_by_index(1)
        
        this.limbs = {}

        for limb_name in body.limb_nodes:
            
            l = create_limb(*body.limb_nodes[limb_name])

            if l is not None:
                this.limbs[limb_name] = l


class dancer:

    """
    # class members:

    limbs: dict     # mapping {limb_name -> angle(time)}

    """

    # `body_instances` is an array of different time instances of `body` for a particular dancer
    # (some of the instances may be missing (~ None-like objects) so that `body_instances` is alligned with the `timeline`)
    #
    # `timeline` is an array containing time moment for each element of `body_instances`;
    # (`timeline` and `body_instances` are required to have the same length)
    #
    # note: if no `timeline` is present it is created as a range(len(`body_instances`))
    #
    def __init__(this, body_instances = [], timeline = []):

        this.limbs = {}

        if len(body_instances) > 0:

            if len(timeline) != len(body_instances):
                raise Exception('`timeline` and `body_instances` are of unequal length') 

            # initilizing `this.limbs`
            for limb_name in body.limb_nodes:
                this.limbs[limb_name] = []

            # collecting all the angles for each limb throughout all the instances
            for k in range(len(body_instances)):

                body_instance = body_instances[k]

                if body_instance:

                    for limb_name in body_instance.limbs:

                        l = body_instance.limbs[limb_name]
                        this.limbs[limb_name].append((timeline[k], l.angle()))
                    #end
                #end
            #end

            # at this point `this.limbs` is a mapping
            # from `limb_name` to a list of tuples of the form (t, angle_of_`limb_name`_at_moment_t)

            # now converting each array of points into an interpolation function:
            #
            for limb_name in list(this.limbs):

                p = this.limbs[limb_name]
                
                if len(p) > 0:

                    # unpacking
                    x, y = zip(*p)

                    # smoothing
                    y = gaussian_filter1d(y, 0.02*len(y))

                    # creating approximation
                    this.limbs[limb_name] = interpolate.CubicSpline(x, y)

                else:
                    # no data to approximate -> erasing an element from the dictionary
                    del this.limbs[limb_name]

                #endif
            #end
        #endif

    # operator[](limb_name) -> spline
    def __getitem__(this, limb_name):
        return this.limbs[limb_name]


#----------------------------------------------------------------------------------------------------
# meta-helpers:

# create bodies from frames in `frames_dir`;
# returns:
#
#    1. a list of arrays of bodies in each frame
#    2. the corresponding timeline
#    3. total number of dancers detected throughout the frame sequence
#       (as some dancers may appear/leave the scene or just failed to be detected)
#
def create_bodies_from_frames(body_estimation, frames_dir, frame_name_template):

    frames_dir = os.path.normpath(frames_dir) + os.sep

    # reading or creating the timeline:

    try:

        # if file exists -> loading
        timeline = np.load(frames_dir + 'timeline.npz')['timeline']

    except:
        # otherwise creating a default timeline based on the contents of the `frames_dir`
        # (counting the files with the same extension as in `frame_name_template(0)`)

        ext = lambda name: name.split('.')[1].lower()

        template_ext = ext(frame_name_template(0))
        n_frames = len(list(filter(lambda name: ext(name) == template_ext, os.listdir(frames_dir))))

        timeline = range(n_frames)
        
    #----------------------------------------------------------------------

    # list of lists of bodies detected in each frame:
    #
    #   note: body order is arbitrary!
    #
    bodies_unordered = []

    # used for counting the number of individual dancers
    n_dancers = 0

    for i in range(len(timeline)):

        img = cv2.imread(frames_dir + frame_name_template(i))
        candidate, subset = body_estimation(img)
        
        # creating bodies:

        res = []

        # iterating over all the detected bodies
        for n in range(len(subset)):

            b = body(candidate, subset[n])

            # if the created body has at least one limb of interest and root initialized
            if b.limbs and b.root is not None:
                res.append(b)

        bodies_unordered.append(res)
        n_dancers = max(n_dancers, len(res))

    #end

    return bodies_unordered, timeline, n_dancers


# create bodies from video in `video_path`;
# if `dt` is specified only frames at least `dt` seconds apart will be used
# (the exact timing is reflected in the returned timeline)
#
# returns:
#
#    1. a list of arrays of bodies in each frame
#    2. the corresponding timeline
#    3. total number of dancers detected throughout the video
#       (as some dancers may appear/leave the scene or just failed to be detected)
#
def create_bodies_from_video(body_estimation, video_path, dt = None):

    cap = cv2.VideoCapture(video_path)

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    timeline = []

    # list of lists of bodies detected in each frame:
    #
    #   note: body order is arbitrary!
    #
    bodies_unordered = []

    # used for counting the number of individual dancers
    n_dancers = 0

    N = 0

    for i in range(n_frames):

        # time (in seconds) of the current frame
        t = i/fps

        success, img = cap.read()

        if not success:
            continue

        if dt is None or round(t/dt) >= N or i == n_frames - 1:
            # (the first and the last frame are included)
        
            N += 1

            timeline.append(t)

            candidate, subset = body_estimation(img)

            # creating bodies:

            res = []

            # iterating over all the detected bodies
            for n in range(len(subset)):

                b = body(candidate, subset[n])

                # if the created body has at least one limb of interest and root initialized
                if b.limbs and b.root is not None:
                    res.append(b)

            bodies_unordered.append(res)
            n_dancers = max(n_dancers, len(res))

    #end

    cap.release()

    return bodies_unordered, timeline, n_dancers
	

# match body instances in `bodies_unordered` (using the fact that `n_dancers` individual dancers were detected);
# returns an array `bodies` such that:
#
#    bodies[n] - array of all the body instances for n-th dancer
#
def track_bodies(bodies_unordered, timeline, n_dancers):

    # helpers:

    # a dummy NoneType-like class for denoting points of intersection of bodies' trajectories
    class ConflictPoint:
        def __bool__(this):
            return False

    # returns cosine of the angle between 2 vectors
    #
    def cosine(p1, p2):

        return np.dot(p1, p2)/(np.linalg.norm(p1) * np.linalg.norm(p2))

    #-------------------------------------------------------------------------

    if not bodies_unordered:
        print('track_bodies: `bodies_unordered` is empty')
        return []


    bodies = [[None]*len(timeline) for _ in range(n_dancers)]

    # sorting the bodies in the first frame by horizontal position of the root (from left to right):
    bodies_unordered[0].sort(key = lambda b: b.root[0])

    # initial positions and body order are defined by the first frame
    for n in range(n_dancers):

        # however, not all dancers may be present in the first frame
        #
        if n < len(bodies_unordered[0]):
            bodies[n][0] = bodies_unordered[0][n]


    n_frames = len(timeline)

    # skipping the first frame
    for k in range(1, n_frames):

        b_current_frame = bodies_unordered[k]

        # an empty frame -> nothing to update
        if not b_current_frame:
            continue

        # indices of candidates for the next body instances
        candidates = {}

        # choosing the closest body instance for each dancer---------------------------------

        for n in range(n_dancers):

            if bodies[n][k-1]:

                # get the index of the body at frame `k` which is closest to `bodies[n][k-1]`
                #
                idx = min\
                      (
                           range(len(b_current_frame)),
                           key = lambda i: abs(bodies[n][k-1].root[0] - b_current_frame[i].root[0])
                      )

                if idx not in candidates:
                    candidates[idx] = [n]
                else:
                    candidates[idx].append(n)

        # continuing bodies' trajectories----------------------------------------------------

        for idx in candidates:

            if len(candidates[idx]) == 1:

                # only one candidate -> updating
                bodies[candidates[idx][0]][k] = b_current_frame[idx]

            elif len(candidates[idx]) > 1:

                # conflict (intersection of trajectories) -> the current instance cannot be used for any body
                #
                for q in candidates[idx]:
                    bodies[q][k] = ConflictPoint()

                # resolvoing the conflict:

                # if not the last frame...
                if k != n_frames - 1:

                    # ... -> disentangling

                    b_next_frame = bodies_unordered[k+1]

                    if not b_next_frame:
                        continue

                    # yep, during disentanglement new entanglement can appear
                    candidates_with_scores = {}

                    for q in candidates[idx]:

                        x0 = b_current_frame[idx].root[0]
                        p1 = (timeline[k] - timeline[k-1], x0 - bodies[q][k-1].root[0])
                        dt2 = timeline[k+1] - timeline[k]

                        # get the index of the body in frame `k+1`
                        # which gives the best trajectory continuation for `bodies[q][k-1]`
                        #
                        idx = max\
                              (
                                  range(len(b_next_frame)),
                                  key = lambda i: cosine(p1, (dt2, b_next_frame[i].root[0] - x0))
                              )

                        score = cosine(p1, (dt2, b_next_frame[idx].root[0] - x0))

                        if idx not in candidates_with_scores:
                            candidates_with_scores[idx] = [(q, score)]
                        else:
                            candidates_with_scores[idx].append((q, score))

                    # resolving conflicts based on found canditates
                    for idx_s in candidates_with_scores:

                        if len(candidates_with_scores[idx_s]) == 1:

                            q = candidates_with_scores[idx_s][0][0]

                            bodies[q][k+1] = b_next_frame[idx_s]
                            del b_next_frame[idx_s]
                        else:

                            # selecting the instance with maximal score:
                            q, _ = max\
                                   (
                                       candidates_with_scores[idx_s],
                                       key = lambda p: p[1]
                                   )

                            bodies[q][k+1] = b_next_frame[idx_s]
                            del b_next_frame[idx_s]

                            # other instances in `candidates_with_scores[idx_s]` are not updated ->
                            # they remain None -> the corresponding body has disappeared

                else:
                    print('2 bodies seems to be overlapping in the last frame - unable to disentangle')
            #end
        #end

        # checking for unmatched body instances----------------------------------------------

        idx_unmatched = set(range(len(b_current_frame))).difference(candidates)

        if idx_unmatched:

            idx_uninitialized = []

            for n in range(n_dancers):
                
                if bodies[n][k] is None:
                    idx_uninitialized.append(n)
            #end

            if len(idx_uninitialized) != len(idx_unmatched):

                # something strange has happened...
                print('A mismatch between unmatched and uninitialized body instanses in frame {} occured'.format(k))
            else:
                
                for uninitialized, unmatched in zip(idx_uninitialized, idx_unmatched):

                    # updating uninitialized body instances with unmatched ones
                    bodies[uninitialized][k] = b_current_frame[unmatched]
            #end
        #end
    #end

    return bodies

#----------------------------------------------------------------------------------------------------

# creates an array of `dancer` objects from frames in `frames_dir` directory
#
# arguments:
#
#    `body_estimation`     - a body estimation model
#    `frames_dir`          - path to a directory to load frames from
#    `frame_name_template` - a function used for creating sequential frame names
#                            (this function is fed with values from range 0..n_frames-1)
#
# return value:
#
#    an array of `dancer` objects for each dancer detected in the frame sequence
#
def create_dancers_from_frames(body_estimation, frames_dir, \
                               frame_name_template = lambda idx: 'frame' + str(idx) + '.jpg'):

    # creating bodies from frames
    bodies_unordered, timeline, n_dancers = create_bodies_from_frames(body_estimation, \
                                                                      frames_dir, \
                                                                      frame_name_template)

    # get an array of ordered bodies
    bodies = track_bodies(bodies_unordered, timeline, n_dancers)

    # creating dancers:

    dancers = []

    for k in range(len(bodies)):

        d = dancer(bodies[k], timeline)

        # if the created dancer has at least one limb of interest initialized
        if d.limbs:
            dancers.append(d)

    return dancers


# creates an array of `dancer` objects from video in `video_path`;
# if `dt` is specified only frames at least `dt` seconds apart will be used
# (the exact timing is reflected in the returned timeline)
#
# arguments:
#
#    `body_estimation`     - a body estimation model
#    `video_path`          - path to a video to load frames from
#    `dt`                  - the minimal time interval between frames that are to be used for pose extraction
#
# return value:
#
#    an array of `dancer` objects for each dancer detected in the video
#
def create_dancers_from_video(body_estimation, video_path, dt = 0.12):

    # creating bodies from frames
    bodies_unordered, timeline, n_dancers = create_bodies_from_video(body_estimation, video_path, dt)

    # get an array of ordered bodies
    bodies = track_bodies(bodies_unordered, timeline, n_dancers)

    # creating dancers:

    dancers = []

    for k in range(len(bodies)):

        d = dancer(bodies[k], timeline)

        # if the created dancer has at least one limb of interest initialized
        if d.limbs:
            dancers.append(d)

    return dancers

#----------------------------------------------------------------------------------------------------

# save an array of dancers to JSON file
#
def save_dancers(dancers, path):

    os.makedirs(os.path.dirname(path), exist_ok = True)

    json_dancers = [None]*len(dancers)

    for k in range(len(json_dancers)):

        json_dancers[k] = {}

        for limb_name in dancers[k].limbs:
            x = dancers[k].limbs[limb_name].x
            y = dancers[k].limbs[limb_name](x)
            json_dancers[k][limb_name] = list(zip(x, y))

    with open(path, 'w') as f:
        json.dump(json_dancers, f)


# load dancers from JSON file
#
def load_dancers(path):

    with open(path, 'r') as f:
        json_dancers = json.load(f)

    dancers = [None]*len(json_dancers)

    for k in range(len(json_dancers)):

        dancers[k] = dancer()

        for limb_name in json_dancers[k]:

            x, y = zip(*json_dancers[k][limb_name])
            dancers[k].limbs[limb_name] = interpolate.CubicSpline(x, y)

    return dancers