import os, json
import cv2
import numpy as np

# https://github.com/1adrianb/face-alignment

import face_alignment
from face_alignment.utils import *

fa = face_alignment.FaceAlignment(
    face_alignment.LandmarksType._2D,
    enable_cuda=True,
    enable_cudnn=True,
    use_cnn_face_detector=False,
    #flip_input=True,
)


def f_image_to_landmark_file(f_image):
    dname = f_image.split('/')[-2]
    save_dest = os.path.join('data', dname, 'landmarks')
    os.system('mkdir -p {}'.format(save_dest))
    return os.path.join(save_dest, os.path.basename(f_image)) + '.json'

def identify_landmarks(points):
    """
    Given a set of landmark points, returns a dict of face feature locations 
    (eyes, nose, etc) for these points
    :return: A list of dicts of face feature locations (eyes, nose, etc)
    """

    # For a definition of each point index,
    # see https://cdn-images-1.medium.com/max/1600/1*AbEg31EgkbXSQehuNJBlWg.png
    return {
        "all_points":points[:],
        "chin": points[0:17],
        "left_eyebrow": points[17:22],
        "right_eyebrow": points[22:27],
        "nose_bridge": points[27:31],
        "nose_tip": points[31:36],
        "left_eye": points[36:42],
        "right_eye": points[42:48],
        "top_lip": (points[48:55] + [points[64]] + [points[63]] +
                    [points[62]] + [points[61]] + [points[60]]),
        "bottom_lip": (points[54:60] + [points[48]] + [points[60]] +
                       [points[67]] + [points[66]] + [points[65]] +
                       [points[64]]),
    }


def locate_landmarks(img):
    
    faces = fa.get_landmarks(img)
    landmarks = []

    if faces:
        for face in faces:       
            face = face.astype(np.uint8)
            landmarks.append(identify_landmarks(face))

    return landmarks

def landmarks_from_image(
        f_img,
        save_data=False,
):
    
    if save_data:
        f_json = f_image_to_landmark_file(f_img)
        if os.path.exists(f_json):
            return False

    # Load the image file into a numpy array
    img = cv2.imread(f_img)
    landmarks = locate_landmarks(img)

    if save_data:

        save_copy = []
        for face in landmarks:
            face2 = {}
            for key in face:
                face2[key] = face[key].tolist()
            save_copy.append(face2)
            
        js = json.dumps(save_copy)
        
        with open(f_json,'w') as FOUT:
            FOUT.write(js)

        print "Saved {} faces to {}".format(len(landmarks), f_json)

    return landmarks



def show(img):
    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys, glob
    from tqdm import tqdm
    import joblib

    URI = sys.argv[1]
    F_IMG = sorted(glob.glob("source/frames/{}/*".format(URI)))
    for f_img in tqdm(F_IMG):
        landmarks_from_image(f_img, save_data=True)

    exit()
    '''
    f = sys.argv[1]

    img = cv2.imread(f)
        
    faces = fa.get_landmarks(f)
    pts = faces[0].astype(np.int)
    print pts.max(), pts.min()

    h,w = img.shape[:2]
    idx = (pts[:,0]<w) & (pts[:,1]<h)   
    pts = pts[idx]
    img[pts[:,1], pts[:,0]] = [255, 255, 255]
    
    show(img)
    cv2.imwrite("demo.jpg", img)
    '''
    

    # Speed tests, if slow, check dlib has CNN support
    img = cv2.imread(f)

    import itertools
    ITR = tqdm(itertools.cycle([img,]))

    for img in ITR:
        locate_landmarks(img)

