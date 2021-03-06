import os, json
import cv2
import numpy as np
from skimage import io
from frame_stabilization import face_residuals

# https://github.com/1adrianb/face-alignment
import face_alignment

### Monkey patch face detector so we can potentially upsample
class face_alignment_upsample(face_alignment.FaceAlignment):
    def detect_faces(self, image, n_upsample=1):
        return self.face_detector(image, n_upsample)

print "Loading face alignment"
fa = face_alignment_upsample(
    face_alignment.LandmarksType._2D,
    enable_cuda=True,
    enable_cudnn=True,
    use_cnn_face_detector=True,
    flip_input=False,
)

'''
fa3D = face_alignment.FaceAlignment(
    face_alignment.LandmarksType._3D,
    enable_cuda=True,
    enable_cudnn=True,
    flip_input=False
)
'''


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
        "top_lip": points[[48,49,50,51,52,53,54,64,63,62,61,60]],
        "bottom_lip": points[[54,55,56,57,58,59,48,60,67,66,65,64]],
    }


def locate_landmarks(img):
    
    faces = fa.get_landmarks(img, all_faces=True)
    landmarks = []

    if faces:
        for face in faces:       
            face = face.astype(int)
            landmarks.append(identify_landmarks(face))

    return landmarks

def landmarks_from_image(f_img):
    
    # Load the image file into a numpy array
    img = io.imread(f_img)
    return locate_landmarks(img)


def serialize_landmarks(f_json, L):
    
    save_copy = []
    for face in L:
        face2 = {}
        for key in face:
            face2[key] = face[key].tolist()
        save_copy.append(face2)

    js = json.dumps(save_copy)

    with open(f_json,'w') as FOUT:
        FOUT.write(js)

    #print "Saved {} faces to {}".format(len(L), f_json)



def show(img):
    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys, glob
    from tqdm import tqdm

    # Single image test
    f = sys.argv[1]
    
    faces = landmarks_from_image(f)
    print "Found {} faces".format(len(faces))

    img = cv2.imread(f)
    h,w = img.shape[:2]

    F_EVAL = face_residuals()
    
    for face in faces:
        pts = face['all_points'].astype(np.int)

        idx = (pts[:,0]<w) & (pts[:,1]<h)   
        pts = pts[idx]

        identify_landmarks(pts)
        is_valid = F_EVAL(pts) < 5
        
        if is_valid:
            img[pts[:,1], pts[:,0]] = [255, 255, 255]
        else:
            img[pts[:,1], pts[:,0]] = [255, 0, 0]
            
    
    show(img)
    cv2.imwrite("demo.jpg", img)
    
    exit()

    # Speed tests, if slow, check dlib has CNN support
    img = cv2.imread(f)

    import itertools
    ITR = tqdm(itertools.cycle([img,]))

    for img in ITR:
        locate_landmarks(img)

    # URI test
    #URI = sys.argv[1]
    #F_IMG = sorted(glob.glob("source/frames/{}/*".format(URI)))
    #for f_img in tqdm(F_IMG):
    #    landmarks_from_image(f_img, save_data=True)
