"""Corinthian Filter

Usage:
  corinthian.py <location> --URI [--scale=<f>] [--stable]
  corinthian.py <f_image> [--debug] [--scale=<f>]

Options:
  -h --help     Show this screen.
  --version     Show version.
  -d --debug       Debug mode
  -s --scale=<f>   Amount to scale mouthes [default: 0.60]
  --stable         Apply stabilization to video
"""

from __future__ import division
import glob, os, json, sys, tempfile
import numpy as np
import cv2
import joblib
from tqdm import tqdm
#from scipy.ndimage.filters import convolve
import scipy.ndimage.morphology as morph
from shutil import copyfile
from docopt import docopt
import imutils
from sklearn.cluster import KMeans
from frame_stabilization import face_residuals

FACE_RESIDUALS = face_residuals()
_max_face_residual = 5.0

def f_image_to_landmark_file(f_image, save_dest):
    return os.path.join(save_dest, os.path.basename(f_image)) + '.json'

def get_mask(pts, height, width):

    # Keep the points within the frame
    idx = (pts[:,0]<width) & (pts[:,1]<height)
    pts = pts[idx]

    hull = cv2.convexHull(pts)
    mask = np.zeros((height, width))
    cv2.fillConvexPoly(mask, hull, 1)
    mask = mask.astype(np.bool)
    
    return mask

def show_mask(mask, img):
    out = np.zeros_like(img)
    out[mask] = img[mask]
    show(img)

def show(img):
    if img.dtype==bool:
        img2 = np.zeros((img.shape[0], img.shape[1], 3))
        img2[img] = [255,255,255]
        img = img2.astype(np.uint8)

    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def overlay(X, Y, offset):
    
    # https://stackoverflow.com/a/14102014/249341
    alpha_X = Y[:, :, 3] / 255.0
    alpha_Y = 1 - alpha_X

    y1, y2 = offset[0], offset[0] + Y.shape[0]
    x1, x2 = offset[1], offset[1] + Y.shape[1]
    
    for c in range(0, 3):
        X[y1:y2, x1:x2, c] = (alpha_Y * Y[:, :, c] +
                                alpha_X * X[y1:y2, x1:x2, c])
    

def get_extent(pts):
    # Return the bounding box (y0,y1,x0,x1)
    y0,y1 = pts[0].min(), pts[0].max()
    x0,x1 = pts[1].min(), pts[1].max()
    
    return y0,y1,x0,x1

def bounding_box_area(pts):
    y0,y1,x0,x1 = get_extent(pts)
    return (y1-y0)*(x1-x0)

def copy_mask(img, mask0, mask1, resize_factor=1.5):
    
    if not mask0.sum():
        print "Target mask empty, skipping copy"
        return np.zeros([img.shape[0], img.shape[1]],dtype=bool)

    # Determine the dimensions of the target
    pts0 = np.array(np.where(mask0))
    CM0 = pts0.mean(axis=1).astype(int)

    if FLAG_DEBUG:
        img[mask0] = [250,250,250]
        img[mask1] = [0,150,150]
        img[CM0[0], CM0[1]] = [250,0,50]

    # Extract a fresh copy of the source mask
    pts1 = np.array(np.where(mask1))
    y0,y1,x0,x1 = get_extent(pts1)
    imgX = img[y0:y1,x0:x1].copy()

    if FLAG_DEBUG:
        imgX[mask1[y0:y1,x0:x1]] = [50,50,50]

    maskX = np.zeros_like(imgX)
    maskX[mask1[y0:y1,x0:x1]] = [255,255,255]
    
    # Resize the target image
    imgX = cv2.resize(imgX, None, fx=resize_factor, fy=resize_factor,
                      interpolation=cv2.INTER_AREA)
    maskX = cv2.resize(maskX, None, fx=resize_factor, fy=resize_factor,
                       interpolation=cv2.INTER_AREA)

    # Try rotating?
    #imgX = imutils.rotate_bound(imgX, -20)
    #idx = np.where((imgX==[0,0,0]).all(axis=2))
    #imgX[idx]= avg_color
    
    mask = 255 * np.ones(imgX.shape, imgX.dtype)    
    offset = tuple(CM0)[::-1]

    img = cv2.seamlessClone(imgX, img, mask, offset, cv2.MIXED_CLONE)
    img = cv2.seamlessClone(imgX, img, maskX, offset, cv2.NORMAL_CLONE)

    return img


def inpaint_mask(img, mask, method=cv2.INPAINT_TELEA):
    # Inpaint with cv2 (fast!)
    
    return cv2.inpaint(img,
        mask.astype(np.uint8), 15, method)

def compute_pt_slope(pts):
    idx0 = np.argmin(pts[:,1])
    idx1 = np.argmax(pts[:,1])

    r0 = pts[idx0].astype(float)
    r1 = pts[idx1].astype(float)
    r0 /= np.linalg.norm(r0)
    r1 /= np.linalg.norm(r1)

    return np.arccos(np.dot(r0, r1))
    


def remove_eyes_from_landmarks(L, f_img):

    # Cast all the landmarks to numpy arrays
    for key in L:
        L[key] = np.array(L[key])
    
    assert(os.path.exists(f_img))
    img = cv2.imread(f_img)
    org_img = img.copy()

        
    height, width, _ = img.shape

    left_eye = get_mask(L['left_eye'], height, width)
    right_eye = get_mask(L['right_eye'], height, width)

    mouth_keys = ['top_lip','bottom_lip']
    masks = [get_mask(L[k], height, width) for k in mouth_keys]
    mouth = np.array(masks).astype(int).sum(axis=0)

    # Inpaint the whole eye area, dialated a few times
    avg_eye_area = (left_eye.sum() + right_eye.sum())/2
    itr = int(0.25*np.sqrt(avg_eye_area))

    eye_mask = morph.binary_dilation(left_eye|right_eye,iterations=1)

    # Find the whites of the eyes! Do in HSV
    
    if not FLAG_DEBUG:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        blur_img = cv2.GaussianBlur(hsv, (5,5), 10)
        eyes = blur_img[eye_mask]

        n_clusters = 4
        clf = KMeans(n_clusters=n_clusters)
        clf.fit(eyes)

        centroids = np.array([
            np.mean(eyes[clf.labels_==i],axis=0) for i in range(n_clusters)])

        eye_white_idx_label = np.argmin(centroids[:,1])

        # With the whites of the eyes, blur a bit more!
        expanded_eye_mask = morph.binary_dilation(eye_mask,iterations=itr)
        labels = clf.predict(blur_img[expanded_eye_mask])

        idx_x, idx_y = np.where(expanded_eye_mask)
        eye_mask[idx_x[labels==i], idx_y[labels==i]] = True
        eye_mask = morph.binary_dilation(eye_mask,iterations=4)
        img = inpaint_mask(img, eye_mask, )

        
    # Fill in the mouth a bit
    mouth  = morph.binary_dilation(mouth, iterations=itr)
    
    # Fill the mouth in if it isn't too open
    mouth = morph.binary_fill_holes(mouth)
    mouth_pts = np.vstack([L[k] for k in mouth_keys])

    nose_pts = np.vstack([L[k] for k in ['nose_tip','nose_bridge']])
    nose_mask = get_mask(nose_pts, height, width)

    face_pts = L['all_points']
    face_mask = get_mask(face_pts, height, width)
    
    #mouth_to_face_ratio = np.sqrt(
    #    bounding_box_area(mouth_pts) /
    #    bounding_box_area(face_pts) )
    
    # Clip the ratio so the mouth-eyes don't get too small
    #mouth_to_face_ratio = np.clip(mouth_to_face_ratio, 0.5, 1.2)    

    for n in range(1):
        img = copy_mask(img, left_eye, mouth, scale_factor)
        img = copy_mask(img, right_eye, mouth, scale_factor)

    

    # Drop points
    img[L['all_points'][:,1], L['all_points'][:,0]] = [0,255,0]
    for key in ['top_lip', 'bottom_lip', 'right_eye', 'left_eye']:
            X = L[key]
            img[X[:,1], X[:,0]] = [255,255,255]

            print key, compute_pt_slope(L[key])

    show(img)
    exit()

    # Draw back over the nose part a bit
    #img[nose_mask] = org_img[nose_mask]
    #cfilter = np.ones((7,7))
    #nose_mask = convolve(nose_mask, cfilter).astype(np.bool)
    #img[nose_mask] = blend_images_over_mask(img, org_img, nose_mask, 3.0)

    d0 = morph.binary_dilation(E0,iterations=1)
    d1 = morph.binary_dilation(E1,iterations=1)
    outline = (d0|d1) & (~(E0|E1|nose_mask))
    outline = morph.binary_dilation(outline,iterations=1)

    if not FLAG_DEBUG:
        # Inpaint around the eyes one out and one in from the outer edge
        img = inpaint_mask(img, outline)
    
    elif FLAG_DEBUG:
        # Show the outline mask
        img[outline] = [90,150,150,100]

        img[L['all_points'][:,1], L['all_points'][:,0]] = [255,255,255,0]

        for key in ['top_lip', 'bottom_lip', 'right_eye', 'left_eye']:
            X = L[key]
            img[X[:,1], X[:,0]] = [255,0,0,0]            

    return img


def remove_eyes(L, f_img, f_out=None):

    # If output file exists, skip
    if f_out is not None and os.path.exists(f_out):
        return False

    TMP = tempfile.NamedTemporaryFile(suffix='.jpg')
    f_tmp = TMP.name

    # Create a copy, needed for multiple faces
    copyfile(f_img, f_tmp)
    
    for faceL in L:
        if FACE_RESIDUALS(faceL['all_points']) > _max_face_residual:
            continue
        
        img = remove_eyes_from_landmarks(faceL, f_tmp)
        cv2.imwrite(f_tmp, img)
        
    if L and (FLAG_DEBUG or FLAG_VIEW):
        show(img)
        exit()
        
    if f_out is not None:
        print "Saved", f_out
        copyfile(f_tmp, f_out)

    TMP.close()                

        
    
def process_image(f_img, f_json, f_out=None):

    if not os.path.exists(f_json):
        print "Building landmarks for", f_img

        import find_landmarks_FAN as FAN
        L = FAN.landmarks_from_image(f_img)
        FAN.serialize_landmarks(f_json, L)

    assert( os.path.exists(f_json) )

    with open(f_json, 'r') as FIN:
        L = json.loads(FIN.read())

    remove_eyes(L, f_img, f_out)


if __name__ == "__main__":
    
    args = docopt(__doc__, version='corinthian 0.1')

    FLAG_DEBUG = args["--debug"]
    FLAG_VIEW = False
    scale_factor = float(args["--scale"])

    # If we are parsing a single image
    if not args['--URI']:
        FLAG_VIEW = True
        f_img = args["<f_image>"]
        f_json = f_img+'_landmarks.json'
        process_image(f_img, f_json)

    # If we are parsing a set of images
    if args['--URI']:

        apply_stable = args['--stable']

        loc = args['<location>']
        F_IMG = sorted(glob.glob("source/frames/{}/*".format(loc)))
        
        # Preprocess landmarks first (can't be parallel)
        has_imported_FAN = False
        print "Computing Landmarks for {}".format(loc)
        json_save_dest = os.path.join('data', loc, 'landmarks')
        os.system('mkdir -p {}'.format(json_save_dest))
        
        for f_img in tqdm(F_IMG):
            f_json = f_image_to_landmark_file(f_img, json_save_dest)

            if not os.path.exists(f_json):

                if not has_imported_FAN:
                    import find_landmarks_FAN as FAN
                    has_imported_FAN = True

                L = FAN.landmarks_from_image(f_img)
                FAN.serialize_landmarks(f_json, L)

        # Call frame_stabilization.py if needed
        if apply_stable:
            print "Starting frame stabilization"
            os.system('python frame_stabilization.py {}'.format(loc))
            json_save_dest = os.path.join('data', loc, 'stable_landmarks')

        # Compute faces in parallel
        img_save_dest = "data/{}/corinthian/".format(loc)
        os.system('mkdir -p {}'.format(img_save_dest))
        F_OUT  = [os.path.join(img_save_dest, os.path.basename(f))
                  for f in F_IMG]
        
        F_JSON = [f_image_to_landmark_file(f, json_save_dest) for f in F_IMG]
        THREADS = -1

        with joblib.Parallel(THREADS,batch_size=10) as MP:
            func = joblib.delayed(process_image)
            MP(
                func(f_img, f_json, f_out)
                for f_img, f_json, f_out in
                tqdm(zip(F_IMG, F_JSON, F_OUT))
            )
