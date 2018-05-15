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
from scipy.ndimage.filters import convolve
import scipy.ndimage.morphology as morph
from shutil import copyfile
from docopt import docopt


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
        img[mask0] = [250,250,250,0]
        img[mask1] = [0,150,150,100]
        img[CM0[0], CM0[1]] = [250,0,50,0]

    # Extract a fresh copy of the source mask
    pts1 = np.array(np.where(mask1))
    y0,y1,x0,x1 = get_extent(pts1)
    imgX = img[y0:y1,x0:x1].copy()

    # On the mask, apply a transparent filter
    TC = np.array([0,0,0,255])
    imgX[~mask1[y0:y1,x0:x1]] = TC

    if FLAG_DEBUG:
        imgX[mask1[y0:y1,x0:x1]] = [50,50,50,50]

    # Resize the target image
    imgX = cv2.resize(imgX, None, fx=resize_factor, fy=resize_factor)
    ptsX = np.array(np.where(imgX!=[0,0,0,0]))[:2]
    CMX = ptsX.mean(axis=1).astype(int)

    # Adjust so that the center of masses line up
    offset = np.clip(CM0 - CMX, 0, 10**20)

    # If the values go off the screen clip imgX
    sy,sx,sc = imgX.shape
    clip_val = np.clip((imgX.shape[:2] + offset) - img.shape[:2], 0, 10**10)
    imgX = imgX[:sy-clip_val[0], :sx-clip_val[1], :]

    # Overlay the image and account of transparent background
    overlay(img, imgX, offset)

    padding = np.array([
        [offset[0], img.shape[0]-imgX.shape[0]-offset[0]],
        [offset[1], img.shape[1]-imgX.shape[1]-offset[1]],
    ])

    export_mask = (imgX != TC).max(axis=2)    
    export_mask = np.pad(export_mask, padding,  mode='constant')

    return export_mask


def inpaint_mask(img, mask, method=cv2.INPAINT_TELEA):
    # Inpaint with cv2 (fast!), need to remove alpha channel because reasons
    
    tmp_img = img[:,:,:3]
    tmp_img = cv2.inpaint(tmp_img,
        mask.astype(np.uint8),3,method)

    img[:,:,:3] = tmp_img
    return img

def blend_images_over_mask(img0, img1, mask, w=1.0):

    # Get the two subsets, cast as floats
    f0 = img0[mask].astype(float)
    f1 = img1[mask].astype(float)

    f10 = (1.0*f0+w*f1)/(1+w)

    return f10.astype(np.uint8)


def remove_eyes_from_landmarks(L, f_img):

    # Cast all the landmarks to numpy arrays
    for key in L:
        L[key] = np.array(L[key])
    
    assert(os.path.exists(f_img))
    img = cv2.imread(f_img)

    # Convert JPG into PNG with alpha channel
    bc, gc, rc = cv2.split(img)
    ac = np.ones(bc.shape, dtype=bc.dtype) * 0
    img = cv2.merge((bc,gc,rc,ac))
    org_img = img.copy()
        
    height, width, _ = img.shape

    left_eye = get_mask(L['left_eye'], height, width)
    right_eye = get_mask(L['right_eye'], height, width)

    mouth_keys = ['top_lip','bottom_lip']
    masks = [get_mask(L[k], height, width) for k in mouth_keys]
    mouth = np.array(masks).astype(int).sum(axis=0)

    # Inpaint the whole eye area, dialated a few times
    if not FLAG_DEBUG:
        eye_mask = morph.binary_dilation(left_eye|right_eye,iterations=3)
        img = inpaint_mask(img, eye_mask)

    # Fill in the mouth a bit
    cfilter = np.ones((3,3))
    mouth = convolve(mouth, cfilter).astype(np.bool)
    
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
        
    E0 = copy_mask(img, left_eye, mouth, scale_factor)
    E1 = copy_mask(img, right_eye, mouth, scale_factor)

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
    
    for k,faceL in enumerate(L):
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
