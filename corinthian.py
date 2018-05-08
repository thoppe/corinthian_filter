from __future__ import division
import glob, os, json, sys
import numpy as np
import cv2
import joblib
from tqdm import tqdm
from skimage.restoration import inpaint
from scipy.ndimage.filters import convolve
import scipy.ndimage.morphology as morph
from skimage.restoration import inpaint
from shutil import copyfile

from find_landmarks import f_image_to_landmark_file, locate_landmarks

FLAG_DEBUG = [False, True,][0]
FLAG_SHOW = [False, True,][0]

scale_product = 1.10

def read_landmarks(f_json):
    
    assert( os.path.exists(f_json) )

    with open(f_json, 'r') as FIN:
        js = json.loads(FIN.read())

    for face in js:
        for key in face:
            face[key] = np.array(face[key])
            
    return js

def get_mask(pts, height, width):
    mask = np.zeros((height, width))
    cv2.fillConvexPoly(mask, pts, 1)
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

def overlay(X, Y, x_offset, y_offset):
    # https://stackoverflow.com/a/14102014/249341
    alpha_X = Y[:, :, 3] / 255.0
    alpha_Y = 1 - alpha_X

    y1, y2 = y_offset, y_offset + Y.shape[0]
    x1, x2 = x_offset, x_offset + Y.shape[1]
    
    for c in range(0, 3):
        X[y1:y2, x1:x2, c] = (alpha_Y * Y[:, :, c] +
                                alpha_X * X[y1:y2, x1:x2, c])

def get_extent(pts):
    # Return the bounding box (y0,y1,x0,x1)
    return pts[0].min(), pts[0].max(), pts[1].min(), pts[1].max()

def bounding_box_area(pts):
    y0,y1,x0,x1 = get_extent(pts)
    return (y1-y0)*(x1-x0)


def copy_mask(img, mask0, mask1, resize_factor=1.5):

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
    y_offset, x_offset = CM0 - CMX

    # Overlay the image and account of transparent background
    overlay(img, imgX, x_offset, y_offset)

    padding = np.array([
        [y_offset, img.shape[0]-imgX.shape[0]-y_offset],
        [x_offset, img.shape[1]-imgX.shape[1]-x_offset],
    ])

    export_mask = (imgX != TC).max(axis=2)
    export_mask = np.pad(export_mask, padding,  mode='constant')

    return export_mask

def blend_images_over_mask(img0, img1, mask, w=1.0):

    # Get the two subsets, cast as floats
    f0 = img0[mask].astype(float)
    f1 = img1[mask].astype(float)

    f10 = (1.0*f0+w*f1)/(1+w)

    return f10.astype(np.uint8)


def remove_eyes_from_landmarks(L, f_img, f_out=None):
    #load_dest = "source/frames/{}".format(URI)
    #    f_img = f_landmark_to_image_file(f)
    #''.join(
    #    os.path.join(load_dest, os.path.basename(f)).split('.json'))

    assert(os.path.exists(f_img))
    img = cv2.imread(f_img)

    # Convert JPG into PNG with alpha channel
    bc, gc, rc = cv2.split(img)
    ac = np.ones(bc.shape, dtype=bc.dtype) * 0
    img = cv2.merge((bc,gc,rc,ac))
    org_img = img.copy()
    
    height, width, _ = img.shape

    mouth_keys = ['top_lip','bottom_lip']
    masks = [get_mask(L[k], height, width) for k in mouth_keys]
    mouth = np.array(masks).astype(int).sum(axis=0)

    # Fill in the mouth a bit
    cfilter = np.ones((3,3))
    mouth = convolve(mouth, cfilter).astype(np.bool)
    
    # Fill the mouth in if it isn't too open
    mouth = morph.binary_fill_holes(mouth)

    whole_face_pts = np.vstack([L[k] for k in L])
    mouth_pts = np.vstack([L[k] for k in mouth_keys])

    nose_pts = np.vstack([L[k] for k in ['nose_tip','nose_bridge']])
    nose_mask = get_mask(nose_pts, height, width)
    face_mask = get_mask(whole_face_pts, height, width)
    
    mouth_to_face_ratio = np.sqrt(
        bounding_box_area(mouth_pts) /
        bounding_box_area(whole_face_pts) )

    # Clip the ratio so the mouth-eyes don't get too small
    mouth_to_face_ratio = np.clip(mouth_to_face_ratio, 0.5, 1.2)

    scale_factor = scale_product*mouth_to_face_ratio

    left_eye = get_mask(L['left_eye'], height, width)
    right_eye = get_mask(L['right_eye'], height, width)
    
    E0 = copy_mask(img, left_eye, mouth, scale_factor)
    E1 = copy_mask(img, right_eye, mouth, scale_factor)
    
    # Inpaint around the eyes one out and one in from the outer edge
    d = morph.binary_dilation(E0,iterations=1) & (~E0) #& (~nose_mask)
    d = morph.binary_dilation(d,iterations=1)
    img = inpaint.inpaint_biharmonic(img, d, multichannel=True)
    img = np.clip((img*255).astype(np.uint8), 0, 255)

    d = morph.binary_dilation(E1,iterations=1) & (~E1) #& (~nose_mask)
    d = morph.binary_dilation(d,iterations=1)
    img = inpaint.inpaint_biharmonic(img, d, multichannel=True)
    img = np.clip((img*255).astype(np.uint8), 0, 255)

    # Draw back over the nose part a bit
    #img[nose_mask] = org_img[nose_mask]
    #cfilter = np.ones((7,7))
    #nose_mask = convolve(nose_mask, cfilter).astype(np.bool)
    #img[nose_mask] = blend_images_over_mask(img, org_img, nose_mask, 3.0)
    
    if FLAG_DEBUG or FLAG_SHOW:
        show(img)

        if FLAG_DEBUG:
            cv2.imgwrite("debug_image.jpg", img)
        exit()

    elif f_out is not None:
        print "Saved", f_out
        cv2.imwrite(f_out, img)
    
    return img

def remove_eyes(f_json, f_img, f_out=None):

    # If output file exists, skip
    if f_out is not None and os.path.exists(f_out):
        print "Skipping {}".format(f_out)
        return False

    # Create a copy, needed for multiple faces
    if f_out is not None:
        copyfile(f_img, f_out)
    
    for k,faceL in enumerate(read_landmarks(f_json)):
        print "Starting face {}, {}".format(k, f_out)
        
        if f_out is not None:
            remove_eyes_from_landmarks(faceL, f_out, f_out)
        else:
            remove_eyes_from_landmarks(faceL, f_img)
        

def process_image(f_img, f_out=None):
    # Useful for debuging (start directly from an image)
        
    f_json = f_image_to_landmark_file(f_img)
    if not os.path.exists(f_json):
        print "Building landmarks for", f_img
        locate_landmarks(f_img, save_data=True, model='hog')

    remove_eyes(f_json, f_img, f_out)


#process_image("source/frames/KZrMRvvLg58/000203.jpg")
#exit()
'''
#process_image("source/frames/KZrMRvvLg58/002705.jpg")
remove_eyes('data/o3ujLxQP8hE/landmarks/000577.jpg.json')
'''

'''
remove_eyes(
    'data/KZrMRvvLg58/landmarks/000217.jpg.json',
    'source/frames/KZrMRvvLg58/000217.jpg',
    'test.jpg'
    )
'''

if __name__ == "__main__":
    URI = sys.argv[1]    
    F_IMG = sorted(glob.glob("source/frames/{}/*".format(URI)))
    
    save_dest = "data/{}/corinthian/".format(URI)
    os.system('mkdir -p {}'.format(save_dest))

    F_OUT = [os.path.join(save_dest, os.path.basename(f)) for f in F_IMG]

    if FLAG_DEBUG or FLAG_SHOW:
        THREADS = 1
    else:
        THREADS = -1

    with joblib.Parallel(THREADS,batch_size=4) as MP:

        func = joblib.delayed(process_image)
        MP(func(f_img, f_out) for f_img, f_out in tqdm(zip(F_IMG, F_OUT)))
