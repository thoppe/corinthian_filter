from __future__ import division
import glob, os, json, sys
import numpy as np
import cv2
import joblib
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.restoration import inpaint
from scipy.ndimage.filters import convolve
import scipy.ndimage.morphology as morph
from skimage.restoration import inpaint

FLAG_DEBUG = [False, True,][0]
FLAG_SHOW = [False, True,][0]

URI = sys.argv[1]
scale_product = 1.00

def read_landmarks(f_json):
    assert( os.path.exists(f_json) )

    with open(f_json, 'r') as FIN:
        js = json.loads(FIN.read())

    for key in js:
        js[key] = np.array(js[key])
        
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


def remove_eyes(f, f_out=None):
    L = read_landmarks(f)

    load_dest = "source/frames/{}".format(URI)
    f_img = ''.join(
        os.path.join(load_dest, os.path.basename(f)).split('.json'))

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
    mouth_to_face_ratio = (bounding_box_area(mouth_pts) /
                           bounding_box_area(whole_face_pts))

    scale_factor = scale_product*mouth_to_face_ratio

    left_eye = get_mask(L['left_eye'], height, width)
    right_eye = get_mask(L['right_eye'], height, width)
    
    E0 = copy_mask(img, left_eye, mouth, scale_factor)
    E1 = copy_mask(img, right_eye, mouth, scale_factor)

    
    # Inpaint around the eyes one out and one in from the outer edge
    d = morph.binary_dilation(E0,iterations=1) & (~E0)
    d = morph.binary_dilation(d,iterations=1)
    img = inpaint.inpaint_biharmonic(img, d, multichannel=True)
    img = np.clip((img*255).astype(np.uint8), 0, 255)

    d = morph.binary_dilation(E1,iterations=1) & (~E1)
    d = morph.binary_dilation(d,iterations=1)
    img = inpaint.inpaint_biharmonic(img, d, multichannel=True)
    img = np.clip((img*255).astype(np.uint8), 0, 255)

    if f_out is not None:
        print "Saved", f_out
        cv2.imwrite(f_out, img)

    if FLAG_DEBUG or FLAG_SHOW:
        show(img)
        exit()
    
    return img


landmark_files = sorted(glob.glob("data/{}/landmarks/*".format(URI)))
save_dest = "data/{}/corinthian/".format(URI)
os.system('mkdir -p {}'.format(save_dest))

##remove_eyes('source_movies/images/000205.jpg')
#remove_eyes('data/cVW6jBbD5Q8/landmarks/000201.jpg.json')
#exit()

ITR = landmark_files

F_OUT = [''.join(os.path.join(save_dest, os.path.basename(f)).split('.json'))
         for f in ITR]

THREADS = -1

if FLAG_DEBUG or FLAG_SHOW:
    THREADS = 1

with joblib.Parallel(THREADS,batch_size=2) as MP:
    func = joblib.delayed(remove_eyes)
    MP(func(f,f_out) for f,f_out in tqdm(zip(ITR, F_OUT)))
