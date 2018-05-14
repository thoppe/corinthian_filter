import glob, os, json, sys
import numpy as np
import pandas as pd
import cv2
import joblib
from tqdm import tqdm

def compute_frame_delta(URI):
    F_IMG = sorted(glob.glob("source/frames/{}/*".format(URI)))[:]

    data = []
    for k,f in enumerate(tqdm(F_IMG)):
        img = cv2.imread(f).astype(float)
        
        if k == 0:
            prior_img = img
            continue

        item = {
            "frame":k,
            "mean_abs":np.abs(prior_img-img).mean()
        }
        data.append(item)
        prior_img = img
        
    df = pd.DataFrame(data).set_index('frame')
    return df


if __name__ == "__main__":

    URI = sys.argv[1]
    save_dest = os.path.join('data', URI)

    f_frame_delta = os.path.join(save_dest, 'frame_delta.csv')
    if not os.path.exists(f_frame_delta):
        df = compute_frame_delta(URI)
        df.to_csv(f_frame_delta)

    df = pd.read_csv(f_frame_delta)
    print df
