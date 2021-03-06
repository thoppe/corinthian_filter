import glob, os, json, sys, shutil
import numpy as np
import pandas as pd
import cv2
import joblib
from tqdm import tqdm
from sklearn.decomposition import PCA
from scipy.signal import savgol_filter
_abs_mean_threshold = 10.0


def compute_frame_delta(URI):
    F_IMG = sorted(glob.glob("source/frames/{}/*".format(URI)))[:]

    data = []
    for f in tqdm(F_IMG):
        img = cv2.imread(f).astype(float)
        n = int(os.path.basename(f).split('.')[0])

        if n == 1:
            prior_img = img
            continue

        item = {
            "frame":n-1,
            "mean_abs":np.abs(prior_img-img).mean()
        }
        data.append(item)
        prior_img = img
        
    df = pd.DataFrame(data).set_index('frame')
    return df

def count_faces_per_image(URI):
    F_JSON = sorted(glob.glob("data/{}/landmarks/*".format(URI)))[:]

    data = []
    for f in tqdm(F_JSON):
        with open(f) as FIN:
            js = json.loads(FIN.read())
            n = int(os.path.basename(f).split('.')[0])

            item = {
                "frame":n,
                "n_faces":len(js),
            }
            data.append(item)
            
    df = pd.DataFrame(data).set_index('frame')
    return df


def identify_landmarks(points):
    """
    Given a set of landmark points, returns a dict of face feature locations 
    (eyes, nose, etc) for these points
    :return: A list of dicts of face feature locations (eyes, nose, etc)
    """

    # For a definition of each point index,
    # see https://cdn-images-1.medium.com/max/1600/1*AbEg31EgkbXSQehuNJBlWg.png
    data = {
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

    for key in data:
        data[key] = data[key].tolist()
        
    return data
    

def low_rank_transform(pts, nc=None):

    pts = np.array(pts).astype(float)
    #mu_pts = pts.mean(axis=0)    
    #pts = pts - mu_pts

    #norm_pts = np.linalg.norm(pts,axis=0)
    #norm_pts = np.clip(norm_pts, 0.01, 10**10)
    #pts /= norm_pts

    if nc is None:
        nc = len(pts)//10
        
    clf = PCA(n_components=nc, whiten=True)
    
    # Low rank reconstruction
    tpts = pts.reshape([-1, 68*2])
    tpts = clf.inverse_transform(clf.fit_transform(tpts))
    tpts = savgol_filter(tpts, 3, 2)
        
    tpts = tpts.reshape([-1, 68, 2])
    #tpts *= norm_pts
    #tpts += mu_pts
    
    tpts = tpts.astype(int)

    return tpts, clf



class face_residuals(object):
    def __init__(self):
        f_face_samples = 'data/sample_face_landmarks.npy'
        assert(os.path.exists(f_face_samples))
        faces = np.load(f_face_samples)

        _, self.clf = low_rank_transform(faces, nc=15)
    def __call__(self, pts):
        pts = np.array(pts)
        tpts = pts.reshape([1, 68*2])
        tpts = self.clf.inverse_transform(self.clf.transform(tpts))
        tpts = tpts.reshape([68,2])
        return np.abs(tpts-pts).mean()

    #F = face_residuals()


if __name__ == "__main__":

    URI = sys.argv[1]
    save_dest = os.path.join('data', URI)

    f_frame_delta = os.path.join(save_dest, 'frame_delta.csv')
    if not os.path.exists(f_frame_delta):
        df = compute_frame_delta(URI)
        df.to_csv(f_frame_delta)

    df = pd.read_csv(f_frame_delta).set_index('frame')[:]

    if "n_faces" not in df.columns:
        df['n_faces'] = count_faces_per_image(URI)
        df.to_csv(f_frame_delta)


    os.system('mkdir -p data/{}/stable_landmarks'.format(URI))
    pts = []; f_jsons = []

    # Useful for constructing a master face
    all_valid_points = []

    for i,row in tqdm(df.iterrows()):
        f = os.path.join("data/{}/landmarks/{:06d}.jpg.json".format(URI,i))
        f2 = os.path.join("data/{}/stable_landmarks/{:06d}.jpg.json".format(URI,i))
        shutil.copy(f, f2)
        
        if (row.n_faces != 1) | (row.mean_abs > _abs_mean_threshold):
            
            if len(pts) > 15:
                pts,_ = low_rank_transform(pts)
                for pt, (fx1, fx2) in zip(pts, f_jsons):
                    js = [identify_landmarks(pt)]
                    with open(fx2,'w') as FOUT:
                        FOUT.write(json.dumps(js))
                        
            pts = []; f_jsons = []

        else:
            with open(f) as FIN:
                js = json.loads(FIN.read())
                p = js[0]['all_points']
                pts.append(p)
                all_valid_points.append(p)
                f_jsons.append((f,f2))

    # # Dump out the last set (can refactor here)
    if len(pts) > 15:
        pts,_ = low_rank_transform(pts)
        for pt, (fx1, fx2) in zip(pts, f_jsons):
            js = [identify_landmarks(pt)]
            with open(fx2,'w') as FOUT:
                FOUT.write(json.dumps(js))

    # Add in the final frame
    i = df.index.max() + 1
    f = os.path.join("data/{}/landmarks/{:06d}.jpg.json".format(URI,i))
    f2 = os.path.join("data/{}/stable_landmarks/{:06d}.jpg.json".format(URI,i))
    shutil.copy(f, f2)

    # Uncomment if you want to save a master face
    #all_valid_points = np.array(all_valid_points)
    #np.save("data/sample_face_landmarks.npy", all_valid_points)
