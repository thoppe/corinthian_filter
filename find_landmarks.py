import face_recognition
import glob, os, json

save_dest = "data/landmarks"
os.system('mkdir -p {}'.format(save_dest))


def locate_landmarks(f_image):

    f_json = os.path.join(save_dest, os.path.basename(f_image)) + '.json'
    if os.path.exists(f_json):
        return False

    # Load the jpg file into a numpy array
    image = face_recognition.load_image_file(f_image)
    landmarks = face_recognition.face_landmarks(image)
    
    if len(landmarks) == 0:
        print "No faces detected for {}".format(f_image)
        return None

    if len(landmarks) > 1:
        print "Multiple faces detected for {}".format(f_image)
        return None
    
    js = json.dumps(landmarks[0])
    with open(f_json,'w') as FOUT:
        FOUT.write(js)

    print "Completed", f_image


JPG = sorted(glob.glob("source_movies/images/*"))[:]

import joblib
from tqdm import tqdm
func = joblib.delayed(locate_landmarks)

ITR = tqdm(JPG)

with joblib.Parallel(-1, batch_size=2) as MP:
    MP(func(x) for x in ITR)

