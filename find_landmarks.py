import face_recognition
import os, json

def locate_landmarks(f_image, save_data=False, model='hog'):

    if save_data:
        f_json = os.path.join(save_dest, os.path.basename(f_image)) + '.json'
        if os.path.exists(f_json):
            return False

    # Load the jpg file into a numpy array
    image = face_recognition.load_image_file(f_image)

    faces = face_recognition.face_locations(image,model=model)
    landmarks = face_recognition.face_landmarks(image, face_locations=faces)
    
    if len(landmarks) == 0:
        print "No faces detected for {}".format(f_image)
        return None

    if len(landmarks) > 1:
        print "Warning multiple faces detected for {}".format(f_image)

    if save_data:
        js = json.dumps(landmarks)
        
        with open(f_json,'w') as FOUT:
            FOUT.write(js)

        print "Saved {} faces to {}".format(len(landmarks), f_image)

    return landmarks


print locate_landmarks("source/frames/o3ujLxQP8hE/000584.jpg",False)
exit()

if __name__ == "__main__":
    from tqdm import tqdm
    import glob
    import joblib, sys
    
    start_frame = 0
    end_frame  = 10**20
    name = sys.argv[1]

    
    os.system('mkdir -p {}'.format(save_dest))

    JPG = sorted(glob.glob("source/frames/{}/*".format(name)))
    JPG = JPG[start_frame:end_frame]

    func = joblib.delayed(locate_landmarks)
    ITR = tqdm(JPG)

    with joblib.Parallel(-1, batch_size=2) as MP:
        MP(func(x,save_data=True) for x in ITR)

