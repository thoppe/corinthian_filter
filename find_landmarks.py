import face_recognition
import os, json
import dlib
import face_recognition_models

def f_image_to_landmark_file(f_image):
    dname = f_image.split('/')[-2]
    save_dest = os.path.join('data', dname, 'landmarks')
    os.system('mkdir -p {}'.format(save_dest))
    return os.path.join(save_dest, os.path.basename(f_image)) + '.json'

def locate_landmarks(
        f_image,
        save_data=False,
        model='hog',
        upsample_attempts=0
):
    '''
    If upsample attempts > 0, keep upsampling the image until we find at least
    a single face.
    '''

    if save_data:
        f_json = f_image_to_landmark_file(f_image)
        if os.path.exists(f_json):
            return False

    # Load the jpg file into a numpy array
    image = face_recognition.load_image_file(f_image)
    faces = face_recognition.face_locations(image,model=model)

    for n in range(upsample_attempts):
        base_upsample = 1
        if not faces:
            print "No faces found, upsampling", f_image
            faces = face_recognition.face_locations(
                image,base_upsample+n+1,model=model)

    landmarks = face_recognition.face_landmarks(image, face_locations=faces)

    for face in landmarks:
        for key in face:
            face[key] = [(int(x), int(y)) for (x,y) in face[key]]
    
    if len(landmarks) == 0:
        landmarks = {}

    if save_data:
        js = json.dumps(landmarks)
        
        with open(f_json,'w') as FOUT:
            FOUT.write(js)

        print "Saved {} faces to {}".format(len(landmarks), f_json)

    return landmarks


#print locate_landmarks("source/frames/o3ujLxQP8hE/000584.jpg",False)
#exit()

if __name__ == "__main__":
    from tqdm import tqdm
    import glob
    import joblib, sys
    
    start_frame = 0
    end_frame  = 10**20
    name = sys.argv[1]
   


    JPG = sorted(glob.glob("source/frames/{}/*".format(name)))
    JPG = JPG[start_frame:end_frame]

    func = joblib.delayed(locate_landmarks)
    ITR = tqdm(JPG)

    with joblib.Parallel(-1, batch_size=2) as MP:
        MP(func(x,save_data=True) for x in ITR)

