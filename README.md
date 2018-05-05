# Corinthian Filter

**Pipeline:**

Download a video, convert it into frames, find landmarks

    youtube-dl https://www.youtube.com/watch?v=cVW6jBbD5Q8`
    python extract_frames.py cVW6jBbD5Q8
    python find_landmarks.py cVW6jBbD5Q8

Begin the nightmare

    python corinthian.py cVW6jBbD5Q8 2.0

![Obama Corinthinan Filter](demo_image_obama.jpg)

Audio notes, make 3 tracks, copy voice track to other two, shift pitch by 60% and other by 80%. Offset tracks +/- a few milli seconds. Turn left/right channels to max respectively. Add paulstrech at 0.5. Add a fourth track with the original on paulstrech with no strech but 1.5 delay.