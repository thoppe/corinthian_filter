# Corinthian Filter

**Pipeline:**

Download a video, convert it into frames, find landmarks

    youtube-dl https://www.youtube.com/watch?v=cVW6jBbD5Q8`
    python extract_frames.py cVW6jBbD5Q8
    python find_landmarks.py cVW6jBbD5Q8

Begin the nightmare

    python corinthian.py cVW6jBbD5Q8

![Obama Corinthinan Filter](demo_image_obama.jpg)