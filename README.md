# Corinthian Filter

**Pipeline:**

Download a video, convert it into frames, find landmarks

    youtube-dl https://www.youtube.com/watch?v=cVW6jBbD5Q8`
    python extract_frames.py cVW6jBbD5Q8
    python find_landmarks.py cVW6jBbD5Q8

Begin the nightmare

    python corinthian.py cVW6jBbD5Q8

![Obama Corinthinan Filter](docs/images/demo_image_obama.jpg)
![Gaimen Corinthinan Filter](docs/images/demo_image_gaimen.jpg)
![Woody Paige Corinthinan Filter](docs/images/demo_image_woody.jpg)
![Trump Corinthinan Filter](docs/images/demo_image_trump.jpg)
![Charlize Corinthinan Filter](docs/images/demo_image_charlize.jpg)
![E. Banks Filter](docs/images/demo_image_ebanks.jpg)

Watch it in action:

https://youtu.be/JPhnuYbPG1s


Known bugs:

+ [ ] Sideways faces fill in wrong