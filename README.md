# Corinthian Filter

Watch it in action: https://youtu.be/JPhnuYbPG1s

![Obama Corinthinan Filter](docs/images/demo_image_obama.jpg)
![Gaimen Corinthinan Filter](docs/images/demo_image_gaimen.jpg)
![Woody Paige Corinthinan Filter](docs/images/demo_image_woody.jpg)
![Trump Corinthinan Filter](docs/images/demo_image_trump.jpg)
![Charlize Corinthinan Filter](docs/images/demo_image_charlize.jpg)
![E. Banks Filter](docs/images/demo_image_ebanks.jpg)

With everything installed (good luck on that!), change the target in the Makefile to a valid youtube-URI and run `make corinthian`

#### Developer Notes:

Version 0.2

If it's very slow, check that dlib uses CUDA properly.

+ [ ] Even using a CNN over a HOG, sometimes an obvious face isn't found (upsample?)
+ [ ] Motion stabilization with low-rank PCA over time-series should be possible