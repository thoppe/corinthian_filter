target = "IH8K0bPc-BE"

all:
	mkdir -p source
	youtube-dl $(target) -o source/$(target).mp4
	python extract_frames.py $(target)
	python corinthian.py --URI $(target)
	make movie_corinthian

movie_corinthian:
	ffmpeg -thread_queue_size 512 -y -framerate 30 -i data/$(target)/corinthian/%06d.jpg -i source/audio/$(target).mp4 -acodec copy -c:v libx264 -r 30 -pix_fmt yuv420p -shortest -map 0:v:0 -map 1:a:0 -crf 23 data/$(target)_corinthian.mp4

movie_evil_eyes:
	ffmpeg -thread_queue_size 512 -y -framerate 30 -i data/$(target)/evil_eyes/%06d.jpg -i source/audio/$(target).mp4 -acodec copy -c:v libx264 -r 30 -pix_fmt yuv420p -shortest -map 0:v:0 -map 1:a:0 -crf 23 data/$(target)_evil_eyes.mp4

