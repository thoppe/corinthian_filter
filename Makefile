target = qAEM95ixp-s

all:
	make pull
	make extract
	make corinthian
	make movie_corinthian

pull:
	mkdir -p source
	youtube-dl $(target) -o source/$(target).mp4

extract:
	python extract_frames.py $(target)

corinthian:
	python corinthian.py --URI $(target) --stable

movie_corinthian:
	ffmpeg -thread_queue_size 512 -y -framerate 30 -i data/$(target)/corinthian/%06d.jpg -i source/audio/$(target).mp4 -acodec copy -c:v libx264 -r 30 -pix_fmt yuv420p -shortest -map 0:v:0 -map 1:a:0 -crf 23 data/$(target)_corinthian.mp4

movie_no_sound_corinthian:
	ffmpeg -thread_queue_size 512 -y -framerate 30 -i data/$(target)/corinthian/%06d.jpg -c:v libx264 -r 30 -pix_fmt yuv420p -crf 23 data/$(target)_corinthian.mp4

test:
	python corinthian.py problem_frames/005178.jpg
