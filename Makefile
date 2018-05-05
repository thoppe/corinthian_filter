target = "o3ujLxQP8hE"

all:
	mkdir -p source
	youtube-dl $(target) -o source/$(target).mp4
	python extract_frames.py $(target)
	python find_landmarks.py $(target)
