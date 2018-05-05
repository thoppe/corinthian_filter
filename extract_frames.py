import os, sys

f_movie = sys.argv[1]

URI = os.path.basename(f_movie).split('-')[-1].split('.')[0]

save_dest = 'source/frames/{}'.format(URI)
os.system('mkdir -p {}'.format(save_dest))

cmd = '''avconv -threads auto -y -r 20 -an -q:v 1 {dest}/%06d.jpg -i "{f_movie}"'''
cmd = cmd.format(dest=save_dest, f_movie=f_movie)
print cmd

os.system(cmd)

