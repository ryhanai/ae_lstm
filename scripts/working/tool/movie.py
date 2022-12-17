import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

episode_dir = '220831/procedural/4'
nframes = 54

cimgs = []
for i in range(nframes):
    img = np.array(Image.open(os.path.join(episode_dir, 'image_frame%05d.jpg'%i)))
    fimg = np.array(Image.open(os.path.join(episode_dir, 'fimage_frame%05d.jpg'%i)))
    cimg = np.concatenate([img, fimg], axis=0)
    cimgs.append(Image.fromarray(cimg))

cimgs[0].save(os.path.join(episode_dir, 'movie.gif'), save_all=True, append_images=cimgs[1:], optimize=False, duration=100, loop=0)
