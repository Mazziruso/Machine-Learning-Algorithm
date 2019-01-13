import os
import numpy as np
from PIL import Image
import pickle

img = Image.open('olivettifaces.gif')
img_ndarray = np.asarray(img, dtype=np.float32)

olivettifaces = np.empty((400, 2679))
for row in range(20):
	for col in range(20):
		olivettifaces[row*20+col] = np.ndarray.flatten(img_ndarray[row*57:(row+1)*57, col*47:(col+1)*47])

olivettifaces_labels = np.asarray([int(i/10) for i in range(400)], dtype=np.uint8)

saver_images = open('olivettifaces_images.pkl', 'wb')
saver_labels = open('olivettifaces_labels.pkl', 'wb')
pickle.dump(olivettifaces, saver_images, -1)
pickle.dump(olivettifaces_labels, saver_labels, -1)
saver_images.close()
saver_labels.close()
