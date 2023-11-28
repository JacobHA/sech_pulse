import os
import imageio.v2 as imageio

images = os.listdir('frames')
# sort by number
images = sorted(images, key=lambda x: int(x[5:-4]))
images = [imageio.imread(f'frames/{i}') for i in images]

# imageio.mimsave('bloch.gif', images, fps=20)
# save with looping:
imageio.mimsave('bloch.gif', images, fps=15, loop=0)