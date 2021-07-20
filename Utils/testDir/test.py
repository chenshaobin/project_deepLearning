import Utils
import os
import numpy as np
sonPath = ('a', 'b', 'c')
fatherPath = os.path.join(* sonPath)
print(fatherPath)

img = np.array([255,0,128,128,255,0]).reshape((2,3))
print('original img:', img)
new_img = np.ones_like(img)
# print(img > 128)
new_img[img > 128] = 0
new_img[img < 128] = 2
print('new img:', new_img)