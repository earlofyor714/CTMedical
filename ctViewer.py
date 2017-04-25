import numpy as np
import matplotlib.pyplot as plt

DIR = '.'

with np.load('data/ct-medical-image-analysis-tutorial/full_archive.npz') as img_data:
    full_image_dict = dict(zip(img_data['idx'], img_data['image']))

print("{}".format(len(full_image_dict)))
plt.matshow(full_image_dict[588])
plt.show()

# 20% = 117.6
