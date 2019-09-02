#@Time   :2019/9/2 10:19
#@author : qtgavc

import numpy as np
import pandas as pd
import seaborn as sns
from skimage.data import imread
import matplotlib.pyplot as plt

import os
PATH = "E:\\airbus-ship-detection"
print(os.listdir(PATH))

# number of train sets
train = os.listdir(PATH+'\\train_v2')
print(len(train))

# print first ten data
masks = pd.read_csv(PATH + '/train_ship_segmentations_v2.csv')
masks.head(10)

# Number of images with ships and without ships
n_im_no_ships = len(masks[masks['EncodedPixels'].isna()]['ImageId'].unique())
n_im_ships = len(masks[~masks['EncodedPixels'].isna()]['ImageId'].unique())
sns.barplot(x=['Ships', 'No ships'], y=[n_im_ships, n_im_no_ships])

# Distribution of number of ships in images
df_tmp = masks[~masks['EncodedPixels'].isna()]
sns.distplot(df_tmp['ImageId'].value_counts().values, kde=False)

def rle_decode(mask_rle, shape=(768, 768)):

    # mask_rle: run-length as string formated (start length)
    # shape: (height,width) of array to return
    # Returns numpy array, 1 - mask, 0 - background

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T

ImageId = '0006c52e8.jpg'

img = imread(PATH + '\\train_v2\\'+ ImageId)
img_masks = masks.loc[masks['ImageId'] == ImageId, 'EncodedPixels'].tolist()

# Take the individual ship masks and create a single mask array for all ships
all_masks = np.zeros((768, 768))
for mask in img_masks:
    all_masks += rle_decode(mask)

fig, axarr = plt.subplots(1, 3, figsize=(15, 40))
axarr[0].axis('off')
axarr[1].axis('off')
axarr[2].axis('off')
axarr[0].imshow(img)
axarr[1].imshow(all_masks)
axarr[2].imshow(img)
axarr[2].imshow(all_masks, alpha=0.4)
plt.tight_layout(h_pad=0.1, w_pad=0.1)
plt.show()