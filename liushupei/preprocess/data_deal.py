from PIL import Image
import numpy as np


def array_to_image(array, shape):
    img = Image.new("L", (shape[0], shape[1]))
    pixdata = img.load()
    for i in range(shape[0]):
        for j in range(shape[1]):
            pixdata[i, j] = tuple(array[j][i])

    img.show()


def rle_to_array(img, rles):
    x = np.asarray(img).reshape((768, 768, 3))
    y = np.zeros(768 * 768, dtype=np.uint8)

    for rle in rles.values:
        rle = rle.split(' ')
        starts, lengths = [np.asarray(x, dtype=int) for x in (rle[0:][::2], rle[1:][::2])]
        starts -= 1
        ends = starts + lengths
        for s, e in zip(starts, ends):
            y[s:e] = 1

    y = y.reshape((768, 768)).T.reshape((768, 768, 1))
    return x, y

# import pandas as pd
#
# results = pd.read_csv(r"E:\DataSet\airbus-ship-detection\segmentations.csv")
# img = Image.open(r"E:\DataSet\airbus-ship-detection\ship\0aea263bb.jpg")
# mask = results["EncodedPixels"][results["ImageId"] == "0aea263bb.jpg"]
# rle_to_array(img, mask)
