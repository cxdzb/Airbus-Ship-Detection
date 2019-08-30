from liushupei.preprocess.data_deal import rle_to_array
import numpy as np
from PIL import Image


def generator(imgs, results, batch_size, seed=None):
    if seed:
        np.random.seed(seed)
    ImageId, EncodedPixels = results["ImageId"], results["EncodedPixels"]
    while True:
        samples = np.random.choice(imgs, size=batch_size)
        X, Y = [], []
        for s in samples:
            img = Image.open(r"E:\DataSet\airbus-ship-detection\ship" + "\\" + s)
            rles = EncodedPixels[ImageId == s]
            x, y = rle_to_array(img, rles)
            X.append(x)
            Y.append(y)
        X, Y = np.asarray(X) / 255, np.asarray(Y)
        yield (X, Y)

# import pandas as pd
# import os
#
# imgs = os.listdir(r"E:\DataSet\airbus-ship-detection\ship")
# results = pd.read_csv(r"E:\DataSet\airbus-ship-detection\segmentations.csv")
# gen=generator(imgs, results, 5)
# for i in gen:
#     continue
