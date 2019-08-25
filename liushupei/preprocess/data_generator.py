from ..preprocess.data_deal import rle_to_array
import numpy as np
from PIL import Image


def generator(imgs, results, batch_size):
    ImageId, EncodedPixels = results["ImageId"], results["EncodedPixels"]
    while True:
        samples = np.random.choice(imgs, size=batch_size)
        X, Y = [], []
        for s in samples:
            img = Image.open(r"E:\DataSet\airbus-ship-detection\ship" + "\\" + s)
            mask = EncodedPixels[ImageId == s]
            x, y = rle_to_array(img, mask)
            X.append(x)
            Y.append(y)
        X, Y = np.asarray(X)/255, np.asarray(Y)
        yield (X, Y)
