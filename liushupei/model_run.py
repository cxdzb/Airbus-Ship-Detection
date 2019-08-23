from preprocess.data_deal import rle_to_mask
from model.model_build import Unet
from PIL import Image
import pandas as pd
import numpy as np
import os

filenames = os.listdir(os.getcwd() + "\\data")
results = pd.read_csv(os.getcwd() + "\\data\\data.csv")

X, Y = [], []
for filename in filenames:
    img = Image.open(os.getcwd() + "\\data\\" + filename)
    print(filename)
    mask = results["EncodedPixels"][results["ImageId"] == filename].values[0]
    x, y = rle_to_mask(img, mask)
    X.append(x)
    Y.append(y)
X,Y=np.asarray(X),np.asarray(Y)
print(Y)
