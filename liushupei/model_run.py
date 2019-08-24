from liushupei.preprocess.data_deal import rle_to_array,array_to_image
from PIL import Image
import pandas as pd
import numpy as np
import os

filenames = os.listdir(os.getcwd() + "\\data")[1:2]
results = pd.read_csv(os.getcwd() + "\\data\\data.csv")

X, Y = [], []
for filename in filenames:
    img = Image.open(os.getcwd() + "\\data\\" + filename)
    mask = results["EncodedPixels"][results["ImageId"] == filename].values[0]
    x, y = rle_to_array(img, mask)
    X.append(x)
    Y.append(y)
X,Y=np.asarray(X),np.asarray(Y)

from sklearn.model_selection import train_test_split as tts

x_train,x_test,y_train,y_test=tts(X,Y,test_size=0.2,random_state=1)

from liushupei.model.model_build import Unet

model=Unet(input_shape=(X[0].shape))

model.fit(x_train,y_train,batch_size=4,epochs=10,validation_data=(x_test,y_test))