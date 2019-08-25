from liushupei.preprocess.data_generator import generator
import pandas as pd
import os

# filenames = os.listdir(os.getcwd() + "\\data")[1:2]
# results = pd.read_csv(os.getcwd() + "\\data\\data.csv")
#
# X, Y = [], []
# for filename in filenames:
#     img = Image.open(os.getcwd() + "\\data\\" + filename)
#     mask = results["EncodedPixels"][results["ImageId"] == filename].values[0]
#     x, y = rle_to_array(img, mask)
#     X.append(x)
#     Y.append(y)
# X,Y=np.asarray(X),np.asarray(Y)
# from sklearn.model_selection import train_test_split as tts
#
# x_train, x_test, y_train, y_test = tts(X, Y, test_size=0.2, random_state=1)

imgs = os.listdir(r"E:\DataSet\airbus-ship-detection\ship")
results = pd.read_csv(r"E:\DataSet\airbus-ship-detection\segmentations.csv")

train_gen = generator(imgs, results, 5)
val_gen = generator(imgs, results, 5)

from liushupei.model.model_build import Unet

model = Unet(input_shape=(768, 768, 3))

model.fit_generator(train_gen, steps_per_epoch=32, epochs=5, validation_data=val_gen, validation_steps=1)
model.save("5_32_5.h5")

from liushupei.preprocess.data_deal import rle_to_array, array_to_image
from PIL import Image

x, y_true = rle_to_array(Image.open(os.getcwd() + "\\data\\0af984fcd.jpg"),
                         results["EncodedPixels"][results["ImageId"] == "0af984fcd.jpg"])
x = x.reshape(1, 768, 768, 3) / 255
y_pre = model.predict(x)[0]
print(y_true.shape)
array_to_image(y_true * 255, (768, 768))
array_to_image(y_pre * 255, (768, 768))
