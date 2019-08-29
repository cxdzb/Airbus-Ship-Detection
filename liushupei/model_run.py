from liushupei.preprocess.data_generator import generator
from liushupei.model.model_assess import draw_loss
import pandas as pd
import os

imgs = os.listdir(r"E:\DataSet\airbus-ship-detection\ship")
results = pd.read_csv(r"E:\DataSet\airbus-ship-detection\segmentations.csv")

train_gen = generator(imgs, results, 2)
val_gen = generator(imgs, results, 2)

from liushupei.model.model_build import Unet

model = Unet(input_shape=(768, 768, 3))

history = model.fit_generator(train_gen, steps_per_epoch=128, epochs=3,\
                              validation_data=val_gen,validation_steps=1).history
model.save("2_128_3.h5")

draw_loss(history["loss"], history["val_loss"])
