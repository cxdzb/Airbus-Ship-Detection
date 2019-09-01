from liushupei.preprocess.data_generator import generator
from liushupei.model.model_assess import draw_loss
from liushupei.model.model_build import Unet
import pandas as pd
import os

imgs = os.listdir(r"E:\DataSet\airbus-ship-detection\balance_train_data")
results = pd.read_csv(r"E:\DataSet\airbus-ship-detection\segmentations.csv")

train_gen = generator(imgs, results, 4)
val_gen = generator(imgs, results, 4, seed=1)

model = Unet(input_shape=(768, 768, 3))
history = model.fit_generator(train_gen, steps_per_epoch=128, epochs=5, \
                              validation_data=val_gen, validation_steps=10).history
model.save("4_128_5.h5")

draw_loss(history["loss"], history["val_loss"])
