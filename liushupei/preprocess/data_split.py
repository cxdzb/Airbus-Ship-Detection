import os, shutil, pandas as pd, numpy as np


def split_empty():
    src = r"E:\DataSet\airbus-ship-detection\train_data"
    dst1 = r"E:\DataSet\airbus-ship-detection\no_ship"
    dst2 = r"E:\DataSet\airbus-ship-detection\ship"
    table = pd.read_csv(r"E:\DataSet\airbus-ship-detection\segmentations.csv")
    ImageId, EncodedPixels = table["ImageId"], table["EncodedPixels"]
    imgs = os.listdir(src)
    index = 1
    for i in imgs:
        print(index)
        index += 1
        if EncodedPixels[ImageId == i].values[0] is np.nan:
            shutil.copy(src + '\\' + i, dst1 + '\\' + i)
        else:
            shutil.copy(src + '\\' + i, dst2 + '\\' + i)
    os.system("shutdown -s -t 300")
