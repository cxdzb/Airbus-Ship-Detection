from PIL import Image
import numpy as np


def rle_to_array(img, rles):
    width, height = img.size
    img_copy = Image.new('L', (width, height))
    pixdata = img_copy.load()

    for rle in rles.values:
        rle = rle.split(' ')
        masks = []
        for i in range(0, len(rle), 2):
            for j in range(int(rle[i + 1])):
                masks.append(int(rle[i]) + j)

        index = 0
        for i in range(width):
            for j in range(height):
                if index < len(masks) and i * height + j == masks[index]:
                    pixdata[i, j] = 1
                    index += 1
                elif pixdata[i, j] != 1:
                    pixdata[i, j] = 0

    data1, data2 = img.getdata(), img_copy.getdata()
    data1, data2 = np.asarray(data1).reshape((width, height, 3)), np.asarray(data2).reshape((width, height, 1))
    return [data1, data2]


def array_to_image(array, shape):
    img = Image.new("L", (shape[0], shape[1]))
    pixdata = img.load()
    for i in range(shape[0]):
        for j in range(shape[1]):
            pixdata[i, j] = tuple(array[j][i])

    img.show()
