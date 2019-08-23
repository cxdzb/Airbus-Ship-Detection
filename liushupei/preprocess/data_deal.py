from PIL import Image


def rle_to_mask(img, mask):
    width, height = img.size
    img_copy = Image.new('L', (width, height))
    data = img_copy.load()

    mask = mask.split(' ')
    masks = []
    for i in range(0, len(mask), 2):
        for j in range(int(mask[i + 1])):
            masks.append(int(mask[i]) + j)

    index = 0
    for i in range(width):
        for j in range(height):
            if index < len(masks) and i * height + j == masks[index]:
                data[i, j] = 1
                index += 1
            else:
                data[i, j] = 0

    return [img,img_copy]
