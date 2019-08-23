from keras import models
from keras.layers import *
from keras.optimizers import *
from keras import backend as K


def IoU(y_true, y_pred, eps=1e-6):
    if K.max(y_true) == 0.0:
        return IoU(1 - y_true, 1 - y_pred)
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3]) - intersection
    return -K.mean((intersection + eps) / (union + eps), axis=0)


def Unet(input_shape=(512, 512, 3)):
    input = Input(input_shape)
    c1 = Conv2D(8, (3, 3), activation='relu', padding='same')(input)
    c1 = Conv2D(8, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(16, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(16, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(32, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(32, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(64, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(64, (3, 3), activation='relu', padding='same')(c4)
    p4 = MaxPooling2D((2, 2))(c4)

    c5 = Conv2D(128, (3, 3), activation='relu', padding='same')(p4)
    c5 = Conv2D(128, (3, 3), activation='relu', padding='same')(c5)

    u1 = UpSampling2D((2, 2))(c5)
    c6 = Conv2D(64, (2, 2), activation='relu', padding='same')(u1)
    m1 = concatenate([c4, c6], axis=3)
    c6 = Conv2D(64, (3, 3), activation='relu', padding='same')(m1)
    c6 = Conv2D(64, (3, 3), activation='relu', padding='same')(c6)

    u2 = UpSampling2D((2, 2))(c6)
    c7 = Conv2D(32, (2, 2), activation='relu', padding='same')(u2)
    m2 = concatenate([c3, c7], axis=3)
    c7 = Conv2D(32, (3, 3), activation='relu', padding='same')(m2)
    c7 = Conv2D(32, (3, 3), activation='relu', padding='same')(c7)

    u3 = UpSampling2D((2, 2))(c7)
    c8 = Conv2D(16, (2, 2), activation='relu', padding='same')(u3)
    m3 = concatenate([c2, c8], axis=3)
    c8 = Conv2D(16, (3, 3), activation='relu', padding='same')(m3)
    c8 = Conv2D(16, (3, 3), activation='relu', padding='same')(c8)

    u4 = UpSampling2D((2, 2))(c8)
    c9 = Conv2D(8, (2, 2), activation='relu', padding='same')(u4)
    m4 = concatenate([c1, c9], axis=3)
    c9 = Conv2D(8, (3, 3), activation='relu', padding='same')(m4)
    c9 = Conv2D(8, (3, 3), activation='relu', padding='same')(c9)

    output = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = models.Model(inputs=[input], outputs=[output])
    model.compile(optimizer=Adam(lr=0.0001), loss=IoU, metrics=['binary_accuracy'])

    return model

# Unet()
