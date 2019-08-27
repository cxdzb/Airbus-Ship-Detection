import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.optimizers import RMSprop

(x_train,y_train),(x_test,y_test)=mnist.load_data()

x_train=x_train.reshape(x_train.shape[0],-1)/255 # normalize
x_test=x_test.reshape(x_test.shape[0],-1)/255
y_train=np_utils.to_categorical(y_train,num_classes=10) # number to vector(10 size)
y_test=np_utils.to_categorical(y_test,num_classes=10)

model=Sequential([
    Dense(32,input_dim=784),
    Activation('relu'),
    Dense(10), # 默认input_dim=previous output_dim
    Activation('softmax'), # softmax用于分类
])

#rms=RMSprop()

model.compile(
    #optimizer=rms,
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

model.fit(x_train,y_train,batch_size=32,epochs=2)

loss,accuracy=model.evaluate(x_test,y_test)

print(loss,accuracy)
