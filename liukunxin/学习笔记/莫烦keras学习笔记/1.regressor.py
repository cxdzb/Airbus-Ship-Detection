import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

X=np.linspace(-1,1,200)
Y=0.5*X+2+np.random.normal(0,0.05,(200,))
plt.scatter(X,Y)
plt.show()

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2)

model=Sequential()
model.add(Dense(
    output_dim=1,
    input_dim=1
))
model.compile(
    loss='mse', # mean(square())
    optimizer='sgd'
)

for i in range(1000):
    cost=model.train_on_batch(x_train,y_train)
    if i%50==0:
        print(cost)

print()
cost=model.evaluate(x_test,y_test,batch_size=40)
print(cost)
w,b=model.layers[0].get_weights()
print(w,b)

y_pre=model.predict(x_test)
plt.scatter(x_test,y_test)
plt.plot(x_test,y_pre)
plt.show()
