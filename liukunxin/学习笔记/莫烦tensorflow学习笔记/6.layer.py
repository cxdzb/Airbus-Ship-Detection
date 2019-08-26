import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(inputs,in_size,out_size,AF=None): # 添加网络层
    weight=tf.Variable(tf.random_normal([in_size,out_size])) # 一个矩阵
    biases=tf.Variable(tf.zeros([1,out_size])+0.1) # 一行多列的不为0的偏移量
    wx_plus_b=tf.matmul(inputs,weight)+biases
    if AF is None:
        outputs=wx_plus_b
    else:
        outputs=AF(wx_plus_b)
    return outputs

x_data=np.linspace(-1,1,300)[:,np.newaxis] # 300行1列
noise=np.random.normal(0,0.05,x_data.shape) # 加噪
y_data=np.square(x_data)-0.5+noise

xs=tf.placeholder(tf.float32,[None,1])
ys=tf.placeholder(tf.float32,[None,1])
l1=add_layer(xs,1,10,AF=tf.nn.relu) # 输入层到隐藏层
prediction=add_layer(l1,10,1,AF=None) # 隐藏层到输出层

loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),
    reduction_indices=[1]))
train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)

fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
plt.ion() # 使得plt.show()后程序不会因此暂停
plt.show()

init=tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        sess.run(train_step,feed_dict={xs:x_data, ys:y_data})
        if i%50==0:
            print(sess.run(loss,feed_dict={xs:x_data, ys:y_data}))
            try:
                ax.lines.remove(lines[0]) # 删除上一次绘制的曲线
            except:
                pass
            prediction_val=sess.run(prediction,feed_dict={xs:x_data})
            lines=ax.plot(x_data,prediction_val,'r-',lw=5)
            plt.pause(0.1)