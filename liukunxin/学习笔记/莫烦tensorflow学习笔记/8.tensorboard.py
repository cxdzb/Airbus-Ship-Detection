# -*- coding: utf-8 -*-  

import tensorflow as tf
import numpy as np

def add_layer(x,insz,outsz,n_layer,AF=None):
    layer_name='layer%s'%n_layer
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            a=tf.Variable(tf.random_normal([insz,outsz]),name='W')
            tf.summary.histogram(layer_name+'/weight',a)
        with tf.name_scope('biases'):
            b=tf.Variable(tf.zeros([1,outsz])+0.1,name='biases')
            tf.summary.histogram(layer_name+'/biases',b)
        with tf.name_scope('Wx_plus_b'):
            ax_b=tf.matmul(x,a)+b
        if AF is None:
            outputs=ax_b
        else:
            outputs=AF(ax_b)
        tf.summary.histogram(layer_name+'/outputs',outputs)
        return outputs

x_data=np.linspace(-1,1,300)[:,np.newaxis]
noise=np.random.normal(0,0.05,x_data.shape)
y_data=np.square(x_data)+noise

with tf.name_scope('inputs'):
    x=tf.placeholder(tf.float32,[None,1],name='x_input')
    y=tf.placeholder(tf.float32,[None,1],name='y_input')
l1=add_layer(x,1,10,1,tf.nn.relu)
predict=add_layer(l1,10,1,2)

with tf.name_scope('loss'):
    loss=tf.reduce_mean(tf.reduce_sum(tf.square(y-predict),
        reduction_indices=[1]))
    tf.summary.scalar('loss',loss)
with tf.name_scope('train'):
    train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init=tf.initialize_all_variables()
with tf.Session() as ss:
    merged=tf.summary.merge_all()
    writer=tf.summary.FileWriter(r'logs',ss.graph)
    ss.run(init)

    for i in range(1000):
        ss.run(train_step,feed_dict={x:x_data,y:y_data})
        if i%50==0:
            res=ss.run(merged,feed_dict={x:x_data,y:y_data})
            writer.add_summary(res,i)

# 查看方法：
# activate tensorflow
# tensorboard --logdir logs
# 注意：windows下最好直接放默认路径，事后再删掉，放其他地方很可能生成失败
# rd /s /q logs