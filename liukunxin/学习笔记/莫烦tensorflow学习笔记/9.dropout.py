import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import os

#load data
digits=load_digits()
X=digits.data
y=LabelBinarizer().fit_transform(digits.target)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.3)

def add_layer(x,in_size,out_size,lname,AF=None):
    A=tf.Variable(tf.random_normal([in_size,out_size]))
    B=tf.Variable(tf.zeros([1,out_size])+0.1)
    Ax_plus_B=tf.add(tf.matmul(x,A),B)
    # 这里删掉神经结点，避免过拟合overfitting
    Ax_plus_B=tf.nn.dropout(Ax_plus_B,keep_prob)
    if AF is None:
        outputs=Ax_plus_B
    else:
        outputs=AF(Ax_plus_B)
    tf.summary.histogram(lname+'/outputs',outputs)
    return outputs

keep_prob=tf.placeholder(tf.float32) # 神经结点保留的概率
x=tf.placeholder(tf.float32,[None,64])
y=tf.placeholder(tf.float32,[None,10])

l1=add_layer(x,64,100,'l1',AF=tf.nn.tanh)
prediction=add_layer(l1,100,10,'l2',AF=tf.nn.softmax)

# loss
cross_entropy=tf.reduce_mean(-tf.reduce_sum(y*tf.log(prediction),
    reduction_indices=[1]))
tf.summary.scalar('loss',cross_entropy)
train_step=tf.train.GradientDescentOptimizer(0.6).minimize(cross_entropy)

path=r'C:\Users\liu\Desktop\logs'
with tf.Session() as ss:
    merged=tf.summary.merge_all()
    train_writer=tf.summary.FileWriter(os.path.join(path,'train'),ss.graph)
    test_writer=tf.summary.FileWriter(os.path.join(path,'test'),ss.graph)

    ss.run(tf.initialize_all_variables())

    for i in range(1000):
        ss.run(train_step,feed_dict={x:X_train,y:y_train,keep_prob:0.5}) # 保留50%的神经结点
        if i%50==0:
            train_result=ss.run(merged,feed_dict={x:X_train,y:y_train,keep_prob:1}) # 验证时全部保留
            test_result=ss.run(merged,feed_dict={x:X_test,y:y_test,keep_prob:1})
            train_writer.add_summary(train_result,i)
            test_writer.add_summary(test_result,i)

