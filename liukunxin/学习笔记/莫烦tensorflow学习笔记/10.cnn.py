import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets('MNIST.data',one_hot=True)

def compute_accuracy(vx,vy):
    global prediction
    y_pre=ss.run(prediction,feed_dict={x:vx,keep_prob:1})
    correct_prediction=tf.equal(tf.argmax(y_pre,1),tf.argmax(vy,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    res=ss.run(accuracy,feed_dict={x:vx,y:vy,keep_prob:1})
    return res

def weight_variable(shape):
    init=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(init)

def bias_variable(shape):
    init=tf.constant(0.1,shape=shape)
    return tf.Variable(init)

def conv2d(x,A):
    # stride:[1,x_move,y_move,1]
    # 'SAME'比原图片一样，'VALID'比原图片小
    return tf.nn.conv2d(x,A,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

x=tf.placeholder(tf.float32,[None,784])
y=tf.placeholder(tf.float32,[None,10])
keep_prob=tf.placeholder(tf.float32)
x_image=tf.reshape(x,[-1,28,28,1]) # nx28x28x1

# conv1 layer
A=weight_variable([5,5,1,32]) # patch 5x5x1, out size 5x5x32
B=bias_variable([32])
conv1=tf.nn.relu(conv2d(x_image,A)+B) # out size: 28x28x32
pool1=max_pool_2x2(conv1) # out size: 14x14x32
# conv2 layer
A=weight_variable([5,5,32,64]) # patch 5x5x32, out size 5x5x64
B=bias_variable([64])
conv2=tf.nn.relu(conv2d(pool1,A)+B) # out size: 14x14x64
pool2=max_pool_2x2(conv2) # out size: 7x7x64
# func1 layer
A=weight_variable([7*7*64,1024]) # 7*7*64 -> 1x1024
B=bias_variable([1024])
pool2_flat=tf.reshape(pool2,[-1,7*7*64]) # [n_samples,7,7,64] -> [n_samples,7*7*64]
f1=tf.nn.relu(tf.matmul(pool2_flat,A)+B)
f1=tf.nn.dropout(f1,keep_prob)
# fuc2 layer
A=weight_variable([1024,10])
B=bias_variable([10])
prediction=tf.nn.softmax(tf.matmul(f1,A)+B)

cross_entropy=tf.reduce_mean(-tf.reduce_sum(y*tf.log(prediction),
    reduction_indices=[1]))
train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

with tf.Session() as ss:
    ss.run(tf.initialize_all_variables())

    for i in range(1000):
        batch_x, batch_y=mnist.train.next_batch(100)
        ss.run(train_step,feed_dict={x:batch_x,y:batch_y,keep_prob:0.5})
        if i%50==0:
            print(compute_accuracy(mnist.test.images,mnist.test.labels))




