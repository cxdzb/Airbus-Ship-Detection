import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 在kaggle上跑要加上这行代码，清理之前的模型
tf.reset_default_graph()

mnist=input_data.read_data_sets('MNIST.data',one_hot=True)

lr=0.001 # learning rate
training_iters=100000 # 迭代次数
batch_size=128 # batch 大小

n_inputs=28 # 28列
n_steps=28 # 28行，RNN中以行作为时间轴
n_hidden_units=128 # 隐藏层神经元数量
n_classes=10 # 分类的数目

x=tf.placeholder(tf.float32,[None,n_steps,n_inputs])
y=tf.placeholder(tf.float32,[None,n_classes])

weights={
    'in':tf.Variable(tf.random_normal([n_inputs,n_hidden_units])),
    'out':tf.Variable(tf.random_normal([n_hidden_units,n_classes])),
}
biases={
    'in':tf.Variable(tf.constant(0.1,shape=[n_hidden_units])),
    'out':tf.Variable(tf.constant(0.1,shape=[n_classes])),
}

def RNN(X,weights,biases):
    # hidden layer for input to cell
    # X:batch_size*n_steps*n_inputs
    X=tf.reshape(X,[-1,n_inputs])
    # --> batch_size*n_steps*n_hidden_units
    X_in=tf.matmul(X,weights['in'])+biases['in']
    X_in=tf.reshape(X_in,[-1,n_steps,n_hidden_units])

    # cell
    lstm_cell=tf.nn.rnn_cell.BasicLSTMCell(
        n_hidden_units,
        forget_bias=1.0,
        state_is_tuple=True
    )
    # state is a tuple (c_state,m_state) 分别代表支线和主线记忆
    _init_state=lstm_cell.zero_state(batch_size,dtype=tf.float32)
    outputs,states=tf.nn.dynamic_rnn(
        lstm_cell,
        X_in,
        initial_state=_init_state,
        time_major=False # True表示从第一个维度开始，这里第一个维度时batch_size，第二个维度才是step，应该选False
    )

    # hidden layer for outputs as the final results
    result=tf.matmul(states[1],weights['out'])+biases['out'] # state[1]表示主线结果m_state

    return result


pred=RNN(x,weights,biases)
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=pred,
    labels=y
))
train_op=tf.train.AdamOptimizer(lr).minimize(loss)

correct_pred=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))

with tf.Session() as ss:
    ss.run(tf.initialize_all_variables())
    step=0
    while step*batch_size<training_iters:
        batch_x,batch_y=mnist.train.next_batch(batch_size)
        batch_x=batch_x.reshape([batch_size,n_steps,n_inputs])
        ss.run([train_op],feed_dict={
            x:batch_x,
            y:batch_y,
        })
        if step%20==0:
            print(ss.run(accuracy,feed_dict={
                x:batch_x,
                y:batch_y,
            }))
        step+=1

