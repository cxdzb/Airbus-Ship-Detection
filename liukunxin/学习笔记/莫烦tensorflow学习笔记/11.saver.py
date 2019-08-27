import tensorflow as tf

A=tf.Variable([[1,2,3],[3,4,5]],dtype=tf.float32,name='weight')
B=tf.Variable([[1,2,3]],dtype=tf.float32,name='biases')

init=tf.initialize_all_variables()

# 保存
with tf.Session() as ss:
    ss.run(init)
    save_path=tf.train.Saver().save(ss,r'C:\Users\liu\Desktop\saver\save_net.ckpt')
    print(save_path)

# 加载
with tf.Session() as ss:
    tf.train.Saver().restore(ss,r'C:\Users\liu\Desktop\saver\save_net.ckpt')
    print(ss.run(A))
    print(ss.run(B))    