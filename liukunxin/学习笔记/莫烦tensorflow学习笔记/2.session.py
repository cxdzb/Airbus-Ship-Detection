import tensorflow as tf

mat1=tf.constant([[3,3]])
mat2=tf.constant([[2],[2]])

product=tf.matmul(mat1,mat2)

# method 1
sess=tf.Session()
res1=sess.run(product)
print(res1)
sess.close()

# method 2
with tf.Session() as sess:
    res2=sess.run(product)
    print(res2)