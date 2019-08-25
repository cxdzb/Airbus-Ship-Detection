import tensorflow as tf

'''
initial_value=None, trainable=None, collections=None,
validate_shape=True, caching_device=None, name=None, variable_def=None,
dtype=None, expected_shape=None, import_scope=None, constraint=None,
use_resource=None, synchronization=VariableSynchronization.AUTO,
aggregation=VariableAggregation.NONE, shape=None)
'''
state=tf.Variable(0,name='cnt')
#print(state.name)
one=tf.constant(1)

new_value=tf.add(state,one)
update=tf.assign(state,new_value) # state=new_value=state+one

init=tf.initialize_all_variables() # 初始化这步千万不要忘记

with tf.Session() as sess:
    sess.run(init)
    for i in range(3):
        sess.run(update)
        print(sess.run(state))