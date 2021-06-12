import tensorflow as tf
import sonnet as snt

print(tf.reduce_sum(tf.random.normal([1000, 1000])))
print("TensorFlow version {}".format(tf.__version__))
print("Sonnet version {}".format(snt.__version__))
