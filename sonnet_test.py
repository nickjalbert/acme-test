import tensorflow as tf
import sonnet as snt

batch_size = 10
input_size = 30

mlp = snt.Sequential([
    snt.Linear(1024),
    tf.nn.relu,
    snt.Linear(20),
])

logits = mlp(tf.random.normal([batch_size, input_size]))

import pdb
pdb.set_trace()
