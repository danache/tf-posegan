import sys
sys.path.append(sys.path[0])
del sys.path[0]
import tensorflow as tf

from hg_models.discriminator import discrim
import numpy as np
import tensorlayer as tl

with tf.Session() as sess:
    a = tf.placeholder(tf.float32, shape=[None,64,64,256])
    init = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())

    hg = discrim(nFeat=256,  nModules=2, outputDim=14,npool=4).createModel(a).outputs

    train_writer = tf.summary.FileWriter("./log", sess.graph)
    sess.run(init)
    tl.layers.initialize_global_variables(sess)
    _ = sess.run(hg,feed_dict={a:np.ones([8,64,64,256])})
    train_writer.flush()

