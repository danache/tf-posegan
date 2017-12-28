import numpy as np
# import cv2
import os
import matplotlib.pyplot as plt
import random
import time
from skimage import transform
import scipy.misc as scm
import tensorflow as tf
import pandas as pd
import tensorlayer as tl
from tools.img_tf import *
from dataGenerator.datagen_v3 import DataGenerator



test = DataGenerator(imgdir="/media/bnrc2/_backup/ai/ai_challenger_keypoint_train_20170902/keypoint_train_images_20170902/",
                     nstack= 2,label_dir="/media/bnrc2/_backup/ai/ai_challenger_keypoint_train_20170902/keypoint_train_annotations_20170909.json",
                               out_record="/media/bnrc2/_backup/dataset/new_tfrecord/valid/",num_txt="/media/bnrc2/_backup/dataset/new_tfrecord/new_valid.txt",
                               batch_size=8, name="train_mini", is_aug=False,isvalid=False,refine_num=10)



img = test.getData()
sess = tf.Session()
init = tf.group(tf.global_variables_initializer(),
                tf.local_variables_initializer())

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord, sess=sess)

sess.run(init)
for i in range(2):
    a = sess.run([img])
print(a)
coord.request_stop()
#
#     # Wait for threads to finish.
coord.join(threads)
sess.close()





