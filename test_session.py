import numpy as np
# import cv2


import configparser
from hg_models.hg import *

def process_network(conf_file):
    params = {}
    config = configparser.ConfigParser()
    config.read(conf_file)
    for section in config.sections():

        for option in config.options(section):
            params[option] = eval(config.get(section, option))
    return params

network_params = process_network("./config/hourglass.cfg")

#
# test = DataGenerator(imgdir="/media/bnrc2/_backup/ai/ai_challenger_keypoint_train_20170902/keypoint_train_images_20170902/",
#                      nstack= 2,label_dir="/media/bnrc2/_backup/ai/ai_challenger_keypoint_train_20170902/keypoint_train_annotations_20170909.json",
#                                out_record="/media/bnrc2/_backup/dataset/new_tf_mini/",num_txt="/media/bnrc2/_backup/dataset/new_tf_mini/new_train_mini.txt",
#                                batch_size=8, name="train_mini", is_aug=False,isvalid=False)
#
# train_img, train_mini, train_heatmap, train_center, train_scale, train_name = test.getData()
train_img = tf.placeholder(dtype=tf.float32,shape=[None,256,256,3])
hg = hgmodel(nStack=network_params['nstack'],nModules=network_params['nmodules'])
out = hg.createModel(inputs=train_img,reuse=False).outputs


sess = tf.Session()
init = tf.group(tf.global_variables_initializer(),
                tf.local_variables_initializer())

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord, sess=sess)

sess.run(init)
a = sess.run(out)

    # train_predictions = dict()
    # train_predictions['image_ids'] = []
    # train_predictions['annos'] = dict()
    # train_predictions = getjointcoord(train_c, name, train_predictions)
    #
    # train_return_dict = dict()
    # train_return_dict['error'] = None
    # train_return_dict['warning'] = []
    # train_return_dict['score'] = None
    # train_anno = load_annotations(
    #     "/media/bnrc2/_backup/ai/ai_challenger_keypoint_train_20170902/keypoint_train_annotations_20170909.json",
    #     train_return_dict)
    # train_score = getScore(train_predictions, train_anno, train_return_dict)
    #
    # test_predictions = dict()
    # test_predictions['image_ids'] = []
    # test_predictions['annos'] = dict()
    # test_predictions = getjointcoord(test_c, name, test_predictions)
    #
    # test_return_dict = dict()
    # test_return_dict['error'] = None
    # test_return_dict['warning'] = []
    # test_return_dict['score'] = None
    # test_anno = load_annotations(
    #     "/media/bnrc2/_backup/ai/ai_challenger_keypoint_train_20170902/keypoint_train_annotations_20170909.json",
    #     train_return_dict)
    # test_score = getScore(test_predictions, test_anno, test_return_dict)
    # print("train score = %f ,test score = %f" % (train_score,test_score))

coord.request_stop()
#
#     # Wait for threads to finish.
coord.join(threads)
sess.close()





